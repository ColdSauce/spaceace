"""Kinodynamic A* planner for SpaceAce.

This solver searches the actual simulator transition graph. A state contains
position, velocity, rotation, and the collected-pickup mask; edges are the six
real SpaceAce actions held for a short macro duration. The planner therefore
does not solve one pickup at a time. It searches for a complete physically
reachable trajectory, so a fast pickup touch that leaves the ship unrecoverable
is naturally bad because the remaining-pickup heuristic stays large.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import spaceace_rl

from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS

NUM_ACTIONS = len(ALL_ACTIONS)

# Physics constants mirror `src/real_physics.rs`.
_GRAVITY = 100.0
_THRUST_POWER = 400.0
_DT = 1.0 / 60.0

# Optimistic acceleration bounds for heuristic estimates. These are deliberately
# generous because the real transition model, not the heuristic, enforces
# rotation and thrust-direction constraints.
_A_X_MAX = _THRUST_POWER
_A_Y_DOWN_MAX = _THRUST_POWER + _GRAVITY
_A_Y_UP_MAX = _THRUST_POWER - _GRAVITY
_A_ANY_MAX = _A_Y_DOWN_MAX

# Effective 1D progress model used to turn static route length into time.
# This is intentionally about route progress through corridors, not raw ship
# acceleration in open air; the simulator remains the source of truth.
_ROUTE_ACCEL = 220.0
_ROUTE_VMAX = 420.0
_PICKUP_COLLECTION_RADIUS = 46.5


@dataclass
class _TraceEntry:
    parent_idx: int
    action_idx: int
    repeat: int
    g_frames: int


def _pickup_bits(pickup_states: list[bool]) -> int:
    bits = 0
    for i, collected in enumerate(pickup_states):
        if collected:
            bits |= 1 << i
    return bits


def _bang_bang_time(v_toward: float, a_max: float, d_abs: float) -> float:
    """Optimistic 1D double-integrator time bound in seconds."""
    if d_abs <= 0.0:
        return 0.0
    disc = v_toward * v_toward + 2.0 * a_max * d_abs
    return (math.sqrt(disc) - v_toward) / a_max


class AStarSolver:
    """Whole-level kinodynamic weighted A*.

    The search is "weighted" by default through ``heuristic_weight``. A weight
    above 1.0 intentionally trades proof of optimality for practical planning
    speed; the simulator still validates every returned action.
    """

    def __init__(
        self,
        env: SpaceAceDirectEnv,
        level: int,
        action_repeat: int = 10,
        pos_bucket: float = 16.0,
        vel_bucket: float = 16.0,
        rot_bucket_deg: float = 15.0,
        max_expansions: int = 200_000,
        time_limit_s: float = 60.0,
        heuristic_weight: float = 2.0,
        max_steps: int = 3000,
        verbose: bool = True,
    ) -> None:
        self._env = env
        self._action_repeat = max(1, int(action_repeat))
        self._repeat_options = self._build_repeat_options(self._action_repeat)
        self._pos_bucket = float(pos_bucket)
        self._vel_bucket = float(vel_bucket)
        self._rot_bucket = math.radians(rot_bucket_deg)
        self._rot_bins = max(1, int(round(2 * math.pi / self._rot_bucket)))
        self._max_expansions = int(max_expansions)
        self._time_limit = float(time_limit_s)
        self._hw = float(heuristic_weight)
        self._max_steps = int(max_steps)
        self._verbose = verbose

        self._pf = spaceace_rl.PyPathfinder(level, "grid")
        self._pickup_coords = list(self._pf.get_pickup_coords())
        self._total_pickups = len(self._pickup_coords)
        self._all_collected_mask = (1 << self._total_pickups) - 1

        self._path_pair = self._build_path_pair_matrix()
        self._route_tail = self._build_route_tail_table()

    def solve(self) -> list[int]:
        env = self._env
        env.reset()

        start_state = env.save_state()
        start_obs = env.get_observation()
        start_pickups = list(env.get_pickup_states())
        start_bits = _pickup_bits(start_pickups)

        if self._verbose:
            print(
                "\nA* planner (whole-level kinodynamic):\n"
                f"  action_repeat={self._action_repeat} "
                f"motion_primitives={self._repeat_options} "
                f"buckets=(pos={self._pos_bucket}, vel={self._vel_bucket}, "
                f"rot={math.degrees(self._rot_bucket):.0f} deg) "
                f"hw={self._hw}\n"
                f"  budget: {self._max_expansions} expansions / "
                f"{self._time_limit:.0f}s"
            )

        if start_bits == self._all_collected_mask:
            return []

        records: list[_TraceEntry] = [
            _TraceEntry(parent_idx=-1, action_idx=-1, repeat=0, g_frames=0),
        ]

        h0 = self._heuristic_frames(start_obs, start_pickups)
        start_key = self._canonical_key(start_obs, start_pickups)
        best_seen: dict[tuple, list[tuple[int, float]]] = {start_key: [(0, h0)]}

        # Heap entry: (f, h, remaining_pickups, counter, record_idx, g, state, obs, bits)
        pq: list[tuple] = []
        counter = 0
        heapq.heappush(
            pq,
            (
                self._hw * h0,
                h0,
                self._remaining_count(start_bits),
                counter,
                0,
                0,
                start_state,
                start_obs,
                start_bits,
            ),
        )
        counter += 1

        expansions = 0
        t_start = time.time()
        last_log = t_start
        best_pickups_left = self._remaining_count(start_bits)
        closest_remaining = self._nearest_remaining_distance(start_obs, start_pickups)

        while pq:
            elapsed = time.time() - t_start
            if expansions >= self._max_expansions or elapsed > self._time_limit:
                if self._verbose:
                    reason = "expansion cap" if expansions >= self._max_expansions else "time limit"
                    print(
                        f"  [whole] {reason}: expansions={expansions} "
                        f"frontier={len(pq)} best_remaining={best_pickups_left} "
                        f"closest={closest_remaining:.0f}px elapsed={elapsed:.1f}s"
                    )
                return []

            _f, _h, _rem, _cnt, rec_idx, g_frames, state, parent_obs, parent_bits = heapq.heappop(pq)
            if g_frames >= self._max_steps:
                continue

            expansions += 1

            repeat_options = self._repeat_options_for(parent_obs, parent_bits)
            for action_idx in range(NUM_ACTIONS):
                for repeat in repeat_options:
                    expanded = self._expand_macro(state, action_idx, repeat, parent_bits)
                    if expanded is None:
                        continue

                    new_state, new_obs, new_pickups, frames_taken, completed = expanded
                    child_g = g_frames + frames_taken
                    if child_g > self._max_steps:
                        continue

                    records.append(
                        _TraceEntry(
                            parent_idx=rec_idx,
                            action_idx=action_idx,
                            repeat=frames_taken,
                            g_frames=child_g,
                        )
                    )
                    child_idx = len(records) - 1

                    bits = _pickup_bits(new_pickups)
                    if completed or bits == self._all_collected_mask:
                        actions = self._reconstruct(records, child_idx)
                        if self._verbose:
                            print(
                                f"  [whole] completed in {len(actions)} frames "
                                f"({expansions} expansions, {time.time() - t_start:.1f}s)."
                            )
                        self._validate(actions)
                        return actions

                    h = self._heuristic_frames(new_obs, new_pickups)
                    key = self._canonical_key(new_obs, new_pickups)
                    if self._is_dominated(best_seen, key, child_g, h):
                        continue
                    self._remember_state(best_seen, key, child_g, h)

                    remaining = self._remaining_count(bits)
                    if remaining < best_pickups_left:
                        best_pickups_left = remaining
                        if self._verbose:
                            print(
                                f"  [whole] reached {self._total_pickups - remaining}/"
                                f"{self._total_pickups} pickups at frame {child_g} "
                                f"({expansions} expansions)."
                            )
                    closest_remaining = min(
                        closest_remaining,
                        self._nearest_remaining_distance(new_obs, new_pickups),
                    )

                    heapq.heappush(
                        pq,
                        (
                            child_g + self._hw * h,
                            h,
                            remaining,
                            counter,
                            child_idx,
                            child_g,
                            new_state,
                            new_obs,
                            bits,
                        ),
                    )
                    counter += 1

            if self._verbose:
                now = time.time()
                if now - last_log > 5.0:
                    last_log = now
                    best_f = pq[0][0] if pq else float("inf")
                    print(
                        f"  [whole] expansions={expansions} frontier={len(pq)} "
                        f"seen={len(best_seen)} best_f={best_f:.0f} "
                        f"best_remaining={best_pickups_left} "
                        f"elapsed={now - t_start:.1f}s"
                    )

        if self._verbose:
            print("  [whole] frontier exhausted.")
        return []

    # ------------------------------------------------------------------

    def _build_repeat_options(self, action_repeat: int) -> tuple[int, ...]:
        return (action_repeat,)

    def _repeat_options_for(self, obs, collected_bits: int) -> tuple[int, ...]:
        remaining = self._total_pickups - collected_bits.bit_count()
        nearest = self._nearest_uncollected_euclid_from_bits(obs, collected_bits)
        if remaining == 1 and nearest <= 120.0:
            return tuple(sorted({1, max(2, self._action_repeat // 2), self._action_repeat}))
        if remaining == 1 and nearest <= 220.0:
            return tuple(sorted({max(2, self._action_repeat // 2), self._action_repeat}))
        return self._repeat_options

    def _expand_macro(self, parent_state, action_idx: int, repeat: int, parent_bits: int):
        env = self._env
        env.load_state(parent_state)
        action = ALL_ACTIONS[action_idx]

        obs = None
        frames_taken = 0
        for _ in range(repeat):
            obs, _r, terminated, truncated, info = env.step(action)
            frames_taken += 1

            if info.get("ship_exploded", False):
                return None
            pickup_states = list(env.get_pickup_states())
            bits = _pickup_bits(pickup_states)
            if info.get("level_completed", False) or bits != parent_bits:
                return env.save_state(), obs, pickup_states, frames_taken, info.get("level_completed", False)
            if terminated or truncated:
                return None

        return env.save_state(), obs, list(env.get_pickup_states()), frames_taken, False

    def _canonical_key(self, obs, pickup_states: list[bool]) -> tuple:
        bits = _pickup_bits(pickup_states)
        remaining = self._total_pickups - bits.bit_count()
        nearest = self._nearest_uncollected_euclid_from_bits(obs, bits)
        precision = 1 if remaining == 1 and nearest <= 140.0 else 0
        pos_bucket = self._pos_bucket * (0.5 if precision else 1.0)
        vel_bucket = self._vel_bucket * (0.5 if precision else 1.0)
        rot_bucket = self._rot_bucket * (0.5 if precision else 1.0)
        rot_bins = max(1, int(round(2 * math.pi / rot_bucket)))

        x = int(round(float(obs[0]) / pos_bucket))
        y = int(round(float(obs[1]) / pos_bucket))
        vx = int(round(float(obs[2]) / vel_bucket))
        vy = int(round(float(obs[3]) / vel_bucket))
        rot = float(obs[4]) % (2.0 * math.pi)
        rb = int(rot / rot_bucket) % rot_bins
        return (precision, x, y, vx, vy, rb, bits)

    def _heuristic_frames(self, obs, pickup_states: list[bool]) -> float:
        remaining_mask = self._remaining_mask(pickup_states)
        if remaining_mask == 0:
            return 0.0

        x = float(obs[0])
        y = float(obs[1])
        vx = float(obs[2])
        vy = float(obs[3])

        best = float("inf")
        for first in self._iter_mask(remaining_mask):
            tx, ty = self._pickup_coords[first]
            euclid = math.hypot(tx - x, ty - y)
            try:
                path_dist, _, _ = self._pf.get_distance_to_specific_pickup(x, y, first)
            except Exception:
                path_dist = euclid

            if not math.isfinite(path_dist) or path_dist <= 0.0:
                path_dist = euclid

            ux = (tx - x) / max(1.0, euclid)
            uy = (ty - y) / max(1.0, euclid)
            speed_along = vx * ux + vy * uy

            tail_mask = remaining_mask & ~(1 << first)
            final_pickup = tail_mask == 0
            radius_discount = _PICKUP_COLLECTION_RADIUS if final_pickup else 0.0
            route_dist = max(0.0, path_dist - radius_discount)
            route_dist += self._route_tail[tail_mask][first]
            route_time = self._route_time_frames(route_dist, speed_along)
            collection_gap = max(0.0, euclid - radius_discount)
            gx = x + ux * collection_gap
            gy = y + uy * collection_gap
            open_air_time = self._time_to_point_frames(x, y, vx, vy, gx, gy)
            best = min(best, max(route_time, open_air_time))

        # At least a few frames per pickup are needed to perform distinct
        # collection events, even in the optimistic model.
        return max(best, 4.0 * remaining_mask.bit_count())

    def _time_to_point_frames(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        tx: float,
        ty: float,
    ) -> float:
        dx = tx - x
        dy = ty - y
        euclid = math.hypot(dx, dy)
        if euclid <= 1e-9:
            return 0.0

        ux = dx / euclid
        uy = dy / euclid
        v_direct = vx * ux + vy * uy
        t_direct = _bang_bang_time(v_direct, _A_ANY_MAX, euclid)

        v_toward_x = vx if dx >= 0.0 else -vx
        t_x = _bang_bang_time(v_toward_x, _A_X_MAX, abs(dx))

        if dy >= 0.0:
            t_y = _bang_bang_time(vy, _A_Y_DOWN_MAX, abs(dy))
        else:
            t_y = _bang_bang_time(-vy, _A_Y_UP_MAX, abs(dy))

        return max(t_direct, t_x, t_y) / _DT

    def _route_time_frames(self, distance: float, initial_speed: float = 0.0) -> float:
        if distance <= 0.0:
            return 0.0
        v0 = max(0.0, min(_ROUTE_VMAX, initial_speed))
        accel_dist = max(0.0, (_ROUTE_VMAX * _ROUTE_VMAX - v0 * v0) / (2.0 * _ROUTE_ACCEL))
        if distance <= accel_dist:
            seconds = (math.sqrt(v0 * v0 + 2.0 * _ROUTE_ACCEL * distance) - v0) / _ROUTE_ACCEL
        else:
            seconds = (_ROUTE_VMAX - v0) / _ROUTE_ACCEL
            seconds += (distance - accel_dist) / _ROUTE_VMAX
        return seconds / _DT

    def _build_path_pair_matrix(self) -> list[list[float]]:
        n = len(self._pickup_coords)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i, (ax, ay) in enumerate(self._pickup_coords):
            for j, (bx, by) in enumerate(self._pickup_coords):
                if i == j:
                    continue
                try:
                    dist, _, _ = self._pf.get_distance_to_specific_pickup(ax, ay, j)
                except Exception:
                    dist = math.hypot(bx - ax, by - ay)
                if not math.isfinite(dist) or dist <= 0.0:
                    dist = math.hypot(bx - ax, by - ay)
                matrix[i][j] = float(dist)
        return matrix

    def _build_route_tail_table(self) -> list[list[float]]:
        n = len(self._pickup_coords)
        mask_count = 1 << n
        tail = [[0.0 for _ in range(n)] for _ in range(mask_count)]

        for mask in range(1, mask_count):
            for last in range(n):
                best = float("inf")
                for nxt in self._iter_mask(mask):
                    d = self._path_pair[last][nxt] + tail[mask & ~(1 << nxt)][nxt]
                    if d < best:
                        best = d
                tail[mask][last] = best if math.isfinite(best) else 0.0
        return tail

    def _remaining_mask(self, pickup_states: list[bool]) -> int:
        mask = 0
        for i, collected in enumerate(pickup_states):
            if not collected:
                mask |= 1 << i
        return mask

    def _iter_mask(self, mask: int):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def _remaining_count(self, bits: int) -> int:
        return self._total_pickups - bits.bit_count()

    def _nearest_remaining_distance(self, obs, pickup_states: list[bool]) -> float:
        return self._nearest_uncollected_euclid_from_bits(obs, _pickup_bits(pickup_states))

    def _nearest_uncollected_euclid_from_bits(self, obs, collected_bits: int) -> float:
        x = float(obs[0])
        y = float(obs[1])
        best = float("inf")
        for i in range(self._total_pickups):
            if not (collected_bits & (1 << i)):
                px, py = self._pickup_coords[i]
                best = min(best, math.hypot(px - x, py - y))
        return 0.0 if best == float("inf") else best

    def _is_dominated(
        self,
        best_seen: dict[tuple, list[tuple[int, float]]],
        key: tuple,
        g_frames: int,
        h_frames: float,
    ) -> bool:
        entries = best_seen.get(key)
        if not entries:
            return False
        return any(prev_g <= g_frames and prev_h <= h_frames for prev_g, prev_h in entries)

    def _remember_state(
        self,
        best_seen: dict[tuple, list[tuple[int, float]]],
        key: tuple,
        g_frames: int,
        h_frames: float,
    ) -> None:
        entries = best_seen.setdefault(key, [])
        entries[:] = [
            (prev_g, prev_h)
            for prev_g, prev_h in entries
            if not (g_frames <= prev_g and h_frames <= prev_h)
        ]
        entries.append((g_frames, h_frames))
        if len(entries) > 4:
            entries.sort(key=lambda item: item[0] + self._hw * item[1])
            del entries[4:]

    def _reconstruct(self, records: list[_TraceEntry], goal_idx: int) -> list[int]:
        macro: list[tuple[int, int]] = []
        idx = goal_idx
        while idx > 0:
            rec = records[idx]
            macro.append((rec.action_idx, rec.repeat))
            idx = rec.parent_idx
        macro.reverse()

        frames: list[int] = []
        for action_idx, repeat in macro:
            frames.extend([action_idx] * repeat)
        return frames

    def _validate(self, actions: list[int]) -> None:
        env = self._env
        env.reset()
        for i, action_idx in enumerate(actions):
            _obs, _r, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
            if info.get("ship_exploded", False):
                print(f"  WARNING: validation crash at frame {i}")
                return
            if info.get("level_completed", False):
                if self._verbose:
                    print(f"  Validation OK: completed at frame {i + 1}")
                return
            if terminated or truncated:
                print(f"  WARNING: validation truncated at frame {i}")
                return
        if actions:
            print("  WARNING: validation ran out of actions without completing")
        else:
            print("  WARNING: empty trajectory")
