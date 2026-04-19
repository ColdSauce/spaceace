"""A* planner for SpaceAce — decomposed outer-TSP + inner single-target A*.

Rationale
---------
A single monolithic A* over ``(x, y, vx, vy, θ, pickup_mask)`` suffers from a
2^N state-multiplier (N = number of pickups) and a single-target-only heuristic
that cannot guide the search across multiple collection events. Both effects
compound: the search sprawls over a combinatorially bloated space with weak
global guidance and fails to reach the first pickup even on simple levels.

This module instead uses the classical robotics/game-AI decomposition:

1. **Outer loop** — pickup ordering. A TSP tour over remaining pickups using
   the Rust grid pathfinder's Held-Karp solver. Re-queried after each leg so
   opportunistic pickups (swept up mid-flight) don't leave stale plans behind.

2. **Inner loop** — single-target kinodynamic A* from the current ship state
   to one pickup. The state excludes the pickup mask entirely (huge state-space
   reduction), and the heuristic is admissible per-axis 1D bang-bang bounds
   combined with ``max`` (concurrent, not sequential).

The admissible heuristic `h(s, P)`:
    h = max(t_x, t_y) / DT           [in physics frames]
  where
    t_x = 1D min-time to close x-gap under |a_x| ≤ THRUST
    t_y = 1D min-time to close y-gap under
              a_y ∈ [−(THRUST−g), +(THRUST+g)]  (asymmetric because gravity
              helps downward and hurts upward)

Each t_axis is the positive root of
    v² + 2·a·|d| − (|v_aligned| + a·t)² ≥ 0
evaluated with the initial velocity projected onto the goal direction
(``v_aligned = max(v_toward, 0)``; using ``max(·,0)`` rather than
``v_toward`` keeps the bound valid when velocity points *away* from the
target — ignoring that deceleration phase only makes the bound smaller, which
is fine for admissibility).

Rotation is deliberately *not* added to the max: the ship may arrive at the
goal in any orientation, and gravity-only ballistic segments need no rotation
at all, so adding a rotation term would break admissibility. Rotation cost
enters implicitly through g(n) (each rotate frame is a real step).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

import spaceace_rl

from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS

NUM_ACTIONS = len(ALL_ACTIONS)

# Physics constants mirror `src/real_physics.rs`.
_GRAVITY = 100.0
_THRUST_POWER = 400.0
_ROTATION_SPEED = 4.363323
_DT = 1.0 / 60.0

# Per-axis max accelerations (always admissible lower bounds on travel time).
_A_X_MAX = _THRUST_POWER                    # 400: pure horizontal thrust
_A_Y_DOWN_MAX = _THRUST_POWER + _GRAVITY    # 500: thrust + gravity, both downward
_A_Y_UP_MAX = _THRUST_POWER - _GRAVITY      # 300: thrust fighting gravity


@dataclass
class _NodeRecord:
    parent_idx: int  # -1 for root
    action_idx: int  # 0..5, -1 for root
    g_frames: int


class _Frontier:
    """Priority-queue entry. Comparison on (f, tiebreak) only."""

    __slots__ = ("f", "tiebreak", "node_idx", "state", "obs")

    def __init__(self, f: float, tiebreak: int, node_idx: int, state, obs) -> None:
        self.f = float(f)
        self.tiebreak = int(tiebreak)
        self.node_idx = int(node_idx)
        self.state = state
        self.obs = obs

    def __lt__(self, other: "_Frontier") -> bool:
        if self.f != other.f:
            return self.f < other.f
        return self.tiebreak < other.tiebreak


def _bang_bang_time(v_toward: float, a_max: float, d_abs: float) -> float:
    """Admissible lower-bound on 1D travel time under |a| ≤ a_max.

    Uses ``v_aligned = max(v_toward, 0)`` so that adverse initial velocity
    only *tightens* the bound (lengthens the time), preserving admissibility
    without needing a deceleration-phase treatment.
    """
    if d_abs <= 0.0:
        return 0.0
    v = max(v_toward, 0.0)
    disc = v * v + 2.0 * a_max * d_abs
    return (math.sqrt(disc) - v) / a_max


class AStarSolver:
    """Outer TSP + inner physics-aware A*.

    Parameters mostly carry over from the previous monolithic implementation;
    the key additions are ``leg_time_limit_s`` and ``leg_max_expansions``
    which budget the inner A* per-leg (the outer loop solves one pickup at a
    time and each leg gets its own budget).
    """

    def __init__(
        self,
        env: SpaceAceDirectEnv,
        level: int,
        action_repeat: int = 4,
        pos_bucket: float = 8.0,
        vel_bucket: float = 8.0,
        rot_bucket_deg: float = 10.0,
        leg_max_expansions: int = 200_000,
        leg_time_limit_s: float = 60.0,
        heuristic_weight: float = 1.0,
        max_steps: int = 3000,
        verbose: bool = True,
    ) -> None:
        self._env = env
        self._action_repeat = action_repeat
        self._pos_bucket = float(pos_bucket)
        self._vel_bucket = float(vel_bucket)
        self._rot_bucket = math.radians(rot_bucket_deg)
        self._rot_bins = max(1, int(round(2 * math.pi / self._rot_bucket)))
        self._leg_max_expansions = leg_max_expansions
        self._leg_time_limit = leg_time_limit_s
        self._hw = heuristic_weight
        self._max_steps = max_steps
        self._verbose = verbose

        self._pf = spaceace_rl.PyPathfinder(level, "grid")
        self._pickup_coords = list(self._pf.get_pickup_coords())

    # ------------------------------------------------------------------ outer

    def solve(self) -> list[int]:
        env = self._env
        env.reset()

        if self._verbose:
            print(
                f"\nA* planner (outer TSP + inner single-target):\n"
                f"  action_repeat={self._action_repeat} "
                f"buckets=(pos={self._pos_bucket}, vel={self._vel_bucket}, "
                f"rot={math.degrees(self._rot_bucket):.0f}°) "
                f"hw={self._hw}\n"
                f"  per-leg budget: {self._leg_max_expansions} expansions / "
                f"{self._leg_time_limit:.0f}s"
            )

        state = env.save_state()
        all_actions: list[int] = []
        frames_used = 0
        leg_idx = 0

        while True:
            env.load_state(state)
            pstates = list(env.get_pickup_states())
            if all(pstates):
                break
            obs = env.get_observation()

            # Re-query TSP order each leg — opportunistic side-pickups may
            # have shifted the optimal ordering.
            try:
                order = list(self._pf.get_tsp_order(float(obs[0]), float(obs[1]), pstates))
            except Exception as e:
                print(f"  TSP failed: {e}")
                return []
            if not order:
                break
            target_idx = order[0]
            target_x, target_y = self._pickup_coords[target_idx]

            frames_remaining = self._max_steps - frames_used
            if frames_remaining <= 0:
                print("  Frame budget exhausted.")
                return []

            if self._verbose:
                print(
                    f"\n  Leg {leg_idx}: target pickup #{target_idx} at "
                    f"({target_x:.0f}, {target_y:.0f}), "
                    f"ship at ({obs[0]:.0f}, {obs[1]:.0f}), "
                    f"frames_remaining={frames_remaining}"
                )

            leg_result = self._solve_leg(
                start_state=state, target_pickup_idx=target_idx,
                target_x=target_x, target_y=target_y,
                max_frames=frames_remaining,
            )
            if leg_result is None:
                print(f"  Leg {leg_idx}: no path found to pickup #{target_idx}.")
                return []

            leg_actions, end_state, new_frames = leg_result
            all_actions.extend(leg_actions)
            state = end_state
            frames_used += new_frames
            leg_idx += 1

            # Check for level completion (the last collected pickup ends the
            # level immediately in the sim).
            env.load_state(state)
            if env.get_info().get("level_completed", False):
                break

        if self._verbose:
            print(f"\nA* total: {len(all_actions)} frames across {leg_idx} legs.")
        self._validate(all_actions)
        return all_actions

    # ------------------------------------------------------------------ inner

    def _solve_leg(
        self, start_state, target_pickup_idx: int,
        target_x: float, target_y: float, max_frames: int,
    ):
        """A* from ``start_state`` to the given pickup.

        Goal test is "target pickup was collected during the transition" —
        reading ``get_pickup_states`` after each macro-action.

        Returns
        -------
        (leg_actions, end_state, frames_used) on success, else None.
        """
        env = self._env
        env.load_state(start_state)
        init_obs = env.get_observation()

        init_key = self._canonical_key(init_obs)
        records: list[_NodeRecord] = [
            _NodeRecord(parent_idx=-1, action_idx=-1, g_frames=0),
        ]
        best_g: dict[tuple, int] = {init_key: 0}

        pq: list[_Frontier] = []
        tiebreak = 0

        h0 = self._heuristic_frames(init_obs, target_x, target_y, target_pickup_idx)
        heapq.heappush(
            pq,
            _Frontier(f=self._hw * h0, tiebreak=tiebreak, node_idx=0,
                      state=start_state, obs=init_obs),
        )
        tiebreak += 1

        t_start = time.time()
        expansions = 0
        last_log_t = t_start
        closest_dist = math.hypot(init_obs[0] - target_x, init_obs[1] - target_y)

        while pq:
            if expansions >= self._leg_max_expansions:
                if self._verbose:
                    print(
                        f"    [leg] expansion cap ({expansions}) — closest "
                        f"approach: {closest_dist:.0f}px."
                    )
                return None
            elapsed = time.time() - t_start
            if elapsed > self._leg_time_limit:
                if self._verbose:
                    print(
                        f"    [leg] time limit ({elapsed:.1f}s) — closest "
                        f"approach: {closest_dist:.0f}px."
                    )
                return None

            entry = heapq.heappop(pq)
            rec = records[entry.node_idx]

            # Stale-entry check.
            key = self._canonical_key(entry.obs)
            if best_g.get(key, 1 << 30) < rec.g_frames:
                continue

            # Frame-budget bound: any descendant of this node will have
            # g ≥ rec.g_frames + action_repeat.
            if rec.g_frames + self._action_repeat > max_frames:
                continue

            expansions += 1

            for action_idx in range(NUM_ACTIONS):
                result = self._expand(entry.state, action_idx, target_pickup_idx)
                new_state, new_obs, collected_target, crashed = result
                if crashed:
                    continue

                child_g = rec.g_frames + self._action_repeat

                if collected_target:
                    # Goal — record and reconstruct.
                    records.append(_NodeRecord(
                        parent_idx=entry.node_idx, action_idx=action_idx,
                        g_frames=child_g,
                    ))
                    if self._verbose:
                        print(
                            f"    [leg] pickup collected in {child_g} frames "
                            f"({expansions} expansions, {time.time() - t_start:.1f}s)."
                        )
                    leg_actions = self._reconstruct(records, len(records) - 1)
                    return leg_actions, new_state, child_g

                if child_g >= max_frames:
                    continue

                child_key = self._canonical_key(new_obs)
                prior = best_g.get(child_key)
                if prior is not None and prior <= child_g:
                    continue
                best_g[child_key] = child_g

                h = self._heuristic_frames(new_obs, target_x, target_y, target_pickup_idx)
                f = child_g + self._hw * h

                records.append(_NodeRecord(
                    parent_idx=entry.node_idx, action_idx=action_idx,
                    g_frames=child_g,
                ))
                child_node_idx = len(records) - 1

                d = math.hypot(new_obs[0] - target_x, new_obs[1] - target_y)
                if d < closest_dist:
                    closest_dist = d

                heapq.heappush(
                    pq,
                    _Frontier(f=f, tiebreak=tiebreak, node_idx=child_node_idx,
                              state=new_state, obs=new_obs),
                )
                tiebreak += 1

            if self._verbose:
                now = time.time()
                if now - last_log_t > 5.0:
                    last_log_t = now
                    print(
                        f"    [leg] expansions={expansions} frontier={len(pq)} "
                        f"tt={len(best_g)} closest={closest_dist:.0f}px "
                        f"elapsed={now - t_start:.1f}s"
                    )

        if self._verbose:
            print(f"    [leg] frontier exhausted — closest={closest_dist:.0f}px.")
        return None

    # -------------------------------------------------------------- internals

    def _canonical_key(self, obs) -> tuple:
        x = int(round(float(obs[0]) / self._pos_bucket))
        y = int(round(float(obs[1]) / self._pos_bucket))
        vx = int(round(float(obs[2]) / self._vel_bucket))
        vy = int(round(float(obs[3]) / self._vel_bucket))
        rot = float(obs[4]) % (2.0 * math.pi)
        rb = int(rot / self._rot_bucket) % self._rot_bins
        return (x, y, vx, vy, rb)

    def _heuristic_frames(
        self, obs, target_x: float, target_y: float,
        target_pickup_idx: int,
    ) -> float:
        """Physics-aware single-target admissible lower bound (in frames).

        Three admissible lower bounds combined with ``max`` (concurrent):

        * **t_x** — 1D bang-bang in x under |a_x| ≤ THRUST.
        * **t_y** — 1D bang-bang in y under asymmetric gravity-aware accel.
        * **t_path** — bang-bang through the grid pathfinder's wall-routed
          path distance to the target pickup (handles detour-around-wall
          cases where the 1D Euclidean bounds are too optimistic).

        All three are valid lower bounds on travel time, so ``max`` of them
        is also a lower bound. The path-aware term is what prevents the
        search from sprawling toward Euclidean-closest points that are
        actually walled off.
        """
        x, y = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        speed = math.hypot(vx, vy)
        dx = target_x - x
        dy = target_y - y

        # x-axis bound.
        v_toward_x = vx if dx >= 0 else -vx
        t_x = _bang_bang_time(v_toward_x, _A_X_MAX, abs(dx))

        # y-axis bound, asymmetric in gravity's favored direction.
        if dy >= 0:
            t_y = _bang_bang_time(vy, _A_Y_DOWN_MAX, abs(dy))
        else:
            t_y = _bang_bang_time(-vy, _A_Y_UP_MAX, abs(dy))

        # Wall-aware bound: path_dist ≥ Euclidean, so traversing it with
        # peak omnidirectional accel is also a valid lower bound. Use the
        # weakest axis accel (_A_Y_UP_MAX = 300) to stay safe.
        try:
            path_dist, _, _ = self._pf.get_distance_to_specific_pickup(
                x, y, target_pickup_idx,
            )
        except Exception:
            path_dist = float("inf")
        if math.isfinite(path_dist) and path_dist > 0.0:
            t_path = _bang_bang_time(speed, _A_Y_UP_MAX, path_dist)
        else:
            t_path = 0.0

        t_sec = max(t_x, t_y, t_path)
        return t_sec / _DT

    def _expand(
        self, parent_state, action_idx: int, target_pickup_idx: int,
    ):
        """Apply one macro-action. Returns (new_state, obs, collected_target, crashed)."""
        env = self._env
        env.load_state(parent_state)
        pre_states = list(env.get_pickup_states())
        target_already_collected = pre_states[target_pickup_idx]
        action = ALL_ACTIONS[action_idx]

        obs = None
        crashed = False
        for _ in range(self._action_repeat):
            obs, _r, terminated, truncated, info = env.step(action)
            if info.get("ship_exploded", False):
                crashed = True
                break
            if info.get("level_completed", False):
                break
            if terminated or truncated:
                crashed = True
                break

        if crashed:
            return None, None, False, True

        post_states = list(env.get_pickup_states())
        collected_target = (
            not target_already_collected and post_states[target_pickup_idx]
        )
        new_state = env.save_state()
        return new_state, obs, collected_target, False

    # ------------------------------------------------------------- utilities

    def _reconstruct(
        self, records: list[_NodeRecord], goal_idx: int,
    ) -> list[int]:
        macro: list[int] = []
        idx = goal_idx
        while idx > 0:
            rec = records[idx]
            macro.append(rec.action_idx)
            idx = rec.parent_idx
        macro.reverse()
        frames: list[int] = []
        for a in macro:
            frames.extend([a] * self._action_repeat)
        return frames

    def _validate(self, actions: list[int]) -> None:
        env = self._env
        env.reset()
        for i, a in enumerate(actions):
            _obs, _r, terminated, truncated, info = env.step(ALL_ACTIONS[a])
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
        if not actions:
            print("  WARNING: empty trajectory")
        else:
            print("  WARNING: validation ran out of actions without completing")
