"""Kinodynamic time-optimal solver for SpaceAce.

Orchestrates the three-layer hierarchical architecture:

1. **Combinatorial** — asymmetric TSP over the pickup graph under
   gravity-aware travel-time costs (``_held_karp_atsp``).
2. **Geometric + kinodynamic** — convert the chosen ordering into a
   phase-space reference trajectory by concatenating pathfinder
   polylines, smoothing them into a continuously differentiable curve,
   and running forward-backward velocity-profile sweeps under
   asymmetric gravity and curvature constraints
   (:mod:`spaceace.agents.kinodyn.trajectory`).
3. **Control** — a cascaded PD tracker with gravity feedforward that
   converts the phase-space reference into one of SpaceAce's six
   discrete actions each frame (:mod:`spaceace.agents.kinodyn.controller`).

The simulator is the ground truth throughout: the planner searches
candidate pickup orderings, simulates each end-to-end, and returns the
best completed trajectory.
"""

from __future__ import annotations

import math
import time
from itertools import permutations
from typing import Optional

import spaceace_rl

from spaceace.agents.kinodyn.controller import ControllerConfig, controller_step
from spaceace.agents.kinodyn.heuristic import (
    DT,
    GRAVITY,
    ROTATION_SPEED_RAD_S,
    THRUST_POWER,
    kinodyn_time_ticks,
    route_time_ticks,
)
from spaceace.agents.kinodyn.trajectory import (
    ProfileConfig,
    Trajectory,
    build_trajectory,
)
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS

NUM_ACTIONS = len(ALL_ACTIONS)

# Cruise-speed upper bound used only by the ATSP cost matrix. Admissibility
# of the bound depends only on this being ≥ any achievable cruise speed, so
# slightly generous values are safe.
_CRUISE_VMAX_COST: float = 550.0

_HK_MAX_PICKUPS: int = 14


# --------------------------------------------------------------------------
# ATSP — asymmetric pickup ordering under gravity
# --------------------------------------------------------------------------


def _pair_cost_ticks(
    pf: "spaceace_rl.PyPathfinder",
    src_x: float,
    src_y: float,
    src_vx: float,
    src_vy: float,
    dst_idx: int,
    dst_x: float,
    dst_y: float,
) -> float:
    """Admissible lower bound on travel-time ticks from (src..) to pickup."""
    open_air = kinodyn_time_ticks(src_x, src_y, src_vx, src_vy, dst_x, dst_y)

    try:
        path_dist, _, _ = pf.get_distance_to_specific_pickup(src_x, src_y, dst_idx)
    except Exception:
        path_dist = math.hypot(dst_x - src_x, dst_y - src_y)

    if not math.isfinite(path_dist) or path_dist <= 0.0:
        path_dist = math.hypot(dst_x - src_x, dst_y - src_y)

    return max(open_air, route_time_ticks(path_dist, _CRUISE_VMAX_COST))


def _held_karp_atsp(
    n: int,
    start_costs: list[float],
    pair_costs: list[list[float]],
) -> tuple[list[int], float]:
    """Exact Held-Karp DP over the asymmetric TSP."""
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], start_costs[0]

    full = (1 << n) - 1
    dp: list[list[float]] = [[math.inf] * n for _ in range(1 << n)]
    parent: list[list[int]] = [[-1] * n for _ in range(1 << n)]

    for j in range(n):
        dp[1 << j][j] = start_costs[j]

    for mask in range(1, full + 1):
        for last in range(n):
            if not (mask >> last) & 1:
                continue
            if dp[mask][last] == math.inf:
                continue
            remaining = full ^ mask
            m = remaining
            while m:
                bit = m & -m
                nxt = bit.bit_length() - 1
                m ^= bit
                new_mask = mask | bit
                cand = dp[mask][last] + pair_costs[last][nxt]
                if cand < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = cand
                    parent[new_mask][nxt] = last

    end = min(range(n), key=lambda j: dp[full][j])
    total = dp[full][end]
    if not math.isfinite(total):
        return list(range(n)), math.inf

    order: list[int] = []
    mask = full
    cur = end
    while cur != -1:
        order.append(cur)
        prev = parent[mask][cur]
        mask ^= 1 << cur
        cur = prev
    order.reverse()
    return order, total


def _greedy_plus_2opt(
    n: int,
    start_costs: list[float],
    pair_costs: list[list[float]],
) -> tuple[list[int], float]:
    """Fallback for pickup counts above Held-Karp's practical ceiling."""
    if n == 0:
        return [], 0.0
    visited = [False] * n
    first = min(range(n), key=lambda j: start_costs[j])
    order = [first]
    visited[first] = True
    total = start_costs[first]
    for _ in range(n - 1):
        last = order[-1]
        best = -1
        best_c = math.inf
        for j in range(n):
            if visited[j]:
                continue
            if pair_costs[last][j] < best_c:
                best_c = pair_costs[last][j]
                best = j
        if best < 0:
            break
        order.append(best)
        visited[best] = True
        total += best_c

    def tour_cost(seq: list[int]) -> float:
        c = start_costs[seq[0]]
        for i in range(len(seq) - 1):
            c += pair_costs[seq[i]][seq[i + 1]]
        return c

    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for k in range(i + 1, len(order)):
                new_order = order[:i] + order[i : k + 1][::-1] + order[k + 1 :]
                if tour_cost(new_order) + 1e-6 < total:
                    order = new_order
                    total = tour_cost(new_order)
                    improved = True
                    break
            if improved:
                break
    return order, total


# --------------------------------------------------------------------------
# Top-level solver
# --------------------------------------------------------------------------


class KinodynSolver:
    """ATSP ordering + phase-space reference + cascaded PD tracker."""

    def __init__(
        self,
        env: SpaceAceDirectEnv,
        level: int,
        *,
        ds: float = 6.0,
        smooth_sigma_samples: float = 3.0,
        a_lat_max: float = 220.0,
        v_cap: float = 500.0,
        v_final: float = 180.0,
        kp_pos: float = 6.0,
        kd_vel: float = 3.2,
        rot_tolerance_thrust_deg: float = 18.0,
        thrust_deadband_accel: float = 40.0,
        lookahead_samples: int = 4,
        enumerate_orders: bool = True,
        enumerate_threshold: int = 5,
        max_tick_budget: int = 6000,
        max_idle_frames: int = 600,
        verbose: bool = True,
    ) -> None:
        self._env = env
        self._level = level
        self._profile_cfg = ProfileConfig(
            ds=float(ds),
            smooth_sigma_samples=float(smooth_sigma_samples),
            a_lat_max=float(a_lat_max),
            v_cap=float(v_cap),
            v_final=float(v_final),
        )
        self._ctrl_cfg = ControllerConfig(
            kp_pos=float(kp_pos),
            kd_vel=float(kd_vel),
            rot_tolerance_thrust_rad=math.radians(float(rot_tolerance_thrust_deg)),
            thrust_deadband_accel=float(thrust_deadband_accel),
            lookahead_samples=int(lookahead_samples),
        )
        self._enumerate = bool(enumerate_orders)
        self._enumerate_threshold = int(enumerate_threshold)
        self._tick_budget = int(max_tick_budget)
        self._max_idle = int(max_idle_frames)
        self._verbose = verbose

        self._pf = spaceace_rl.PyPathfinder(level, "grid")
        self._pickup_coords: list[tuple[float, float]] = list(self._pf.get_pickup_coords())
        self._n_pickups = len(self._pickup_coords)

        # Exposed so visualizer / debugger can draw the reference.
        self.last_trajectory: Optional[Trajectory] = None

    # ------------------------------------------------------------------

    def solve(self) -> list[int]:
        env = self._env
        env.reset()
        start_state = env.save_state()
        start_obs = env.get_observation()
        start_pstates = list(env.get_pickup_states())

        if self._verbose:
            print(
                "\nKinodynamic planner (phase-space reference + cascaded PD):\n"
                f"  pickups={self._n_pickups}\n"
                f"  profile: ds={self._profile_cfg.ds} sigma={self._profile_cfg.smooth_sigma_samples}"
                f" a_lat_max={self._profile_cfg.a_lat_max} v_cap={self._profile_cfg.v_cap}"
                f" v_final={self._profile_cfg.v_final}\n"
                f"  controller: kp_pos={self._ctrl_cfg.kp_pos} kd_vel={self._ctrl_cfg.kd_vel}"
                f" look={self._ctrl_cfg.lookahead_samples}\n"
                f"  physics: gravity={GRAVITY} thrust={THRUST_POWER}"
                f" rot_speed={ROTATION_SPEED_RAD_S:.2f} rad/s dt={DT:.4f}s"
            )

        if self._n_pickups == 0 or all(start_pstates):
            return []

        x0 = float(start_obs[0])
        y0 = float(start_obs[1])
        vx0 = float(start_obs[2])
        vy0 = float(start_obs[3])

        orderings = self._candidate_orderings(x0, y0, vx0, vy0)
        if self._verbose:
            for label, order, h in orderings:
                print(
                    f"  candidate [{label}]: order={order} "
                    f"heuristic={h:.0f} ticks (~{h / 60.0:.2f}s)"
                )

        best_actions: Optional[list[int]] = None
        best_traj: Optional[Trajectory] = None
        best_ticks = math.inf
        best_label: Optional[str] = None

        v_start = math.hypot(vx0, vy0)
        for label, order, _h in orderings:
            traj = build_trajectory(
                self._pf,
                (x0, y0),
                self._pickup_coords,
                order,
                self._profile_cfg,
                v_start=v_start,
            )
            if traj is None:
                if self._verbose:
                    print(f"  [{label}]: failed to build reference trajectory")
                continue
            if self._verbose:
                v_mean = float(traj.v.mean())
                v_min = float(traj.v.min())
                v_max = float(traj.v.max())
                total_s = float(traj.s[-1])
                print(
                    f"\n  === simulating [{label}] order={order} "
                    f"samples={len(traj)} arc={total_s:.0f}px "
                    f"v={v_min:.0f}/{v_mean:.0f}/{v_max:.0f} px/s ==="
                )

            t_sim = time.time()
            result = self._simulate(start_state, traj)
            if result is None:
                if self._verbose:
                    print(
                        f"  [{label}]: incomplete after "
                        f"{time.time() - t_sim:.1f}s"
                    )
                continue
            actions, ticks = result
            if self._verbose:
                print(
                    f"  [{label}] complete: {ticks} ticks = {ticks / 60.0:.2f}s "
                    f"(sim {time.time() - t_sim:.1f}s)"
                )
            if ticks < best_ticks:
                best_actions = actions
                best_ticks = ticks
                best_label = label
                best_traj = traj
                if self._verbose:
                    print(
                        f"  *** new best: [{label}] {ticks} ticks "
                        f"({ticks / 60.0:.2f}s) ***"
                    )

        if best_actions is None:
            if self._verbose:
                print("\n  [kinodyn] no ordering completed the level.")
            return []

        if self._verbose:
            print(
                f"\n  [kinodyn] chose [{best_label}]: "
                f"{best_ticks} ticks = {best_ticks / 60.0:.2f}s"
            )
        self.last_trajectory = best_traj
        self._validate(best_actions)
        return best_actions

    # ------------------------------------------------------------------

    def _candidate_orderings(
        self,
        x0: float,
        y0: float,
        vx0: float,
        vy0: float,
    ) -> list[tuple[str, list[int], float]]:
        n = self._n_pickups
        pair_costs = [[0.0] * n for _ in range(n)]
        start_costs = [0.0] * n
        for j in range(n):
            tx, ty = self._pickup_coords[j]
            start_costs[j] = _pair_cost_ticks(self._pf, x0, y0, vx0, vy0, j, tx, ty)
        for i in range(n):
            ix, iy = self._pickup_coords[i]
            for j in range(n):
                if i == j:
                    continue
                jx, jy = self._pickup_coords[j]
                pair_costs[i][j] = _pair_cost_ticks(
                    self._pf, ix, iy, 0.0, 0.0, j, jx, jy
                )

        candidates: list[tuple[str, list[int], float]] = []
        if n <= _HK_MAX_PICKUPS:
            order, total = _held_karp_atsp(n, start_costs, pair_costs)
            candidates.append(("atsp-hk", order, total))
        else:
            order, total = _greedy_plus_2opt(n, start_costs, pair_costs)
            candidates.append(("atsp-g2o", order, total))

        try:
            pf_order = list(self._pf.get_tsp_order(x0, y0, [False] * n))
            if pf_order and pf_order != candidates[0][1]:
                c = start_costs[pf_order[0]]
                for i in range(len(pf_order) - 1):
                    c += pair_costs[pf_order[i]][pf_order[i + 1]]
                candidates.append(("pathfinder-tsp", pf_order, c))
        except Exception:
            pass

        if self._enumerate and n <= self._enumerate_threshold:
            seen = {tuple(o) for _, o, _ in candidates}
            for perm in permutations(range(n)):
                if perm in seen:
                    continue
                seen.add(perm)
                c = start_costs[perm[0]]
                for i in range(len(perm) - 1):
                    c += pair_costs[perm[i]][perm[i + 1]]
                candidates.append((f"perm{perm}", list(perm), c))

        candidates.sort(key=lambda item: item[2])
        return candidates

    # ------------------------------------------------------------------

    def _simulate(
        self,
        start_state: "spaceace_rl.PyGameState",
        traj: Trajectory,
    ) -> Optional[tuple[list[int], int]]:
        env = self._env
        env.load_state(start_state)

        actions: list[int] = []
        sample_idx = 0
        pickups_left = self._n_pickups - sum(
            1 for p in env.get_pickup_states() if p
        )
        idle_since = 0

        for tick in range(self._tick_budget):
            obs = env.get_observation()
            ship_state = (
                float(obs[0]),
                float(obs[1]),
                float(obs[2]),
                float(obs[3]),
                float(obs[4]),
            )
            action_idx, sample_idx = controller_step(
                ship_state, traj, sample_idx, self._ctrl_cfg
            )
            action = ALL_ACTIONS[action_idx]
            _obs2, _r, terminated, truncated, info = env.step(action)
            actions.append(action_idx)

            if info.get("ship_exploded"):
                if self._verbose:
                    print(
                        f"    sim crashed at tick {tick + 1} "
                        f"sample_idx={sample_idx}/{len(traj)}"
                    )
                return None
            if info.get("level_completed"):
                return actions, len(actions)
            if terminated or truncated:
                if self._verbose:
                    print(
                        f"    sim terminated at tick {tick + 1} "
                        f"sample_idx={sample_idx}/{len(traj)}"
                    )
                return None

            new_left = self._n_pickups - sum(
                1 for p in env.get_pickup_states() if p
            )
            if new_left < pickups_left:
                pickups_left = new_left
                idle_since = 0
            else:
                idle_since += 1
                if idle_since > self._max_idle:
                    if self._verbose:
                        print(
                            f"    sim idle-abort: no pickup progress for "
                            f"{self._max_idle} ticks sample_idx={sample_idx}"
                        )
                    return None

        if self._verbose:
            print(f"    sim exhausted tick budget ({self._tick_budget})")
        return None

    # ------------------------------------------------------------------

    def _validate(self, actions: list[int]) -> None:
        env = self._env
        env.reset()
        for i, a_idx in enumerate(actions):
            _obs, _r, terminated, truncated, info = env.step(ALL_ACTIONS[a_idx])
            if info.get("ship_exploded"):
                print(f"  [kinodyn] WARNING: validation crash at frame {i}")
                return
            if info.get("level_completed"):
                if self._verbose:
                    print(f"  [kinodyn] validation OK: completed at frame {i + 1}")
                return
            if terminated or truncated:
                print(f"  [kinodyn] WARNING: validation truncated at frame {i}")
                return
        print("  [kinodyn] WARNING: validation ran out of actions without completing")
