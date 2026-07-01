"""Diagnostic: trace which constraint is binding at every sample of the
kinodynamic agent's reference velocity profile.

Reports, per pickup-segment and overall:
  - max / mean / min reference speed
  - fraction of samples where each constraint is binding (v_cap, curvature,
    forward sweep, backward sweep, pickup floor)
  - estimated reference traversal time (sum of ds / v)
  - peak gravity-assist tangent (max ty over the segment)

Run with default config to confirm whether the existing profile is
artificially speed-limited; re-run with relaxed config to see whether the
profile builder is structurally capable of human-like speeds.

    uv run python scripts/audit_kinodyn_profile.py --level 7
    uv run python scripts/audit_kinodyn_profile.py --level 7 --v-cap 1000 --v-final 1000 --a-lat 600
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spaceace_rl  # noqa: E402

from spaceace.agents.kinodyn.heuristic import GRAVITY, THRUST_POWER  # noqa: E402
from spaceace.agents.kinodyn.trajectory import (  # noqa: E402
    ProfileConfig,
    _concat_polylines,
    _gaussian_smooth,
    _resample_uniform,
    _tangent_and_curvature,
)
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402


def velocity_profile_with_trace(
    tan: np.ndarray,
    kappa: np.ndarray,
    ds: float,
    cfg: ProfileConfig,
    v_start: float,
    v_end: float,
    pickup_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduces _velocity_profile but also returns a per-sample trace of
    which constraint was binding at the end of all sweeps.

    Returns:
        v_final  — final clamped speed per sample
        v_curve  — curvature cap per sample (before sweeps)
        v_fwd    — forward-sweep cap per sample (forward only)
        v_brk    — backward-sweep cap per sample (backward only, applied to v_fwd)
        binding  — int code per sample: 0=cap, 1=curve, 2=fwd, 3=brk, 4=pickup_floor
    """
    n = len(tan)
    binding = np.zeros(n, dtype=np.int8)

    # Curvature cap (and v_cap as a global ceiling)
    kappa_clip = np.minimum(np.maximum(kappa, 1e-6), cfg.kappa_cap)
    v_curve_only = np.sqrt(cfg.a_lat_max / kappa_clip).astype(np.float32)
    v0 = np.full(n, cfg.v_cap, dtype=np.float32)
    v_curve = np.minimum(v0, v_curve_only)
    binding[:] = np.where(v_curve_only < cfg.v_cap, 1, 0)

    # Pickup floor
    if pickup_indices:
        for pi in pickup_indices:
            if 0 <= pi < n:
                if v_curve[pi] < cfg.v_pickup_floor:
                    v_curve[pi] = cfg.v_pickup_floor
                    binding[pi] = 4

    v = v_curve.copy()
    v[0] = min(v[0], v_start)
    v[-1] = min(v[-1], v_end)

    ty = tan[:, 1]
    a_fwd = np.maximum(THRUST_POWER + GRAVITY * ty, cfg.a_tan_floor)
    a_brk = np.maximum(THRUST_POWER - GRAVITY * ty, cfg.a_tan_floor)

    # Forward sweep
    v_fwd = v.copy()
    for i in range(1, n):
        limit = math.sqrt(v_fwd[i - 1] * v_fwd[i - 1] + 2.0 * float(a_fwd[i - 1]) * ds)
        if limit < v_fwd[i]:
            v_fwd[i] = limit
            binding[i] = 2

    # Backward sweep
    v_brk = v_fwd.copy()
    for i in range(n - 2, -1, -1):
        limit = math.sqrt(v_brk[i + 1] * v_brk[i + 1] + 2.0 * float(a_brk[i]) * ds)
        if limit < v_brk[i]:
            v_brk[i] = limit
            binding[i] = 3

    # binding[0] / binding[-1] reflect the boundary clamp from v_start / v_end
    return v_brk, v_curve_only, v_fwd, v_brk, binding


def report_segment(
    label: str,
    seg_v: np.ndarray,
    seg_binding: np.ndarray,
    seg_ty: np.ndarray,
    seg_kappa: np.ndarray,
    ds: float,
) -> None:
    n = len(seg_v)
    if n == 0:
        return
    arc_length = (n - 1) * ds
    # Trapezoidal: dt = ds / v_avg where v_avg = (v[i] + v[i+1]) / 2.
    # Avoids the divide-by-zero blow-up at the v=0 start boundary that pure
    # ds/v[i] suffers from.
    if n >= 2:
        v_avg = 0.5 * (seg_v[:-1] + seg_v[1:])
        v_avg = np.maximum(v_avg, 1.0)  # 1 px/s floor for sanity
        seg_time_s = float(np.sum(ds / v_avg))
    else:
        seg_time_s = 0.0
    pct_cap = 100.0 * float(np.sum(seg_binding == 0)) / n
    pct_curve = 100.0 * float(np.sum(seg_binding == 1)) / n
    pct_fwd = 100.0 * float(np.sum(seg_binding == 2)) / n
    pct_brk = 100.0 * float(np.sum(seg_binding == 3)) / n
    pct_floor = 100.0 * float(np.sum(seg_binding == 4)) / n
    print(f"  {label:20s}  arc={arc_length:>5.0f}px  est_time={seg_time_s:>5.2f}s "
          f"({seg_time_s * 60.0:>5.0f}t)")
    print(f"    speed: max={seg_v.max():>5.0f}  mean={seg_v.mean():>5.0f}  "
          f"min={seg_v.min():>5.0f}")
    print(f"    binding: cap={pct_cap:>4.0f}%  curve={pct_curve:>4.0f}%  "
          f"fwd={pct_fwd:>4.0f}%  brk={pct_brk:>4.0f}%  floor={pct_floor:>4.0f}%")
    print(f"    geometry: peak_ty={seg_ty.max():>+.2f}  min_ty={seg_ty.min():>+.2f}  "
          f"max_kappa={seg_kappa.max():.4f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--v-cap", type=float, default=None)
    p.add_argument("--v-final", type=float, default=None)
    p.add_argument("--a-lat", type=float, default=None)
    p.add_argument("--v-pickup-floor", type=float, default=None)
    args = p.parse_args()

    cfg = ProfileConfig()
    if args.v_cap is not None: cfg.v_cap = args.v_cap
    if args.v_final is not None: cfg.v_final = args.v_final
    if args.a_lat is not None: cfg.a_lat_max = args.a_lat
    if args.v_pickup_floor is not None: cfg.v_pickup_floor = args.v_pickup_floor

    print(f"Level {args.level} kinodyn profile audit")
    print(f"  config: v_cap={cfg.v_cap} v_final={cfg.v_final} "
          f"a_lat={cfg.a_lat_max} v_pickup_floor={cfg.v_pickup_floor}")
    print(f"          ds={cfg.ds} smooth_sigma={cfg.smooth_sigma_samples} "
          f"kappa_cap={cfg.kappa_cap}")
    print()

    env = SpaceAceDirectEnv(level=args.level, max_steps=5000)
    env.reset()
    obs = env.get_observation()
    x0, y0 = float(obs[0]), float(obs[1])
    v_start = float(math.hypot(obs[2], obs[3]))

    pf = spaceace_rl.PyPathfinder(args.level, "grid")
    pickup_coords = list(pf.get_pickup_coords())
    n_pickups = len(pickup_coords)

    # Use the pathfinder's TSP order (matches what the agent itself uses by default)
    try:
        order = list(pf.get_tsp_order(x0, y0, [False] * n_pickups))
    except Exception:
        order = list(range(n_pickups))
    print(f"  pickups={n_pickups}  order={order}  start=({x0:.0f}, {y0:.0f})  "
          f"v_start={v_start:.0f}px/s")
    print()

    src_x, src_y = x0, y0
    raw_polylines = []
    for target in order:
        leg = pf.get_path_to_specific_pickup(src_x, src_y, target)
        if not leg:
            print(f"  pathfinder returned no leg to pickup {target}")
            return 1
        leg_arr = np.asarray(leg, dtype=np.float32)
        pk = np.asarray(pickup_coords[target], dtype=np.float32)
        if np.linalg.norm(leg_arr[-1] - pk) > 1e-3:
            leg_arr = np.vstack([leg_arr, pk[None, :]])
        raw_polylines.append(leg_arr)
        src_x, src_y = float(pk[0]), float(pk[1])

    raw_pts, pickup_raw_idx = _concat_polylines(raw_polylines)
    pin_raw_for_resample = [0] + pickup_raw_idx
    pts_rs, pin_rs = _resample_uniform(raw_pts, cfg.ds, pin_raw_for_resample)
    pts_sm = _gaussian_smooth(pts_rs, cfg.smooth_sigma_samples, pin_rs,
                               cfg.pin_radius_samples)
    tan, kappa = _tangent_and_curvature(pts_sm, cfg.ds)
    pickup_idx = pin_rs[1 : len(pickup_raw_idx) + 1]

    v, v_curve_only, v_fwd, v_brk, binding = velocity_profile_with_trace(
        tan, kappa, cfg.ds, cfg, v_start, cfg.v_final, pickup_idx
    )

    print(f"=== full trajectory ===")
    report_segment("overall", v, binding, tan[:, 1], kappa, cfg.ds)
    print()

    # Per pickup-segment
    seg_starts = [0] + list(pickup_idx)
    print(f"=== per pickup-segment ===")
    for si in range(len(pickup_idx)):
        a, b = seg_starts[si], seg_starts[si + 1] + 1
        label = f"start->P{order[si]}" if si == 0 else f"P{order[si - 1]}->P{order[si]}"
        report_segment(label, v[a:b], binding[a:b], tan[a:b, 1], kappa[a:b], cfg.ds)
    print()

    # First-segment human-comparison
    if pickup_idx:
        first_seg_v = v[: pickup_idx[0] + 1]
        peak = first_seg_v.max()
        cap_frac = 100.0 * float(np.sum(binding[: pickup_idx[0] + 1] == 0)) / len(first_seg_v)
        print(f"=== level-7 first-segment vs human reference ===")
        print(f"  reference peak speed:   {peak:>4.0f} px/s   (human peak: ~830, AI peak: ~600)")
        print(f"  ref capped by v_cap:    {cap_frac:>4.0f}%   (this is what we want to reduce)")
        if len(first_seg_v) >= 2:
            v_avg = 0.5 * (first_seg_v[:-1] + first_seg_v[1:])
            v_avg = np.maximum(v_avg, 1.0)
            seg_time_s = float(np.sum(cfg.ds / v_avg))
        else:
            seg_time_s = 0.0
        print(f"  est segment time:       {seg_time_s:>4.2f}s    "
              f"(human: ~6.11s, AI: ~7.42s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
