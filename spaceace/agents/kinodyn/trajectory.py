"""Phase-space reference trajectory for the SpaceAce kinodynamic agent.

This module produces the ``(position, velocity, tangent, curvature)`` reference
that the cascaded PD tracker then follows. It is the "planning" stage of the
hierarchical architecture: the pathfinder gives a wall-aware geometric route
through the cavern, and we turn that into a full kinodynamic reference by

1. **Resampling the polyline to uniform arc-length spacing** so tangent and
   curvature can be computed by finite differences.
2. **Gaussian-smoothing the resampled path** to replace the pathfinder's
   grid-aligned 90-degree corners with continuous arcs. The smoothing
   sigma is small enough (a few ds) that the path stays inside the
   pre-inflated clearance band around walls. Start and pickup-collection
   points are pinned during the smoothing so the controller still passes
   through the exact collection coordinates.
3. **Computing unit tangents and curvature** at every sample via central
   differences.
4. **Generating a time-optimal speed profile** v(s) using classical
   forward-backward sweeps with **gravity-asymmetric tangential acceleration
   bounds** and a **curvature-limited cornering cap**:

   * Forward sweep: ``v(i+1)^2 <= v(i)^2 + 2 * a_fwd(i) * ds``
     where ``a_fwd(i) = THRUST + GRAVITY * tangent_y(i)`` — projection of
     gravity onto the tangent. Pointing downward (+y) gains gravity assist;
     pointing upward (-y) loses it. This is the document's asymmetric
     double-integrator bound specialised to the path tangent.
   * Backward sweep: ``v(i-1)^2 <= v(i)^2 + 2 * a_brk(i-1) * ds`` with the
     equivalent formula for braking (thrust reversed).
   * Curvature cap: ``v(i) <= sqrt(a_lat_max / |kappa(i)|)`` keeps
     centripetal demand within the lateral thrust budget.

The resulting trajectory is the phase-space reference the tracker follows.
It explicitly encodes *where* the ship should be *and* *how fast* it should
be moving there — both of which the pure-pursuit earlier iteration was
missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spaceace.agents.kinodyn.heuristic import GRAVITY, THRUST_POWER


@dataclass
class ProfileConfig:
    """Parameters for trajectory generation."""

    ds: float = 6.0
    """Arc-length spacing of the resampled trajectory, in pixels."""

    smooth_sigma_samples: float = 3.0
    """Gaussian smoothing sigma, expressed in resampled samples (so the
    physical smoothing window is ``sigma_samples * ds`` pixels)."""

    pin_radius_samples: int = 2
    """Pin the start and pickup points during smoothing by weighting these
    samples on each side. Keeps the smoothed trajectory passing through the
    actual collection coordinates."""

    a_lat_max: float = 220.0
    """Maximum lateral (centripetal) acceleration allowed when cornering, in
    px/s^2. Bounded by the thrust budget available perpendicular to motion.
    Lower = safer corners, higher = faster cornering at the risk of drifting
    into walls."""

    a_tan_floor: float = 80.0
    """Floor on tangential acceleration — prevents the gravity-projection
    term from dropping acceleration to zero or going negative on segments
    that point directly against gravity."""

    v_cap: float = 500.0
    """Absolute cap on speed regardless of what the profile sweeps say."""

    v_final: float = 180.0
    """Target speed at the final waypoint. Slowing for the last pickup
    prevents overshoot into a wall pocket. Not zero — we just need to pass
    through the 10 px collection radius."""

    kappa_cap: float = 0.03
    """Upper bound on curvature used for the cornering-speed limit. Path
    samples at pickup "kinks" (where consecutive legs meet at an angle)
    have effectively infinite curvature because the tangent reverses over
    one sample. The pickup collection radius (10 px) and the smoothing
    pass mean the ship doesn't actually need to navigate that turn —
    it just needs to pass within 10 px. Clipping kappa here prevents a
    degenerate single-sample outlier from dragging the whole profile
    down to a crawl. Corresponds to R = 1/kappa_cap = 33 px, which the
    tracker can handle at cruise speed."""

    v_pickup_floor: float = 260.0
    """Minimum speed allowed by the profile at pickup-collection samples.
    Prevents the kinked geometry at a pickup from sucking the profile down
    to near-zero; the ship only needs to pass through the 10 px pickup
    radius, not come to rest on the coordinate."""


@dataclass
class Trajectory:
    pts: np.ndarray         # (N, 2) positions
    tan: np.ndarray         # (N, 2) unit tangents
    kappa: np.ndarray       # (N,) curvature magnitudes
    v: np.ndarray           # (N,) speed profile
    s: np.ndarray           # (N,) cumulative arc length
    pickup_idx: list[int]   # sample index of each pickup in collection order
    order: list[int]        # pickup indices in collection order

    def __len__(self) -> int:
        return len(self.pts)


# --------------------------------------------------------------------------
# Helpers: resampling, smoothing, differential geometry
# --------------------------------------------------------------------------


def _concat_polylines(
    polylines: list[np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Concatenate leg polylines, return combined points and the index of
    each leg's final waypoint (i.e., the pickup collection point)."""
    combined: list[np.ndarray] = []
    pickup_raw_idx: list[int] = []
    for i, leg in enumerate(polylines):
        if i > 0:
            combined.append(leg[1:])  # skip duplicate boundary
        else:
            combined.append(leg)
        pickup_raw_idx.append(sum(len(c) for c in combined) - 1)
    return np.concatenate(combined, axis=0).astype(np.float32), pickup_raw_idx


def _resample_uniform(
    pts: np.ndarray,
    ds: float,
    pin_indices: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Resample polyline ``pts`` so consecutive samples are ``ds`` apart.

    ``pin_indices`` mark raw-polyline samples (e.g., pickup coordinates)
    that must land *exactly* on an output sample — we re-emit those
    positions verbatim when we reach them. Returns the resampled array and
    the index in the resampled array for each pinned point.
    """
    if len(pts) < 2:
        return pts.copy(), list(pin_indices)

    out: list[np.ndarray] = [pts[0].copy()]
    new_pins: list[int] = []
    pin_set = set(pin_indices)
    if 0 in pin_set:
        new_pins.append(0)
    carry = 0.0  # arc length accumulated past the last emitted sample
    for i in range(1, len(pts)):
        a = pts[i - 1]
        b = pts[i]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            # Pinned point on a zero-length seg still needs to appear.
            if i in pin_set:
                out.append(b.copy())
                new_pins.append(len(out) - 1)
                carry = 0.0
            continue
        direction = seg / seg_len
        # Emit interpolated samples along this segment.
        dist_into_seg = ds - carry
        while dist_into_seg < seg_len - 1e-9:
            out.append(a + direction * dist_into_seg)
            dist_into_seg += ds
        # Handle pin: force the exact endpoint.
        if i in pin_set:
            out.append(b.copy())
            new_pins.append(len(out) - 1)
            carry = 0.0
        else:
            carry = seg_len - (dist_into_seg - ds)

    # Ensure final point is always present.
    if not np.allclose(out[-1], pts[-1]):
        out.append(pts[-1].copy())
    if (len(pts) - 1) in pin_set and new_pins and new_pins[-1] != len(out) - 1:
        new_pins.append(len(out) - 1)
    return np.asarray(out, dtype=np.float32), new_pins


def _gaussian_smooth(
    pts: np.ndarray,
    sigma: float,
    pin_indices: list[int],
    pin_radius: int,
) -> np.ndarray:
    """Gaussian-smooth x and y coordinates along the sample index.

    ``pin_indices`` are samples (after resampling) that must remain exactly
    at their input value. We blend the smoothed value back to the original
    using a triangular weight within ``pin_radius`` samples on either side.
    """
    n = len(pts)
    if n < 3 or sigma <= 0.0:
        return pts.copy()

    # Build a symmetric Gaussian kernel truncated at 3 sigma.
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()

    # Reflect-pad so the smoothing at endpoints doesn't pull them inward.
    padded = np.concatenate(
        [pts[radius:0:-1], pts, pts[-2 : -radius - 2 : -1]],
        axis=0,
    )
    smoothed = np.empty_like(pts)
    for ax in range(2):
        smoothed[:, ax] = np.convolve(padded[:, ax], kernel, mode="valid")[: n]

    # Pin weights: triangular falloff around pinned indices.
    if pin_indices and pin_radius > 0:
        pin_weight = np.zeros(n, dtype=np.float32)
        for pi in pin_indices:
            lo = max(0, pi - pin_radius)
            hi = min(n, pi + pin_radius + 1)
            for k in range(lo, hi):
                w = 1.0 - abs(k - pi) / float(pin_radius + 1)
                pin_weight[k] = max(pin_weight[k], w)
        smoothed = smoothed * (1.0 - pin_weight[:, None]) + pts * pin_weight[:, None]

    return smoothed


def _tangent_and_curvature(
    pts: np.ndarray,
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference unit tangent and curvature magnitude per sample."""
    n = len(pts)
    tan = np.zeros_like(pts)
    kappa = np.zeros(n, dtype=np.float32)
    if n < 3:
        return tan, kappa

    # First derivative (tangent direction).
    d1 = np.zeros_like(pts)
    d1[1:-1] = (pts[2:] - pts[:-2]) / (2.0 * ds)
    d1[0] = (pts[1] - pts[0]) / ds
    d1[-1] = (pts[-1] - pts[-2]) / ds

    # Second derivative (acceleration w.r.t. arc length).
    d2 = np.zeros_like(pts)
    d2[1:-1] = (pts[2:] - 2.0 * pts[1:-1] + pts[:-2]) / (ds * ds)
    d2[0] = d2[1]
    d2[-1] = d2[-2]

    # Unit tangent.
    norms = np.linalg.norm(d1, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    tan = (d1 / norms).astype(np.float32)

    # Curvature: |d1 x d2| / |d1|^3 (scalar cross in 2D).
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    denom = (norms[:, 0] ** 3)
    denom = np.maximum(denom, 1e-9)
    kappa = (np.abs(cross) / denom).astype(np.float32)

    return tan, kappa


# --------------------------------------------------------------------------
# Velocity profile
# --------------------------------------------------------------------------


def _velocity_profile(
    tan: np.ndarray,
    kappa: np.ndarray,
    ds: float,
    cfg: ProfileConfig,
    v_start: float,
    v_end: float,
    pickup_indices: list[int] | None = None,
) -> np.ndarray:
    """Classical forward-backward sweeps for a kinodynamic speed profile.

    Tangential acceleration is gravity-asymmetric: if the tangent has a
    positive y component (moving down in screen coords), gravity projects
    onto motion as assistance; negative y, gravity fights motion. The
    analogue for braking is just the negative. The curvature cap keeps
    centripetal demand within ``cfg.a_lat_max``.

    Pickup samples are exempted from the curvature-based speed cap by
    virtue of ``kappa_cap`` (clipping) and ``v_pickup_floor`` (floor on
    the resulting speed at those samples) — see ``ProfileConfig``.
    """
    n = len(tan)
    v = np.full(n, cfg.v_cap, dtype=np.float32)

    # Clip curvature to a sane ceiling before the cornering formula, then
    # lift the minimum-speed floor at pickup samples so sharp path "kinks"
    # at pickups don't collapse the whole profile.
    kappa_clip = np.minimum(np.maximum(kappa, 1e-6), cfg.kappa_cap)
    v_curve = np.sqrt(cfg.a_lat_max / kappa_clip).astype(np.float32)
    v = np.minimum(v, v_curve)
    if pickup_indices:
        for pi in pickup_indices:
            if 0 <= pi < n:
                v[pi] = max(float(v[pi]), cfg.v_pickup_floor)

    # Boundary conditions.
    v[0] = min(v[0], v_start)
    v[-1] = min(v[-1], v_end)

    # Tangential acceleration budgets per sample, gravity-aware.
    # tangent.y is the projection of +y onto motion direction: positive
    # tangent.y means we're moving with gravity.
    ty = tan[:, 1]
    a_fwd = np.maximum(THRUST_POWER + GRAVITY * ty, cfg.a_tan_floor)
    a_brk = np.maximum(THRUST_POWER - GRAVITY * ty, cfg.a_tan_floor)

    # Forward sweep: v(i+1)^2 <= v(i)^2 + 2 * a_fwd(i) * ds
    for i in range(1, n):
        limit = math.sqrt(v[i - 1] * v[i - 1] + 2.0 * float(a_fwd[i - 1]) * ds)
        if limit < v[i]:
            v[i] = limit

    # Backward sweep: v(i-1)^2 <= v(i)^2 + 2 * a_brk(i-1) * ds
    for i in range(n - 2, -1, -1):
        limit = math.sqrt(v[i + 1] * v[i + 1] + 2.0 * float(a_brk[i]) * ds)
        if limit < v[i]:
            v[i] = limit

    return v


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_trajectory(
    pathfinder: "spaceace_rl.PyPathfinder",  # noqa: F821
    start_xy: tuple[float, float],
    pickup_coords: list[tuple[float, float]],
    order: list[int],
    cfg: ProfileConfig,
    v_start: float = 0.0,
) -> Trajectory | None:
    """Generate the full kinodynamic reference for collection ``order``.

    Returns ``None`` if any leg of the pathfinder query fails.
    """
    if not order:
        return None

    src_x, src_y = start_xy
    raw_polylines: list[np.ndarray] = []
    for target in order:
        try:
            leg = pathfinder.get_path_to_specific_pickup(src_x, src_y, target)
        except Exception:
            return None
        if not leg:
            return None
        leg_arr = np.asarray(leg, dtype=np.float32)
        # Force the exact pickup coordinate as the final waypoint — the
        # pathfinder's grid output can terminate a few pixels short.
        pk = np.asarray(pickup_coords[target], dtype=np.float32)
        if np.linalg.norm(leg_arr[-1] - pk) > 1e-3:
            leg_arr = np.vstack([leg_arr, pk[None, :]])
        raw_polylines.append(leg_arr)
        src_x, src_y = float(pk[0]), float(pk[1])

    raw_pts, pickup_raw_idx = _concat_polylines(raw_polylines)
    # Pin the start (the true ship position) and every pickup sample during
    # *resampling* so we can look them up in the resampled array — but the
    # resampling index is independent of whether those samples are pinned
    # during smoothing.
    pin_raw_for_resample = [0] + pickup_raw_idx

    pts_rs, pin_rs = _resample_uniform(raw_pts, cfg.ds, pin_raw_for_resample)

    # Pin the start and every pickup during smoothing so the smoothed path
    # still passes within 1–2 px of each pickup (collection radius is 10 px).
    # The pinned corners stay sharp, producing high local curvature — we
    # handle that at profile-generation time, not by relaxing the pinning.
    pts_sm = _gaussian_smooth(
        pts_rs, cfg.smooth_sigma_samples, pin_rs, cfg.pin_radius_samples
    )
    tan, kappa = _tangent_and_curvature(pts_sm, cfg.ds)

    # Map pin_rs -> pickup_idx (dropping the leading 0 start pin). Even
    # though pickup samples are not pinned during smoothing, their index in
    # the resampled array is still the location the ship must pass closest
    # to, used for progress tracking and velocity-profile floors.
    pickup_idx = pin_rs[1 : len(pickup_raw_idx) + 1]

    v = _velocity_profile(
        tan, kappa, cfg.ds, cfg, v_start, cfg.v_final, pickup_indices=pickup_idx
    )

    n = len(pts_sm)
    s = np.arange(n, dtype=np.float32) * cfg.ds

    return Trajectory(
        pts=pts_sm,
        tan=tan,
        kappa=kappa,
        v=v,
        s=s,
        pickup_idx=pickup_idx,
        order=list(order),
    )
