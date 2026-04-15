"""Reactive pilot for HRL agent — no neural network, pure control logic.

Follows breadcrumb waypoints using:
1. Steer toward current breadcrumb
2. Thrust when heading is aligned
3. Curvature lookahead for pre-deceleration before corners
4. Wall TTI safety brake (retrobrake when about to hit a wall)
5. Gravity compensation (constant downward pull)
"""

import math
from typing import List, Tuple

import numpy as np

# Physics constants (must match Rust real_physics.rs)
THRUST_POWER = 400.0
ROTATION_SPEED = 4.363323  # rad/s
GRAVITY = 100.0
DT = 1.0 / 60.0

# Per-frame deltas
ROT_PER_FRAME = ROTATION_SPEED * DT  # ~0.0727 rad
THRUST_PER_FRAME = THRUST_POWER * DT  # ~6.67 px/frame

# 8 raycast directions relative to ship heading
_BASE_DIRS = [
    (0.0, -1.0),      # forward
    (0.707, -0.707),   # forward-right
    (1.0, 0.0),        # right
    (0.707, 0.707),    # back-right
    (0.0, 1.0),        # back
    (-0.707, 0.707),   # back-left
    (-1.0, 0.0),       # left
    (-0.707, -0.707),  # forward-left
]


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _angle_to(dx: float, dy: float) -> float:
    """Compute the ship rotation angle that faces direction (dx, dy).

    Ship heading at rotation r is (sin(r), -cos(r)), so the angle
    that produces heading (dx, dy) is atan2(dx, -dy).
    """
    return math.atan2(dx, -dy)


def _compute_curvature(breadcrumbs: List[Tuple[float, float]], start_idx: int,
                       lookahead: int = 4) -> float:
    """Compute total direction change over the next `lookahead` breadcrumb segments.

    Returns total absolute angle change in radians. 0 = straight, pi = hairpin.
    """
    total_turn = 0.0
    end = min(start_idx + lookahead + 1, len(breadcrumbs))
    if end - start_idx < 3:
        return 0.0

    prev_angle = None
    for i in range(start_idx, end - 1):
        dx = breadcrumbs[i + 1][0] - breadcrumbs[i][0]
        dy = breadcrumbs[i + 1][1] - breadcrumbs[i][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle = math.atan2(dy, dx)
        if prev_angle is not None:
            total_turn += abs(_wrap_angle(angle - prev_angle))
        prev_angle = angle

    return total_turn


def _compute_min_tti(obs: np.ndarray) -> float:
    """Minimum time-to-impact — precomputed in Rust at obs[19]."""
    return float(obs[19])


def compute_action(
    obs: np.ndarray,
    breadcrumbs: List[Tuple[float, float]],
    bc_idx: int,
    *,
    max_speed_straight: float = 120.0,
    max_speed_corner: float = 40.0,
    align_threshold: float = 0.35,      # ~20° — thrust when within this
    tti_brake_threshold: float = 1.5,    # seconds — start braking
    curvature_lookahead: int = 4,
) -> np.ndarray:
    """Decide [rotate_left, rotate_right, thrust] given current state and breadcrumbs.

    Returns np.array([left, right, thrust], dtype=int32).
    """
    ship_x, ship_y = float(obs[0]), float(obs[1])
    ship_vx, ship_vy = float(obs[2]), float(obs[3])
    ship_rot = float(obs[4])
    speed = math.sqrt(ship_vx * ship_vx + ship_vy * ship_vy)

    # --- Determine target point ---
    if not breadcrumbs or bc_idx >= len(breadcrumbs):
        # No target — just coast
        return np.array([0, 0, 0], dtype=np.int32)

    bx, by = breadcrumbs[bc_idx]
    dx, dy = bx - ship_x, by - ship_y
    dist_to_bc = math.sqrt(dx * dx + dy * dy)

    # --- Curvature-based speed target ---
    curvature = _compute_curvature(breadcrumbs, bc_idx, curvature_lookahead)
    # Map curvature 0..pi → max_speed_straight..max_speed_corner
    t = min(curvature / math.pi, 1.0)
    target_speed = max_speed_straight * (1.0 - t) + max_speed_corner * t

    # --- Wall TTI safety ---
    min_tti = _compute_min_tti(obs)
    wall_danger = min_tti < tti_brake_threshold

    # --- Decide mode: navigate vs retrobrake ---
    too_fast = speed > target_speed * 1.3
    needs_brake = wall_danger or too_fast

    if needs_brake and speed > 15.0:
        # Retrobrake: face opposite velocity, thrust
        vel_angle = _angle_to(ship_vx, ship_vy)
        desired_angle = _wrap_angle(vel_angle + math.pi)
        angle_err = _wrap_angle(desired_angle - ship_rot)

        rotate_left = 1 if angle_err < -ROT_PER_FRAME else 0
        rotate_right = 1 if angle_err > ROT_PER_FRAME else 0
        # Only thrust retrograde if roughly facing opposite velocity
        thrust = 1 if abs(angle_err) < 0.6 else 0  # ~34°

        return np.array([rotate_left, rotate_right, thrust], dtype=np.int32)

    # --- Normal navigation: steer toward breadcrumb ---
    if dist_to_bc < 1e-6:
        return np.array([0, 0, 0], dtype=np.int32)

    desired_angle = _angle_to(dx, dy)
    angle_err = _wrap_angle(desired_angle - ship_rot)

    rotate_left = 1 if angle_err < -ROT_PER_FRAME else 0
    rotate_right = 1 if angle_err > ROT_PER_FRAME else 0

    # Thrust when heading is roughly aligned AND not too fast
    aligned = abs(angle_err) < align_threshold
    under_speed = speed < target_speed
    thrust = 1 if aligned and under_speed else 0

    # If very slow and gravity is pulling us down, thrust even if not perfectly aligned
    # to maintain altitude (gravity = 100 downward = ~1.67/frame)
    if speed < 30.0 and abs(angle_err) < 1.0:
        thrust = 1

    return np.array([rotate_left, rotate_right, thrust], dtype=np.int32)
