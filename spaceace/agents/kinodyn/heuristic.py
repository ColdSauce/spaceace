"""Asymmetric double-integrator heuristic for SpaceAce phase-space A*.

Every value below is a strict lower bound on the true minimum time, which is
what the A* admissibility proof demands. The two axes are analysed
independently using per-axis bang-bang min-time, and the per-axis results are
combined with ``max(...)`` — admissible because the ship must cover both axis
distances, and cannot cover either faster than its axis bound.

Gravity is constant and oriented along +y in SpaceAce's screen-coordinate
convention (larger y = further down). That makes the y axis asymmetric:

    * Moving in +y (with gravity): max |accel| = thrust + gravity
    * Moving in -y (against gravity): max |accel| = thrust - gravity

The x axis is symmetric: rotating the thruster horizontally yields |T| accel
in either direction; gravity is perpendicular and cannot slow horizontal
closure on its own.

Using the per-axis maxima assumes the thruster could be fully projected onto
that axis for the entire burn. That is not physically realisable simultaneously
on both axes — the true 2D thrust is a rotation of magnitude T — but it is
admissible because we only use each per-axis bound as a lower bound on that
axis's time. ``max`` across axes tightens the bound without breaking
admissibility.
"""

from __future__ import annotations

import math

from spaceace.strategies.actions import ALL_ACTIONS  # noqa: F401  (re-export use)


# Physics constants — must mirror ``src/real_physics.rs`` exactly.
GRAVITY: float = 100.0
THRUST_POWER: float = 400.0
ROTATION_SPEED_RAD_S: float = 4.363323  # ~250 deg/s
DT: float = 1.0 / 60.0

# Per-axis acceleration bounds used by the heuristic.
_A_X_MAX: float = THRUST_POWER
_A_Y_PLUS_MAX: float = THRUST_POWER + GRAVITY   # toward +y: gravity assist
_A_Y_MINUS_MAX: float = THRUST_POWER - GRAVITY  # toward -y: against gravity


def _bang_bang_axis(v_along: float, a_max: float, distance: float) -> float:
    """1D double-integrator minimum time in seconds.

    Given initial signed velocity ``v_along`` (positive = toward the target)
    and bounded scalar acceleration ``a_max`` > 0, returns the minimum time to
    cover ``distance`` >= 0 along the axis. Solves ``d = v0*t + 0.5*a*t^2`` for
    the first positive root.
    """
    if distance <= 0.0:
        return 0.0
    # Positive root of 0.5*a*t^2 + v*t - d = 0.
    disc = v_along * v_along + 2.0 * a_max * distance
    if disc < 0.0:
        # Numerically impossible with a_max, distance >= 0, but guard anyway.
        return 0.0
    return (math.sqrt(disc) - v_along) / a_max


def kinodyn_time_seconds(
    x: float,
    y: float,
    vx: float,
    vy: float,
    tx: float,
    ty: float,
) -> float:
    """Asymmetric lower bound on travel time in seconds."""
    dx = tx - x
    dy = ty - y

    # X axis: thrust can point either direction; project current vx onto the
    # direction of travel so a useful tailwind reduces the estimate.
    v_x_toward = vx if dx >= 0.0 else -vx
    t_x = _bang_bang_axis(v_x_toward, _A_X_MAX, abs(dx))

    # Y axis: asymmetric under gravity.
    if dy >= 0.0:
        v_y_toward = vy
        a_y = _A_Y_PLUS_MAX
    else:
        v_y_toward = -vy
        a_y = _A_Y_MINUS_MAX
    t_y = _bang_bang_axis(v_y_toward, a_y, abs(dy))

    return t_x if t_x > t_y else t_y


def kinodyn_time_ticks(
    x: float,
    y: float,
    vx: float,
    vy: float,
    tx: float,
    ty: float,
) -> float:
    """``kinodyn_time_seconds`` converted to 60 Hz physics ticks."""
    return kinodyn_time_seconds(x, y, vx, vy, tx, ty) / DT


def route_time_ticks(path_distance: float, cruise_speed: float) -> float:
    """Lower-bound ticks to travel ``path_distance`` pixels at ``cruise_speed``.

    Used as a floor when walls force a detour: the Euclidean bang-bang bound
    alone can under-estimate if the detour is long, so we also require the
    travel time to cover the true (wall-inflated) path length at some
    generous cruise speed. ``cruise_speed`` is an upper bound on achievable
    average velocity along the corridor — setting it too high keeps the bound
    admissible but loosens the search.
    """
    if path_distance <= 0.0:
        return 0.0
    return (path_distance / cruise_speed) / DT
