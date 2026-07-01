"""Cascaded trajectory-tracking controller for the SpaceAce kinodynamic agent.

The controller follows the phase-space reference produced by
``trajectory.build_trajectory`` using the architecture the document
prescribes for inertial agents:

* **Feedforward term** — the trajectory's kinodynamic acceleration at the
  current arc-length: ``a_ff = d(v_ref * tan) / dt`` decomposed along the
  path. Mechanically this is the acceleration the *reference* ship
  experiences; supplying it as feedforward frees the feedback loop from
  having to discover it.
* **Position feedback (Kp)** — proportional correction against position
  error between the ship and the arc-length-parameterised reference.
* **Velocity feedback (Kd)** — proportional correction against velocity
  error. Together with Kp this is a PD loop on the state error; the "D"
  label is standard because velocity error is the derivative of position
  error.
* **Gravity feedforward** — explicitly cancel gravity's contribution to the
  ship's acceleration by subtracting ``(0, g)`` from the thrust command.
  Without this the feedback loop would spend effort continuously fighting a
  known disturbance.

The ship's action set is discrete (thrust on/off, rotate L/R), so the
continuous thrust command is projected onto the action set by:

* picking the ship orientation that aligns its thrust vector with the
  commanded direction (``heading = atan2(cmd.x, -cmd.y)``), and
* firing the thruster whenever the commanded magnitude exceeds a deadband
  *and* the current heading is within tolerance of the commanded one.

SpaceAce rotation is kinematic (no angular momentum) so the rotational
channel is a direct "bang-bang" on the heading error rather than a PID —
no D gain is needed because ``d(rotation)/dt`` is commanded directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spaceace.agents.kinodyn.heuristic import GRAVITY
from spaceace.agents.kinodyn.trajectory import Trajectory

# Action indices — must match ``spaceace.strategies.actions.ALL_ACTIONS``.
_ACT_COAST = 0
_ACT_THRUST = 1
_ACT_LEFT = 2
_ACT_LEFT_THR = 3
_ACT_RIGHT = 4
_ACT_RIGHT_THR = 5

PI = math.pi
TWO_PI = 2.0 * PI


@dataclass
class ControllerConfig:
    """PD gains and dead-bands for the cascaded tracker."""

    kp_pos: float = 6.0
    """Position-error proportional gain, 1/s^2. Drives the ship back onto
    the reference when it drifts off. Pair with ``kd_vel`` chosen for a
    well-damped response."""

    kd_vel: float = 3.2
    """Velocity-error proportional gain, 1/s. Empirically near 2*sqrt(kp)
    gives critical damping on the position-velocity cascade."""

    rot_tolerance_thrust_rad: float = math.radians(18.0)
    """Fire thrust only when the heading error is inside this window.
    Wider = more aggressive thrust during turns; tighter = cleaner control
    but slower."""

    rot_deadband_rad: float = math.radians(2.5)
    """Don't command a rotation when the heading error is below this. Keeps
    the ship from chattering between L and R each frame."""

    thrust_deadband_accel: float = 40.0
    """Don't fire thrust when the commanded magnitude is below this, in
    px/s^2. Gates out dithering when the ship is already on-track."""

    lookahead_samples: int = 4
    """Number of resampled samples to look ahead when picking the reference
    point. A small lookahead yields tighter tracking; too small and the
    reference moves too slowly relative to the ship."""


def _wrap_angle(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    a = (a + PI) % TWO_PI
    if a <= 0.0:
        a += TWO_PI
    return a - PI


def _heading_for_direction(dx: float, dy: float) -> float:
    """Ship rotation that aligns its thrust vector with (dx, dy).

    Convention: rotation=0 → thrust direction (0, -1). For arbitrary
    rotation ``r``, thrust direction = (sin r, -cos r). Inverted:
    ``r = atan2(dx, -dy)``.
    """
    return math.atan2(dx, -dy)


def _advance_sample(
    traj: Trajectory,
    sample_idx: int,
    ship_xy: tuple[float, float],
    ship_v: tuple[float, float],
) -> int:
    """Advance ``sample_idx`` to the closest reference sample that is still
    ahead of the ship along the path.

    Start from the previous sample and scan forward (the path is 1-D in arc
    length so a local scan converges quickly). We advance past samples that
    are either close (within ``ds``) or strictly behind the ship's velocity
    vector — the latter catches cases where the ship overshoots a turn.
    """
    n = len(traj)
    pts = traj.pts
    tan = traj.tan
    x, y = ship_xy
    vx, vy = ship_v
    v_mag = math.hypot(vx, vy)

    i = sample_idx
    # Budget advancement so we don't scan the whole trajectory each tick.
    max_advance = max(20, traj.v[i] * 0.2 / 1.0) if n > 1 else 0
    advanced = 0
    while i < n - 1 and advanced < max_advance:
        dx = float(pts[i, 0]) - x
        dy = float(pts[i, 1]) - y
        d2 = dx * dx + dy * dy
        tx, ty = float(tan[i, 0]), float(tan[i, 1])
        # "Behind us" along the path tangent: ship is past this sample.
        dot_tan = (x - float(pts[i, 0])) * tx + (y - float(pts[i, 1])) * ty
        if dot_tan > 0.0 and d2 > 1.0:
            # Past the sample along path tangent.
            i += 1
            advanced += 1
            continue
        if d2 < 1.0:
            i += 1
            advanced += 1
            continue
        if v_mag > 25.0:
            udx = vx / v_mag
            udy = vy / v_mag
            if dx * udx + dy * udy < -1.0:
                i += 1
                advanced += 1
                continue
        break

    return min(i, n - 1)


def controller_step(
    ship_state: tuple[float, float, float, float, float],
    traj: Trajectory,
    sample_idx: int,
    cfg: ControllerConfig,
) -> tuple[int, int]:
    """Compute one discrete action and return (action_index, new_sample_idx).

    ``ship_state`` = (x, y, vx, vy, rot). ``sample_idx`` is the controller's
    previous estimate of the ship's arc-length position on the reference;
    it is advanced in-place.
    """
    x, y, vx, vy, rot = ship_state
    sample_idx = _advance_sample(traj, sample_idx, (x, y), (vx, vy))

    # Reference point: a few samples ahead to give the feedback loop headroom.
    ref_idx = min(sample_idx + cfg.lookahead_samples, len(traj) - 1)
    ref_x = float(traj.pts[ref_idx, 0])
    ref_y = float(traj.pts[ref_idx, 1])
    tan_x = float(traj.tan[ref_idx, 0])
    tan_y = float(traj.tan[ref_idx, 1])
    v_ref = float(traj.v[ref_idx])
    ref_vx = tan_x * v_ref
    ref_vy = tan_y * v_ref

    # Feedforward: reference tangential acceleration. Approximated by a
    # forward difference of the reference velocity vector along the path.
    if ref_idx + 1 < len(traj):
        next_vx = float(traj.tan[ref_idx + 1, 0]) * float(traj.v[ref_idx + 1])
        next_vy = float(traj.tan[ref_idx + 1, 1]) * float(traj.v[ref_idx + 1])
        ds = float(traj.s[ref_idx + 1] - traj.s[ref_idx])
        if v_ref > 1.0:
            dt_ref = ds / v_ref
            a_ff_x = (next_vx - ref_vx) / dt_ref
            a_ff_y = (next_vy - ref_vy) / dt_ref
        else:
            a_ff_x = 0.0
            a_ff_y = 0.0
    else:
        a_ff_x = 0.0
        a_ff_y = 0.0

    # Cascaded PD on state error.
    pos_err_x = ref_x - x
    pos_err_y = ref_y - y
    vel_err_x = ref_vx - vx
    vel_err_y = ref_vy - vy

    a_des_x = a_ff_x + cfg.kp_pos * pos_err_x + cfg.kd_vel * vel_err_x
    a_des_y = a_ff_y + cfg.kp_pos * pos_err_y + cfg.kd_vel * vel_err_y

    # Gravity feedforward. Ship's net acceleration = thrust + (0, +g), so
    # the commanded thrust that realises ``a_des`` is ``a_des - (0, +g)``
    # = (a_des.x, a_des.y - g).
    thrust_x = a_des_x
    thrust_y = a_des_y - GRAVITY

    t_mag = math.hypot(thrust_x, thrust_y)
    if t_mag < 1e-6:
        # Still rotate to maintain a hover orientation (straight up) while
        # coasting; avoids ending up broadside to gravity.
        desired_rot = 0.0
    else:
        desired_rot = _heading_for_direction(thrust_x, thrust_y)

    err_rot = _wrap_angle(desired_rot - rot)
    aligned_thrust = abs(err_rot) < cfg.rot_tolerance_thrust_rad
    want_thrust = t_mag > cfg.thrust_deadband_accel

    # Rotational channel: bang-bang on heading error (no rot momentum).
    if abs(err_rot) > cfg.rot_deadband_rad:
        if err_rot > 0.0:
            action = _ACT_RIGHT_THR if aligned_thrust and want_thrust else _ACT_RIGHT
        else:
            action = _ACT_LEFT_THR if aligned_thrust and want_thrust else _ACT_LEFT
    else:
        action = _ACT_THRUST if want_thrust else _ACT_COAST

    return action, sample_idx
