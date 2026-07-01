"""TAS-style segment polisher.

Does a first MCTS pass end-to-end (or loads an existing action trace), segments
the run on pickup-collection events, then re-runs each segment from its start
snapshot with a portfolio of more aggressive MCTS settings. If a candidate
window yields a shorter validated completion, its actions replace that part of
the route. Finally the stitched action list is replayed once to capture frames
and the result is saved as the "ai" ghost for the level.

Usage:
    uv run python scripts/tas_polish.py --level 6
    uv run python scripts/tas_polish.py --level 6 --base-sims 3000 --polish-sims 20000 --polish-passes 2
    uv run python scripts/tas_polish.py --level 6 --portfolio wide --window-size 3 --polish-seeds 2
    uv run python scripts/tas_polish.py --input-json run.json --dump-json polished.json --no-save
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from numbers import Integral
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spaceace_rl  # noqa: E402
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.strategies.actions import ALL_ACTIONS  # noqa: E402

ACTION_TO_INDEX = {
    tuple(int(x) for x in action.tolist()): idx
    for idx, action in enumerate(ALL_ACTIONS)
}


@dataclass
class Segment:
    """One contiguous run of actions, terminated by a pickup collection (or EOE).

    `start_state` is the opaque PyGameState snapshot saved just before the first
    action of the segment. `action_indices` are ints into ALL_ACTIONS. `ticks`
    is the length in physics ticks == len(action_indices). `pickups_before` and
    `pickups_after` bracket the pickup count at segment boundaries.
    """
    start_state: object
    action_indices: list[int]
    pickups_before: int
    pickups_after: int

    @property
    def ticks(self) -> int:
        return len(self.action_indices)


@dataclass(frozen=True)
class SearchProfile:
    """One MCTS parameter set in the polishing portfolio."""
    name: str
    sims_multiplier: float = 1.0
    exploration_multiplier: float = 1.0
    action_repeat_delta: int = 0
    shaping_weight: float = 0.5
    goofy: bool = False
    thrust_bias: float = 0.0
    thrust_bias_safe_dist: float = 0.0
    ar_depth_bonus: int = 0
    ar_max: int = 20
    widen_k: float = 0.0


DEFAULT_PROFILE = SearchProfile(name="baseline")


def flatten_segments(segments: list[Segment]) -> list[int]:
    return [a for seg in segments for a in seg.action_indices]


def _seed_mcts(seed: Optional[int]) -> None:
    if seed is not None:
        spaceace_rl.set_rng_seed(int(seed) & 0xFFFFFFFF)


def _decode_action_item(item: Any, idx: int) -> int:
    """Accept either an action index or a raw [left, right, thrust] triplet."""
    if isinstance(item, Integral) and not isinstance(item, bool):
        action_idx = int(item)
        if 0 <= action_idx < len(ALL_ACTIONS):
            return action_idx
        raise ValueError(f"action {idx}: index {action_idx} is outside 0..{len(ALL_ACTIONS) - 1}")

    if isinstance(item, (list, tuple)) and len(item) == 3:
        try:
            raw = tuple(int(x) for x in item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action {idx}: raw triplet must contain integers") from exc
        action_idx = ACTION_TO_INDEX.get(raw)
        if action_idx is not None:
            return action_idx
        raise ValueError(f"action {idx}: raw triplet {list(raw)} is not in the 6-action set")

    raise ValueError(f"action {idx}: expected an index or [left, right, thrust] triplet")


def load_action_file(path: Path) -> tuple[Optional[int], list[int]]:
    """Load raw TAS JSON or solver JSON and normalize to action indices."""
    data = json.loads(path.read_text())
    level: Optional[int] = None

    if isinstance(data, dict):
        if data.get("level") is not None:
            level = int(data["level"])
        raw_actions = data.get("action_indices")
        if raw_actions is None:
            raw_actions = data.get("actions")
        if raw_actions is None:
            raw_actions = data.get("raw_actions")
    elif isinstance(data, list):
        raw_actions = data
    else:
        raise ValueError("action JSON must be a list or an object with an actions field")

    if not isinstance(raw_actions, list):
        raise ValueError("action JSON must contain a list of actions")

    return level, [_decode_action_item(item, i) for i, item in enumerate(raw_actions)]


_ROTATION_SPEED_RAD_PER_SEC = 4.363323  # matches src/real_physics.rs ROTATION_SPEED
_TICK_HZ = 60.0
_ROT_PER_TICK = _ROTATION_SPEED_RAD_PER_SEC / _TICK_HZ


GHOST_ACTIONS_DIR = PROJECT_ROOT / "ghost_actions"


def _ghost_actions_path(level: int, ghost_type: str) -> Path:
    return GHOST_ACTIONS_DIR / f"L{level}_{ghost_type}.json"


def load_ghost_actions(level: int, ghost_type: str) -> Optional[list[int]]:
    """Load exact actions for a ghost from the sidecar file, if present.

    Sidecar files are written by save_ghost when the polisher saves a run, so
    re-seeding from a ghost stays exact instead of losing precision through
    frame reconstruction. Returns None if no sidecar exists, in which case the
    caller should fall back to reconstructing from frames.
    """
    path = _ghost_actions_path(level, ghost_type)
    if not path.exists():
        return None
    _level, actions = load_action_file(path)
    return actions


def load_ghost_frames(level: int, ghost_type: str) -> list[dict]:
    """Read a ghost for a level/type from the dashboard DB."""
    from dashboard.db import get_db, init_db
    init_db()
    db = get_db()
    try:
        row = db.execute(
            "SELECT frames_json, time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = ?",
            (level, ghost_type),
        ).fetchone()
    finally:
        db.close()
    if row is None:
        raise ValueError(f"no {ghost_type} ghost found for level {level}")
    frames = json.loads(row["frames_json"])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{ghost_type} ghost for level {level} has no frames")
    return frames


def load_human_ghost_frames(level: int) -> list[dict]:
    return load_ghost_frames(level, "human")


def _wrap_to_pi(angle: float) -> float:
    """Map angle to [-pi, pi] so rotation deltas across the seam are minimal."""
    import math
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def reconstruct_actions_from_ghost(frames: list[dict]) -> list[int]:
    """Convert ghost frames (~10Hz position/rotation/thrust) into a 60Hz action trace.

    For each pair of consecutive frames, compute the rotation delta and split
    the interval into rotation ticks (left or right depending on sign) followed
    by non-rotation ticks. Thrust bool is taken from the *earlier* frame and
    held across the interval. Reconstruction is approximate: the resulting
    trajectory will drift from the human's because rotation is interleaved
    with translation in the original, but the polisher takes over from there.
    """
    if not frames:
        return []
    # Accumulate ticks from cumulative time so sub-tick toggle frames (the
    # recorder emits an extra frame every time `thrusting` flips, even if
    # only ~0.02s has passed) don't each get rounded UP to a full tick. That
    # bug inflated the reconstructed trace by ~10% on toggle-heavy runs.
    t0_base = float(frames[0]["time"])
    actions: list[int] = []
    cumulative_ticks = 0
    for i in range(len(frames) - 1):
        f0 = frames[i]
        f1 = frames[i + 1]
        target_ticks = int(round((float(f1["time"]) - t0_base) * _TICK_HZ))
        ticks = max(0, target_ticks - cumulative_ticks)
        cumulative_ticks = target_ticks
        if ticks == 0:
            continue
        delta_rot = _wrap_to_pi(float(f1["rotation"]) - float(f0["rotation"]))
        rot_ticks = min(ticks, int(round(abs(delta_rot) / _ROT_PER_TICK)))
        rot_dir = 1 if delta_rot > 0 else -1  # +1 = right (rot += speed*dt), -1 = left
        thrust = 1 if f0.get("thrusting") else 0
        for _ in range(rot_ticks):
            if rot_dir > 0:
                triplet = (0, 1, thrust)  # rotate right
            else:
                triplet = (1, 0, thrust)  # rotate left
            actions.append(ACTION_TO_INDEX[triplet])
        for _ in range(ticks - rot_ticks):
            triplet = (0, 0, thrust)
            actions.append(ACTION_TO_INDEX[triplet])
    return actions


def dump_action_file(path: Path, level: int, action_indices: list[int], ticks: int) -> None:
    payload = {
        "level": level,
        "ticks": ticks,
        "seconds": round(ticks / 60.0, 3),
        "action_format": "indices",
        "actions": action_indices,
        "raw_actions": [ALL_ACTIONS[a].astype(int).tolist() for a in action_indices],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_polish_profiles(
    portfolio: str,
    base_profile: SearchProfile,
    *,
    action_repeat: int,
) -> list[SearchProfile]:
    """Build MCTS variants that trade safety, precision, and thrust pressure."""
    if portfolio == "single":
        return [base_profile]

    eager = replace(
        base_profile,
        name="eager",
        sims_multiplier=0.8,
        action_repeat_delta=-1,
        thrust_bias=max(base_profile.thrust_bias, 1.0),
        thrust_bias_safe_dist=max(base_profile.thrust_bias_safe_dist, 140.0),
        widen_k=max(base_profile.widen_k, 1.15),
    )
    deep_fast = replace(
        base_profile,
        name="deep-fast",
        sims_multiplier=0.9,
        exploration_multiplier=0.9,
        action_repeat_delta=1,
        shaping_weight=min(base_profile.shaping_weight, 0.35),
        thrust_bias=max(base_profile.thrust_bias, 1.8),
        thrust_bias_safe_dist=max(base_profile.thrust_bias_safe_dist, 180.0),
        ar_depth_bonus=max(base_profile.ar_depth_bonus, 1),
        ar_max=max(base_profile.ar_max, action_repeat + 12, 20),
        widen_k=max(base_profile.widen_k, 1.3),
    )
    profiles = [
        replace(base_profile, name="baseline"),
        eager,
        deep_fast,
    ]

    if portfolio == "wide":
        profiles.extend([
            replace(
                base_profile,
                name="explore",
                sims_multiplier=0.85,
                exploration_multiplier=1.35,
                action_repeat_delta=-2,
                thrust_bias=max(base_profile.thrust_bias, 0.6),
                thrust_bias_safe_dist=max(base_profile.thrust_bias_safe_dist, 120.0),
            ),
            replace(
                base_profile,
                name="goofy",
                sims_multiplier=0.7,
                action_repeat_delta=-1,
                goofy=True,
                thrust_bias=0.0,
                thrust_bias_safe_dist=0.0,
                widen_k=max(base_profile.widen_k, 1.0),
            ),
        ])

    return profiles


def segment_existing_actions(
    env: SpaceAceDirectEnv,
    action_indices: list[int],
    *,
    max_ticks: int,
) -> tuple[list[Segment], bool]:
    """Replay an existing action trace and split it on pickup boundaries."""
    env.reset()
    segments: list[Segment] = []
    seg_start = env.save_state()
    seg_actions: list[int] = []
    pickups_before = env.get_pickups_remaining()

    for tick, action_idx in enumerate(action_indices, start=1):
        if tick > max_ticks:
            break

        _obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        seg_actions.append(action_idx)
        pickups_after = env.get_pickups_remaining()
        segment_done = (
            pickups_after < pickups_before
            or terminated
            or truncated
            or bool(info.get("level_completed"))
        )
        if not segment_done:
            continue

        segments.append(Segment(
            start_state=seg_start,
            action_indices=seg_actions,
            pickups_before=pickups_before,
            pickups_after=pickups_after,
        ))
        seg_actions = []
        if pickups_after == 0 or terminated or truncated:
            break

        seg_start = env.save_state()
        pickups_before = pickups_after

    if seg_actions:
        segments.append(Segment(
            start_state=seg_start,
            action_indices=seg_actions,
            pickups_before=pickups_before,
            pickups_after=env.get_pickups_remaining(),
        ))

    completed = env.get_pickups_remaining() == 0
    return segments, completed


def _mcts_run(
    env: SpaceAceDirectEnv,
    mcts: "spaceace_rl.PyMCTSEngine",
    start_state,
    num_simulations: int,
    exploration: float,
    gamma: float,
    action_repeat: int,
    profile: SearchProfile = DEFAULT_PROFILE,
    stop_on_pickup: bool = False,
    stop_pickups_remaining: Optional[int] = None,
    max_ticks: Optional[int] = None,
    early_exit: bool = True,
) -> tuple[list[int], bool, int]:
    """Replay MCTS from a given snapshot. Returns (action_indices, terminated, pickups_lost).

    If `stop_on_pickup` is True, returns as soon as ANY pickup is collected (the
    returned action list ends on the tick that grabbed it). This is how we slice
    the run into pickup-bounded segments for polishing.
    """
    env.load_state(start_state)
    mcts.reset_tree_cache()
    pickups_at_start = env.get_pickups_remaining()
    target_pickups_remaining = stop_pickups_remaining
    if stop_on_pickup and target_pickups_remaining is None:
        target_pickups_remaining = pickups_at_start - 1

    actions_out: list[int] = []
    pending_action: Optional[int] = None
    pending_repeats = 0
    terminated = False

    while True:
        if max_ticks is not None and len(actions_out) >= max_ticks:
            break

        if pending_repeats > 0:
            # Execute queued macro-action step
            assert pending_action is not None
            _, _, terminated, truncated, info = env.step(ALL_ACTIONS[pending_action])
            actions_out.append(pending_action)
            pending_repeats -= 1
            if terminated or truncated:
                break
            if (
                target_pickups_remaining is not None
                and env.get_pickups_remaining() <= target_pickups_remaining
            ):
                break
            continue

        # Time for a fresh MCTS decision
        current_state = env.save_state()

        obs = env.get_observation()
        speed = float((obs[2] ** 2 + obs[3] ** 2) ** 0.5)
        min_wall_dist = float(min(obs[8:16]))
        base_ar = max(1, action_repeat + profile.action_repeat_delta)
        # Cap speed-based AR inflation so decisions stay fine-grained at cruise:
        # the previous /50 with no cap pushed ar past 10 at modest speeds, which
        # made the polisher unable to micro-adjust approaches near pickups/walls.
        ar = base_ar + min(3, int(speed / 120.0))
        # Near walls, drop AR for finer control instead of (only) more sims.
        if min_wall_dist < 100.0:
            ar = max(1, ar - 1)
        sims = max(1, int(num_simulations * profile.sims_multiplier))
        if min_wall_dist < 150.0:
            sims = int(sims * (1.0 + (150.0 - min_wall_dist) / 150.0))
        sims = int(sims * (1.0 + speed / 300.0))

        ee_check = 500 if early_exit else 0
        action_idx, _stats, _rh = mcts.search_with_reuse(
            current_state, sims, ar,
            exploration * profile.exploration_multiplier, gamma,
            profile.shaping_weight,
            profile.goofy,
            profile.thrust_bias,
            profile.ar_depth_bonus,
            profile.ar_max,
            profile.widen_k,
            profile.thrust_bias_safe_dist,
            ee_check, 0.7, 10.0,
        )
        if not 0 <= int(action_idx) < len(ALL_ACTIONS):
            break
        pending_action = int(action_idx)
        pending_repeats = ar
        # Loop iterates and executes immediately.

    pickups_lost = pickups_at_start - env.get_pickups_remaining()
    return actions_out, terminated, pickups_lost


_N_ACTIONS = len(ALL_ACTIONS)


def _init_pi(seed_actions: list[int], length: int, init_smooth: float) -> list[list[float]]:
    """Per-tick categorical distribution. Inside the seed range, mass concentrates
    on the seed action; past the seed (the slack tail), the row is uniform so CEM
    can extend the trajectory without prior bias."""
    pi: list[list[float]] = []
    off_seed = init_smooth / max(1, _N_ACTIONS - 1)
    for t in range(length):
        if t < len(seed_actions):
            row = [off_seed] * _N_ACTIONS
            row[seed_actions[t]] = 1.0 - init_smooth
        else:
            row = [1.0 / _N_ACTIONS] * _N_ACTIONS
        pi.append(row)
    return pi


def _sample_sequence(pi: list[list[float]], rng: random.Random) -> list[int]:
    return [rng.choices(range(_N_ACTIONS), weights=row, k=1)[0] for row in pi]


def _cem_run(
    env: SpaceAceDirectEnv,
    start_state,
    seed_actions: list[int],
    target_pickups_remaining: int,
    max_ticks: int,
    *,
    samples: int = 128,
    elite_frac: float = 0.15,
    iterations: int = 4,
    init_smooth: float = 0.1,
    refit_blend: float = 0.4,
    action_repeat: int = 1,
    rng_seed: Optional[int] = None,
) -> tuple[list[int], bool, int]:
    """Cross-entropy method on per-tick action distributions.

    Mirrors `_mcts_run`'s output contract: returns the shortest sampled sequence
    that drives `pickups_remaining` down to (or below) `target_pickups_remaining`,
    along with a `terminated` flag and `pickups_lost` count. Env is left at the
    end-of-best-sequence state so callers can read `get_pickups_remaining()` and
    snapshot for tail re-solve. If no sample reached the target, returns an
    empty action list with env loaded back to `start_state`.

    The first sample of the first iteration is the raw seed itself, so the elite
    set is never empty — that's what keeps the refit signal from collapsing in
    early iterations when most random samples never reach a pickup.
    """
    rng = random.Random(rng_seed)
    K = max(1, action_repeat)
    L_ticks = max(1, min(max_ticks, len(seed_actions) + max(20, len(seed_actions) // 3)))
    L_macro = max(1, (L_ticks + K - 1) // K)
    # Downsample the seed to macro-resolution (one decision per K ticks). Tail
    # past the seed stays uniform so CEM can extend the trajectory.
    seed_macro = [
        seed_actions[i * K]
        for i in range(min(L_macro, len(seed_actions) // K + (1 if len(seed_actions) % K else 0)))
        if i * K < len(seed_actions)
    ]
    pi = _init_pi(seed_macro, L_macro, init_smooth)

    env.load_state(start_state)
    pickups_at_start = env.get_pickups_remaining()

    best_actions: list[int] = []
    best_terminated = False
    best_ticks = max_ticks + 1

    for it in range(iterations):
        scored: list[tuple[tuple, list[int]]] = []
        for k in range(samples):
            if it == 0 and k == 0 and seed_macro:
                seq_macro = list(seed_macro[:L_macro])
                if len(seq_macro) < L_macro:
                    seq_macro.extend(
                        rng.choices(range(_N_ACTIONS), weights=pi[t], k=1)[0]
                        for t in range(len(seq_macro), L_macro)
                    )
            else:
                seq_macro = _sample_sequence(pi, rng)

            env.load_state(start_state)
            actions_taken: list[int] = []
            terminated = False
            reached = False
            min_pickups = pickups_at_start
            done = False
            for macro_a in seq_macro:
                for _ in range(K):
                    if len(actions_taken) >= max_ticks:
                        done = True
                        break
                    _obs, _r, term, trunc, _info = env.step(ALL_ACTIONS[macro_a])
                    actions_taken.append(macro_a)
                    pr = env.get_pickups_remaining()
                    if pr < min_pickups:
                        min_pickups = pr
                    if pr <= target_pickups_remaining:
                        reached = True
                        done = True
                        break
                    if term or trunc:
                        terminated = True
                        done = True
                        break
                if done:
                    break

            # Score: feasible-first, then by ticks. Infeasible samples sort by
            # closest-approach (lower min_pickups = more progress) so the elite
            # set carries gradient signal even when no sample reached the goal.
            if reached:
                score = (0, len(actions_taken))
            else:
                score = (1, min_pickups, max_ticks - len(actions_taken))
            scored.append((score, actions_taken))

            if reached and len(actions_taken) < best_ticks:
                best_ticks = len(actions_taken)
                best_actions = actions_taken
                best_terminated = terminated

        scored.sort(key=lambda r: r[0])
        n_elite = max(2, int(samples * elite_frac))
        elite = scored[:n_elite]

        new_pi: list[list[float]] = []
        for m in range(L_macro):
            tick_idx = m * K
            counts = [0.0] * _N_ACTIONS
            n = 0
            for _, acts in elite:
                if tick_idx < len(acts):
                    counts[acts[tick_idx]] += 1.0
                    n += 1
            if n == 0:
                new_pi.append(pi[m])
                continue
            empirical = [c / n for c in counts]
            blended = [
                (1.0 - refit_blend) * pi[m][i]
                + refit_blend * (0.95 * empirical[i] + 0.05 / _N_ACTIONS)
                for i in range(_N_ACTIONS)
            ]
            s = sum(blended)
            new_pi.append([v / s for v in blended])
        pi = new_pi

    if best_actions:
        env.load_state(start_state)
        for a in best_actions:
            _obs, _r, term, trunc, _info = env.step(ALL_ACTIONS[a])
            if term or trunc:
                break
    else:
        env.load_state(start_state)

    pickups_lost = pickups_at_start - env.get_pickups_remaining()
    return best_actions, best_terminated, pickups_lost


def _estimate_cruise_speed(
    env: SpaceAceDirectEnv,
    segments: list[Segment],
    *,
    fallback_px_per_tick: float = 4.0,
) -> float:
    """Replay segments and return the peak observed |velocity| in px/tick.

    This sets a (loose) physical lower bound on per-tick travel: if the ship
    never reached more than V px/tick during the source pass, no candidate
    starting from the same snapshot can travel a given distance in fewer than
    `distance / V` ticks. Used to estimate per-segment optimality, not as a
    hard target.
    """
    peak = 0.0
    for seg in segments:
        env.load_state(seg.start_state)
        obs = env.get_observation()
        speed = float((obs[2] ** 2 + obs[3] ** 2) ** 0.5)
        if speed > peak:
            peak = speed
        for a_idx in seg.action_indices:
            obs, _, terminated, truncated, _ = env.step(ALL_ACTIONS[a_idx])
            speed = float((obs[2] ** 2 + obs[3] ** 2) ** 0.5)
            if speed > peak:
                peak = speed
            if terminated or truncated:
                break
    if peak <= 0.0:
        return fallback_px_per_tick
    # Engine velocities are in px/sec (real_physics.rs: x += vx * dt with
    # dt = 1/60). Convert to px/tick to match the lower-bound math.
    return peak / _TICK_HZ


_PATHFINDER_DISTANCE_CACHE: dict[int, float] = {}


def _segment_pathfinder_distance(
    mcts: "spaceace_rl.PyMCTSEngine",
    segment: Segment,
) -> float:
    """Pathfinder distance from a segment's start snapshot to the nearest
    uncollected pickup. Cached by segment instance id — segments are replaced
    (not mutated) on accept, so id-based keying stays correct."""
    key = id(segment)
    cached = _PATHFINDER_DISTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    distance, _, _ = mcts.get_pathfinder_stats(segment.start_state)
    value = float(distance)
    _PATHFINDER_DISTANCE_CACHE[key] = value
    return value


def _segment_lower_bound_ticks(
    mcts: "spaceace_rl.PyMCTSEngine",
    segment: Segment,
    cruise_px_per_tick: float,
) -> float:
    """Estimate the minimum ticks needed to reach the next pickup from the
    segment's starting snapshot. Uses pathfinder distance to the nearest
    uncollected pickup ÷ cruise speed. Loose (assumes straight-line at peak
    speed with no rotation), but consistent across segments — good enough
    for ranking which segments have the most slack to recover."""
    distance = _segment_pathfinder_distance(mcts, segment)
    return max(1.0, distance / max(1e-3, cruise_px_per_tick))


def _segment_waste_score(
    mcts: "spaceace_rl.PyMCTSEngine",
    segment: Segment,
    cruise_px_per_tick: float,
) -> float:
    """Higher = more potentially-recoverable ticks. Used to prioritize polish."""
    lb = _segment_lower_bound_ticks(mcts, segment, cruise_px_per_tick)
    return float(segment.ticks) - lb


def _replay_actions(env: SpaceAceDirectEnv, start_state, action_indices: list[int]) -> int:
    """Load snapshot and step through the action list, returning ticks executed."""
    env.load_state(start_state)
    for i, a in enumerate(action_indices):
        _, _, terminated, truncated, _ = env.step(ALL_ACTIONS[a])
        if terminated or truncated:
            return i + 1
    return len(action_indices)


def first_pass(
    env: SpaceAceDirectEnv,
    mcts: "spaceace_rl.PyMCTSEngine",
    *,
    base_sims: int,
    exploration: float,
    gamma: float,
    action_repeat: int,
    max_ticks: int,
    profile: SearchProfile = DEFAULT_PROFILE,
) -> tuple[list[Segment], bool]:
    env.reset()
    mcts.reset_tree_cache()
    return _segment_from_here(
        env, mcts,
        num_simulations=base_sims,
        exploration=exploration,
        gamma=gamma,
        action_repeat=action_repeat,
        profile=profile,
        budget_ticks=max_ticks,
    )


def _segment_from_here(
    env: SpaceAceDirectEnv,
    mcts: "spaceace_rl.PyMCTSEngine",
    *,
    num_simulations: int,
    exploration: float,
    gamma: float,
    action_repeat: int,
    budget_ticks: int,
    profile: SearchProfile = DEFAULT_PROFILE,
) -> tuple[list[Segment], bool]:
    """Keep playing from current env state, splitting on pickups, until the
    level is complete or we hit `budget_ticks`. Each returned Segment starts
    from the snapshot taken at its beginning."""
    segments: list[Segment] = []
    total = 0
    while total < budget_ticks and env.get_pickups_remaining() > 0:
        seg_start = env.save_state()
        pickups_before = env.get_pickups_remaining()
        actions, terminated, _pl = _mcts_run(
            env, mcts, seg_start,
            num_simulations=num_simulations,
            exploration=exploration,
            gamma=gamma,
            action_repeat=action_repeat,
            profile=profile,
            stop_on_pickup=True,
            max_ticks=budget_ticks - total,
        )
        if not actions:
            break
        pickups_after = env.get_pickups_remaining()
        segments.append(Segment(
            start_state=seg_start,
            action_indices=actions,
            pickups_before=pickups_before,
            pickups_after=pickups_after,
        ))
        total += len(actions)
        if terminated:
            break
        if pickups_after >= pickups_before:
            break
    completed = env.get_pickups_remaining() == 0
    return segments, completed


def _candidate_seed(
    seed_base: Optional[int],
    *,
    segment_idx: int,
    window_len: int,
    profile_idx: int,
    seed_idx: int,
) -> Optional[int]:
    if seed_base is None:
        return None
    return seed_base + segment_idx * 1009 + window_len * 101 + profile_idx * 37 + seed_idx


def _try_polish_at(
    env: SpaceAceDirectEnv,
    mcts: "spaceace_rl.PyMCTSEngine",
    segments: list[Segment],
    idx: int,
    *,
    polish_sims: int,
    base_sims: int,
    exploration: float,
    gamma: float,
    action_repeat: int,
    tail_profile: SearchProfile,
    profiles: list[SearchProfile],
    max_window_size: int,
    window_slack: float,
    window_slack_ticks: int,
    polish_seeds: int,
    tail_seeds: int,
    seed_base: Optional[int],
    budget_ticks: int,
    first_accept: bool = False,
    polish_method: str = "mcts",
    cem_samples: int = 128,
    cem_iterations: int = 4,
    cem_elite_frac: float = 0.15,
    cem_action_repeat: int = 1,
    reject_counts: Optional[Counter[str]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[list[Segment]]:
    """Try several MCTS profiles and pickup windows, keeping the shortest replay.

    A window may cover more than one pickup segment. This lets the polisher
    accept racing lines that are not locally faster to the first pickup but are
    faster by the time the downstream tail is replayed.
    """
    def reject(reason: str) -> Optional[list[Segment]]:
        if reject_counts is not None:
            reject_counts[reason] += 1
        return None

    def record(reason: str) -> str:
        if reject_counts is not None:
            reject_counts[reason] += 1
        return reason

    seg = segments[idx]
    old_total_ticks = sum(s.ticks for s in segments)
    old_prefix_ticks = sum(s.ticks for s in segments[:idx])
    old_tail_ticks = old_total_ticks - old_prefix_ticks

    if seg.ticks <= 1:
        return reject("single-tick-segment")

    if polish_method == "cem":
        # CEM is deterministic given seed; the MCTS-portfolio dimension
        # (profiles) buys nothing here. Collapse to a single sentinel so the
        # outer loop structure (windows × seeds × profiles) still works.
        profiles = [DEFAULT_PROFILE]

    if not profiles:
        return reject("no-profiles")

    best_segments: Optional[list[Segment]] = None
    best_total_ticks = old_total_ticks
    last_reason: Optional[str] = None
    max_end = min(len(segments), idx + max(1, max_window_size))

    attempts_total = 0
    attempts_done = 0
    accepts = 0
    for end_idx in range(max_end - 1, idx - 1, -1):
        attempts_total += max(1, polish_seeds) * len(profiles)

    accepted_early = False
    for end_idx in range(max_end - 1, idx - 1, -1):
        if accepted_early:
            break
        window = segments[idx:end_idx + 1]
        window_len = len(window)
        old_window_ticks = sum(s.ticks for s in window)
        # Slack is the larger of (slack% × window) and an absolute tick floor.
        # On short windows the percentage alone gives only a handful of ticks of
        # search budget — too little to discover any racing-line that briefly
        # overshoots before saving time downstream.
        max_window_ticks = old_window_ticks + max(
            window_slack_ticks,
            int(old_window_ticks * window_slack),
        )
        target_remaining = window[-1].pickups_after

        for seed_idx in range(max(1, polish_seeds)):
            if accepted_early:
                break
            for profile_idx, profile in enumerate(profiles):
                if accepted_early:
                    break
                attempts_done += 1
                if progress_cb is not None:
                    progress_cb(
                        f"w{window_len} {profile.name} s{seed_idx+1} "
                        f"({attempts_done}/{attempts_total} acc={accepts})"
                    )
                cand_seed = _candidate_seed(
                    seed_base,
                    segment_idx=idx,
                    window_len=window_len,
                    profile_idx=profile_idx,
                    seed_idx=seed_idx,
                )
                _seed_mcts(cand_seed)
                if polish_method == "cem":
                    window_seed_actions = [
                        a for s in window for a in s.action_indices
                    ]
                    candidate_actions, _terminated, _pl = _cem_run(
                        env, seg.start_state,
                        seed_actions=window_seed_actions,
                        target_pickups_remaining=target_remaining,
                        max_ticks=max_window_ticks,
                        samples=cem_samples,
                        iterations=cem_iterations,
                        elite_frac=cem_elite_frac,
                        action_repeat=cem_action_repeat,
                        rng_seed=cand_seed,
                    )
                else:
                    candidate_actions, _terminated, _pl = _mcts_run(
                        env, mcts, seg.start_state,
                        num_simulations=polish_sims,
                        exploration=exploration,
                        gamma=gamma,
                        action_repeat=action_repeat,
                        profile=profile,
                        stop_pickups_remaining=target_remaining,
                        max_ticks=max_window_ticks,
                        early_exit=False,  # spend the full polish budget
                    )
                if not candidate_actions:
                    last_reason = record("no-candidate")
                    continue

                pickups_after = env.get_pickups_remaining()
                if pickups_after > target_remaining:
                    last_reason = record("missed-window-pickups")
                    continue

                remaining_budget = budget_ticks - old_prefix_ticks - len(candidate_actions)
                if remaining_budget <= 0 and pickups_after > 0:
                    last_reason = record("no-tail-budget")
                    continue

                if pickups_after == 0:
                    new_tail: list[Segment] = []
                    completed = True
                else:
                    # Snapshot the post-polish state once, then re-solve the tail
                    # under multiple seeds and keep the shortest completed run.
                    # Single-seed tail noise was the dominant rejection reason —
                    # MCTS variance on the rebuilt tail routinely made a faster
                    # local window look like a regression overall.
                    post_window_state = env.save_state()
                    new_tail = []
                    completed = False
                    best_tail_ticks = None
                    for tail_seed_idx in range(max(1, tail_seeds)):
                        env.load_state(post_window_state)
                        _seed_mcts(_candidate_seed(
                            seed_base,
                            segment_idx=idx,
                            window_len=window_len,
                            profile_idx=profile_idx,
                            seed_idx=seed_idx * 31 + tail_seed_idx + 1,
                        ))
                        cand_tail, cand_done = _segment_from_here(
                            env, mcts,
                            num_simulations=base_sims,
                            exploration=exploration,
                            gamma=gamma,
                            action_repeat=action_repeat,
                            budget_ticks=max(0, remaining_budget),
                            profile=tail_profile,
                        )
                        if not cand_done:
                            continue
                        cand_ticks = sum(s.ticks for s in cand_tail)
                        if best_tail_ticks is None or cand_ticks < best_tail_ticks:
                            best_tail_ticks = cand_ticks
                            new_tail = cand_tail
                            completed = True
                if not completed:
                    last_reason = record("tail-incomplete")
                    continue

                new_seg = Segment(
                    start_state=seg.start_state,
                    action_indices=candidate_actions,
                    pickups_before=seg.pickups_before,
                    pickups_after=pickups_after,
                )
                new_tail_ticks = new_seg.ticks + sum(s.ticks for s in new_tail)
                if new_tail_ticks >= old_tail_ticks:
                    last_reason = record("tail-not-shorter")
                    continue

                candidate_segments = segments[:idx] + [new_seg] + new_tail
                new_total_ticks = old_prefix_ticks + new_tail_ticks
                if new_total_ticks < best_total_ticks:
                    best_total_ticks = new_total_ticks
                    best_segments = candidate_segments
                    accepts += 1
                    record(f"accepted:{profile.name}/w{window_len}")
                    if progress_cb is not None:
                        progress_cb(
                            f"w{window_len} {profile.name} s{seed_idx+1} "
                            f"-{old_total_ticks - new_total_ticks}t ✓ "
                            f"({attempts_done}/{attempts_total} acc={accepts})"
                        )
                    if first_accept:
                        accepted_early = True
                        break

    if best_segments is None:
        return None if last_reason is not None else reject("no-improvement")

    return best_segments


def stitch_and_capture(
    env: SpaceAceDirectEnv,
    segments: list[Segment],
) -> tuple[list[dict], int, bool]:
    """Replay all segments in order from a fresh reset, capturing one frame per
    physics tick. Returns (frames, final_tick_count, completed)."""
    env.reset()
    frames: list[dict] = []
    total_ticks = 0
    completed = False

    for seg in segments:
        # Sanity: env state should equal seg.start_state here. We trust the
        # deterministic engine and skip the comparison.
        for a_idx in seg.action_indices:
            action = ALL_ACTIONS[a_idx]
            obs, _r, terminated, truncated, info = env.step(action)
            total_ticks += 1
            frames.append({
                "x": round(float(obs[0]), 1),
                "y": round(float(obs[1]), 1),
                "rotation": round(float(obs[4]), 3),
                "thrusting": int(action[2]) > 0,
                "tick": total_ticks,
            })
            if info.get("level_completed"):
                completed = True
            if terminated or truncated:
                return frames, total_ticks, completed

    completed = env.get_pickups_remaining() == 0
    return frames, total_ticks, completed


def build_ghost_frames(frames: list[dict]) -> list[dict]:
    """Downsample to ~10 Hz keyed on physics ticks (matches capture_ai_ghost.py)."""
    ghost_frames: list[dict] = []
    target_stride = 6
    next_emit_tick = 0
    last_idx = len(frames) - 1
    for i, f in enumerate(frames):
        tick = int(f.get("tick", i))
        if tick >= next_emit_tick or i == last_idx:
            ghost_frames.append({
                "x": f["x"], "y": f["y"],
                "rotation": f["rotation"],
                "thrusting": f["thrusting"],
                "time": round(tick / 60.0, 3),
            })
            next_emit_tick = tick + target_stride
    return ghost_frames


def _maybe_write_sidecar(
    level: int,
    ghost_type: str,
    time_seconds: float,
    action_indices: list[int],
    final_ticks: int,
) -> None:
    """Sidecar lifecycle is independent of the DB row: we want to write actions
    whenever the new run is faster than any sidecar we have, even if the DB
    already holds a faster ghost (e.g. a pre-sidecar run from before this
    feature shipped). That's the only way to bootstrap exact seeding for old
    ghosts whose actions weren't persisted at save time."""
    sidecar = _ghost_actions_path(level, ghost_type)
    existing_seconds: Optional[float] = None
    if sidecar.exists():
        try:
            existing_seconds = float(json.loads(sidecar.read_text()).get("seconds"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_seconds = None
    if existing_seconds is not None and existing_seconds <= time_seconds:
        print(f"[save] existing {ghost_type} sidecar is faster "
              f"({existing_seconds:.2f}s <= {time_seconds:.2f}s); not overwriting")
        return
    dump_action_file(sidecar, level, action_indices, final_ticks)
    prev = f" (prev {existing_seconds:.2f}s)" if existing_seconds is not None else ""
    print(f"[save] wrote {ghost_type} action sidecar ({time_seconds:.2f}s{prev}): {sidecar}")


def save_ghost(
    level: int,
    ghost_type: str,
    time_seconds: float,
    ghost_frames: list[dict],
    action_indices: Optional[list[int]] = None,
    final_ticks: Optional[int] = None,
) -> None:
    from dashboard.db import get_db, init_db
    init_db()
    db = get_db()
    try:
        existing = db.execute(
            "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = ?",
            (level, ghost_type),
        ).fetchone()
        if existing and existing["time_seconds"] <= time_seconds:
            print(f"[save] existing {ghost_type} ghost is faster "
                  f"({existing['time_seconds']:.2f}s <= {time_seconds:.2f}s); not overwriting DB row")
        else:
            db.execute(
                """INSERT OR REPLACE INTO ghost_replays
                   (level, ghost_type, steps, time_seconds, frames_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (level, ghost_type, len(ghost_frames), time_seconds, json.dumps(ghost_frames)),
            )
            db.commit()
            prev = f" (prev {existing['time_seconds']:.2f}s)" if existing else ""
            print(f"[save] wrote {ghost_type} level {level}: "
                  f"{len(ghost_frames)} frames, {time_seconds:.2f}s{prev}")
    finally:
        db.close()

    if action_indices is not None:
        ticks = final_ticks if final_ticks is not None else len(action_indices)
        _maybe_write_sidecar(level, ghost_type, time_seconds, action_indices, ticks)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--level", type=int, default=None,
                   help="Game level. Optional when --input-json contains a level.")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--base-sims", type=int, default=3000,
                   help="MCTS sims for the first full pass")
    p.add_argument("--tail-sims", type=int, default=None,
                   help="MCTS sims for regenerated tails after a polish (default: --base-sims)")
    p.add_argument("--polish-sims", type=int, default=15000,
                   help="MCTS sims when re-solving each segment")
    p.add_argument("--polish-passes", type=int, default=3,
                   help="How many polish passes to run over the segment list")
    p.add_argument("--portfolio", choices=["single", "speed", "wide"], default="wide",
                   help="MCTS profile portfolio for polish attempts")
    p.add_argument("--window-size", type=int, default=2,
                   help="Maximum pickup segments to re-solve as one racing window")
    p.add_argument("--window-slack", type=float, default=0.4,
                   help="Extra local ticks allowed for a window (fraction of window length)")
    p.add_argument("--window-slack-ticks", type=int, default=60,
                   help="Absolute tick floor for window slack — short windows otherwise get unusably tight search budgets")
    p.add_argument("--polish-seeds", type=int, default=2,
                   help="RNG restarts per profile/window polish attempt")
    p.add_argument("--tail-seeds", type=int, default=None,
                   help="RNG restarts when re-solving the post-polish tail (default: --polish-seeds). "
                        "Multi-seed tail re-solve neutralizes MCTS noise — without it most polish wins are lost to tail variance.")
    p.add_argument("--polish-method", choices=["mcts", "cem"], default="mcts",
                   help="Inner solver for window polish. 'mcts' is the original; "
                        "'cem' uses cross-entropy on per-tick action distributions, "
                        "seeded from the existing window. CEM is deterministic given "
                        "seed and ignores --portfolio / --polish-sims.")
    p.add_argument("--cem-samples", type=int, default=128,
                   help="CEM rollouts per iteration")
    p.add_argument("--cem-iterations", type=int, default=4,
                   help="CEM refit iterations per polish attempt")
    p.add_argument("--cem-elite-frac", type=float, default=0.15,
                   help="Top fraction of samples kept for CEM refit")
    p.add_argument("--cem-action-repeat", type=int, default=None,
                   help="Macro-action granularity for CEM (default: --action-repeat). "
                        "CEM samples one decision per K ticks and holds it during "
                        "simulation. K=1 is per-tick (noisy). K=5–10 matches the "
                        "structure of optimal bang-bang trajectories and dramatically "
                        "improves quality.")
    p.add_argument("--first-accept", action="store_true",
                   help="Stop polishing a segment after the first accepted candidate. "
                        "Trades polish quality per segment for ~5–10× wall-time speedup. "
                        "Re-prioritization between accepts means the next bottleneck still "
                        "gets attention. Auto-defaults --tail-seeds to 1.")
    p.add_argument("--seed-base", type=int, default=None,
                   help="Optional deterministic seed base for polish attempts")
    p.add_argument("--source-restarts", type=int, default=1,
                   help="Run the first MCTS source pass N times with different seeds and "
                        "keep the shortest completed run. MCTS is high-variance; one bad "
                        "draw can give a 35s seed when 25s is achievable. 3–5 restarts "
                        "at moderate --base-sims is far more reliable than one restart "
                        "at high --base-sims. Ignored when seeding from --input-json or "
                        "a ghost.")
    p.add_argument("--exploration", type=float, default=1.41)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--shaping-weight", type=float, default=0.5)
    p.add_argument("--thrust-bias", type=float, default=0.0,
                   help="Base UCT bonus for thrust-on actions")
    p.add_argument("--thrust-bias-safe-dist", type=float, default=0.0,
                   help="Distance where thrust bias reaches full strength; 0 disables scaling")
    p.add_argument("--ar-depth-bonus", type=int, default=0,
                   help="Extra action repeat per MCTS tree depth")
    p.add_argument("--ar-max", type=int, default=20,
                   help="Maximum depth-scaled action repeat")
    p.add_argument("--widen-k", type=float, default=0.0,
                   help="Progressive widening coefficient; 0 disables it")
    p.add_argument("--prioritize", dest="prioritize", action="store_true", default=True,
                   help="Visit segments in waste-descending order (default: on). "
                        "Waste = actual_ticks − pathfinder_distance/cruise_speed.")
    p.add_argument("--no-prioritize", dest="prioritize", action="store_false",
                   help="Disable priority ordering; walk segments in their natural order.")
    p.add_argument("--cruise-speed", type=float, default=None,
                   help="Override cruise speed (px/tick) used for the lower bound. "
                        "Default: peak |velocity| observed during the source pass.")
    p.add_argument("--label", default="ai",
                   help="ghost_type label under which to save the polished ghost")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing to the dashboard DB")
    p.add_argument("--use-momentum", action="store_true")
    p.add_argument("--input-json", type=Path, default=None,
                   help="Optional action trace to polish. Accepts action indices or raw triplets.")
    p.add_argument("--from-human-ghost", action="store_true",
                   help="Use the human ghost (from dashboard DB) as the source trace. "
                        "Reconstructs an approximate action trace from the recorded "
                        "frames; if it does not complete the level, MCTS extends from "
                        "the last reconstructed state. Mutually exclusive with --input-json.")
    p.add_argument("--from-ai-ghost", action="store_true",
                   help="Use the existing AI ghost (from dashboard DB) as the source trace. "
                        "Skips the variance-prone first MCTS pass when a previous polish "
                        "already produced a good run. Reconstructed actions extend with "
                        "MCTS if needed. Mutually exclusive with --input-json / "
                        "--from-human-ghost.")
    p.add_argument("--from-ghost", default=None,
                   help="Use a ghost of the given type (e.g. 'ai', 'human', or any custom "
                        "label written via --label) as the source trace. Mutually exclusive "
                        "with --input-json / --from-human-ghost / --from-ai-ghost.")
    p.add_argument("--dump-json", type=Path, default=None,
                   help="Write the final raw action trace as JSON.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-pass candidate rejection summaries.")
    args = p.parse_args()

    for attr in ("max_steps", "base_sims", "polish_sims", "action_repeat"):
        if getattr(args, attr) <= 0:
            p.error(f"--{attr.replace('_', '-')} must be positive")
    if args.tail_sims is not None and args.tail_sims <= 0:
        p.error("--tail-sims must be positive")
    if args.polish_passes < 0:
        p.error("--polish-passes must be non-negative")
    if args.window_size <= 0:
        p.error("--window-size must be positive")
    if args.window_slack < 0.0:
        p.error("--window-slack must be non-negative")
    if args.polish_seeds <= 0:
        p.error("--polish-seeds must be positive")
    if args.tail_seeds is not None and args.tail_seeds <= 0:
        p.error("--tail-seeds must be positive")
    if args.window_slack_ticks < 0:
        p.error("--window-slack-ticks must be non-negative")
    if args.ar_depth_bonus < 0:
        p.error("--ar-depth-bonus must be non-negative")
    if args.ar_max <= 0:
        p.error("--ar-max must be positive")
    if args.widen_k < 0.0:
        p.error("--widen-k must be non-negative")
    if args.cem_samples <= 0:
        p.error("--cem-samples must be positive")
    if args.cem_iterations <= 0:
        p.error("--cem-iterations must be positive")
    if not 0.0 < args.cem_elite_frac <= 1.0:
        p.error("--cem-elite-frac must be in (0, 1]")
    if args.cem_action_repeat is not None and args.cem_action_repeat <= 0:
        p.error("--cem-action-repeat must be positive")
    if args.source_restarts <= 0:
        p.error("--source-restarts must be positive")

    ghost_source_flags = sum([
        args.from_human_ghost,
        args.from_ai_ghost,
        args.from_ghost is not None,
    ])
    if ghost_source_flags > 1:
        p.error("--from-human-ghost / --from-ai-ghost / --from-ghost are mutually exclusive")
    if ghost_source_flags > 0 and args.input_json is not None:
        p.error("--input-json is mutually exclusive with the --from-*-ghost flags")

    input_actions: Optional[list[int]] = None
    input_level: Optional[int] = None
    input_source: str = "mcts"
    if args.input_json is not None:
        try:
            input_level, input_actions = load_action_file(args.input_json)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            p.error(f"could not load --input-json: {exc}")
        input_source = str(args.input_json)
    elif args.from_human_ghost or args.from_ai_ghost or args.from_ghost is not None:
        if args.from_human_ghost:
            ghost_type = "human"
        elif args.from_ai_ghost:
            ghost_type = "ai"
        else:
            ghost_type = args.from_ghost
        if args.level is None:
            p.error(f"--from-{ghost_type}-ghost requires --level")
        sidecar_actions = load_ghost_actions(args.level, ghost_type)
        if sidecar_actions is not None:
            input_actions = sidecar_actions
            input_level = args.level
            input_source = (f"{ghost_type}-ghost-sidecar(L{args.level}, "
                            f"{len(input_actions)} actions)")
            print(f"[tas] loaded {len(input_actions)} exact actions "
                  f"({len(input_actions)/_TICK_HZ:.2f}s) from {ghost_type} "
                  f"ghost sidecar")
        else:
            try:
                ghost_frames = load_ghost_frames(args.level, ghost_type)
            except (OSError, ValueError) as exc:
                p.error(f"could not load {ghost_type} ghost: {exc}")
            input_actions = reconstruct_actions_from_ghost(ghost_frames)
            if not input_actions:
                p.error(f"{ghost_type} ghost reconstruction produced no actions")
            input_level = args.level
            input_source = f"{ghost_type}-ghost(L{args.level}, {len(ghost_frames)} frames)"
            print(f"[tas] no sidecar found at {_ghost_actions_path(args.level, ghost_type)}; "
                  f"reconstructing approximately from frames (lossy — first "
                  f"polish run will write a sidecar for exact future seeding)")
            print(f"[tas] reconstructed {len(input_actions)} actions "
                  f"({len(input_actions)/_TICK_HZ:.2f}s) from {len(ghost_frames)} "
                  f"{ghost_type} ghost frames")

    level = args.level if args.level is not None else input_level
    if level is None:
        p.error("--level is required unless --input-json contains a level")
    if input_level is not None and args.level is not None and input_level != args.level:
        p.error(f"--level {args.level} does not match input JSON level {input_level}")

    # Tail revalidation defaults to a HALF-strength polish budget rather than
    # the (usually weaker) base_sims. Tail noise was the dominant source of
    # rejected improvements; matching budget to polish strength fixes that.
    tail_sims = args.tail_sims if args.tail_sims is not None else max(args.base_sims, args.polish_sims // 2)
    if args.tail_seeds is not None:
        tail_seeds = args.tail_seeds
    elif args.first_accept:
        # First-accept means we don't compare accepts against each other, so
        # multi-seed tail averaging buys nothing — drop to 1 to save wall time.
        tail_seeds = 1
    else:
        tail_seeds = args.polish_seeds
    base_profile = SearchProfile(
        name="baseline",
        shaping_weight=args.shaping_weight,
        thrust_bias=args.thrust_bias,
        thrust_bias_safe_dist=args.thrust_bias_safe_dist,
        ar_depth_bonus=args.ar_depth_bonus,
        ar_max=args.ar_max,
        widen_k=args.widen_k,
    )
    polish_profiles = build_polish_profiles(
        args.portfolio,
        base_profile,
        action_repeat=args.action_repeat,
    )

    source = input_source
    print(f"[tas] level={level} source={source} base_sims={args.base_sims} "
          f"tail_sims={tail_sims} polish_sims={args.polish_sims} "
          f"passes={args.polish_passes} portfolio={args.portfolio} "
          f"profiles={','.join(p.name for p in polish_profiles)} "
          f"window={args.window_size}+{args.window_slack:.0%} "
          f"method={args.polish_method}"
          + (f" cem={args.cem_samples}x{args.cem_iterations}"
             if args.polish_method == "cem" else ""))

    env = SpaceAceDirectEnv(level=level, max_steps=args.max_steps)
    mcts = spaceace_rl.PyMCTSEngine(level, args.max_steps, args.use_momentum)

    # ── Pass 1: full MCTS run, split on pickups ───────────────────────────
    t0 = time.time()
    if input_actions is None:
        best_segments: Optional[list[Segment]] = None
        best_completed = False
        best_ticks = args.max_steps + 1
        for restart in range(args.source_restarts):
            # Vary the engine RNG so each restart explores different
            # branches. Seed 0 leaves the engine on its default state — the
            # behavior anyone running --source-restarts=1 used to get.
            if args.source_restarts > 1:
                seed = (args.seed_base or 0xA5A5_A5A5) + restart * 1_000_003
                _seed_mcts(seed)
            r_segments, r_completed = first_pass(
                env, mcts,
                base_sims=args.base_sims,
                exploration=args.exploration,
                gamma=args.gamma,
                action_repeat=args.action_repeat,
                max_ticks=args.max_steps,
                profile=base_profile,
            )
            r_ticks = sum(s.ticks for s in r_segments)
            if args.source_restarts > 1:
                tag = "✓" if r_completed else "✗"
                print(f"[tas] source restart {restart+1}/{args.source_restarts}: "
                      f"{r_ticks} ticks ({r_ticks/60.0:.2f}s) completed={r_completed} {tag}")
            # Prefer completed runs over incomplete ones; among completed
            # runs, prefer shorter.
            replace_best = False
            if r_completed and not best_completed:
                replace_best = True
            elif r_completed and best_completed and r_ticks < best_ticks:
                replace_best = True
            elif not best_completed and r_ticks < best_ticks:
                replace_best = True
            if replace_best:
                best_segments = r_segments
                best_completed = r_completed
                best_ticks = r_ticks
        segments = best_segments or []
        completed = best_completed
    else:
        segments, completed = segment_existing_actions(
            env,
            input_actions,
            max_ticks=args.max_steps,
        )
        # Drop a no-pickup trailing segment (reconstruction overshot the last
        # pickup without grabbing anything new) so the MCTS extension picks up
        # cleanly from a real pickup boundary.
        if (
            not completed
            and segments
            and segments[-1].pickups_after == segments[-1].pickups_before
        ):
            tail_ticks = segments[-1].ticks
            segments.pop()
            print(f"[tas] dropped {tail_ticks}-tick no-pickup tail before extending")
            # Re-replay so env state matches the dropped boundary.
            env.reset()
            for seg in segments:
                for a_idx in seg.action_indices:
                    env.step(ALL_ACTIONS[a_idx])

        recon_ticks = sum(s.ticks for s in segments)
        recon_segments_count = len(segments)
        if not completed and env.get_pickups_remaining() > 0:
            ext_budget = max(0, args.max_steps - sum(s.ticks for s in segments))
            if ext_budget > 0:
                print(f"[tas] reconstruction kept {recon_segments_count} segments "
                      f"({recon_ticks} ticks, {recon_ticks / 60.0:.2f}s); extending "
                      f"with MCTS for up to {ext_budget} ticks "
                      f"(remaining pickups={env.get_pickups_remaining()})")
                ext_segments, completed = _segment_from_here(
                    env, mcts,
                    num_simulations=args.base_sims,
                    exploration=args.exploration,
                    gamma=args.gamma,
                    action_repeat=args.action_repeat,
                    budget_ticks=ext_budget,
                    profile=base_profile,
                )
                segments.extend(ext_segments)
                ext_ticks = sum(s.ticks for s in ext_segments)
                print(f"[tas] MCTS extension added {len(ext_segments)} segments "
                      f"({ext_ticks} ticks, {ext_ticks / 60.0:.2f}s), "
                      f"completed={completed}")
    base_ticks = sum(s.ticks for s in segments)
    base_elapsed = time.time() - t0
    print(f"[tas] seed for polishing: {len(segments)} segments, {base_ticks} ticks "
          f"({base_ticks / 60.0:.2f}s game-time), completed={completed}, "
          f"wall={base_elapsed:.1f}s")

    if not completed:
        print("[tas] source run did not complete the level — aborting polish")
        return 1

    # ── Cruise speed estimate (for waste-based prioritization) ────────────
    if args.prioritize:
        if args.cruise_speed is not None:
            cruise_px_per_tick = float(args.cruise_speed)
        else:
            cruise_px_per_tick = _estimate_cruise_speed(env, segments)
        print(f"[tas] cruise speed estimate: {cruise_px_per_tick:.2f} px/tick "
              f"(used to rank segments by recoverable slack)")

    # ── Polish passes ─────────────────────────────────────────────────────
    # For each segment, try re-solving with more sims; if better, regenerate
    # downstream segments so the stitched run is physically consistent.
    # When --prioritize, segments are visited in waste-descending order so
    # the polish budget concentrates on the segments with the most slack.
    for pass_i in range(args.polish_passes):
        improved = 0
        start_total = sum(s.ticks for s in segments)
        reject_counts: Counter[str] = Counter()

        # Build initial visit order. Walking by segment-instance id (not index)
        # because indices shift after an accept while pre-polish prefix
        # instances stay stable. Downstream tail segments are NEW instances
        # after a polish; we re-queue them with refreshed waste scores.
        attempted_ids: set[int] = set()

        def _enqueue_all() -> list[tuple[float, int, int]]:
            if args.prioritize:
                scored = [
                    (-_segment_waste_score(mcts, segments[i], cruise_px_per_tick), id(segments[i]), i)
                    for i in range(len(segments))
                    if id(segments[i]) not in attempted_ids
                ]
                scored.sort()  # most-wasteful first (negative waste)
                return scored
            return [(0.0, id(segments[i]), i) for i in range(len(segments))
                    if id(segments[i]) not in attempted_ids]

        bar = tqdm(
            total=len(segments),
            desc=f"pass {pass_i+1}/{args.polish_passes}",
            unit="seg",
            dynamic_ncols=True,
            leave=True,
        )
        try:
            queue = _enqueue_all()
            while queue:
                neg_waste, seg_id, idx = queue.pop(0)
                # Stale-entry guard: index could have been displaced by an
                # earlier accept that changed the segment count.
                if idx >= len(segments) or id(segments[idx]) != seg_id:
                    continue
                if seg_id in attempted_ids:
                    continue
                attempted_ids.add(seg_id)

                seg_idx_for_bar = idx
                waste_str = f"w={-neg_waste:.0f}t" if args.prioritize else ""
                bar.set_postfix_str(f"seg {seg_idx_for_bar} starting {waste_str}", refresh=True)

                def _cb(msg: str, _i: int = seg_idx_for_bar) -> None:
                    bar.set_postfix_str(f"seg {_i}: {msg}", refresh=True)

                old_total = sum(s.ticks for s in segments)
                candidate_segments = _try_polish_at(
                    env, mcts, segments, idx,
                    polish_sims=args.polish_sims,
                    base_sims=tail_sims,
                    exploration=args.exploration,
                    gamma=args.gamma,
                    action_repeat=args.action_repeat,
                    tail_profile=base_profile,
                    profiles=polish_profiles,
                    max_window_size=args.window_size,
                    window_slack=args.window_slack,
                    window_slack_ticks=args.window_slack_ticks,
                    polish_seeds=args.polish_seeds,
                    tail_seeds=tail_seeds,
                    seed_base=args.seed_base,
                    budget_ticks=args.max_steps,
                    first_accept=args.first_accept,
                    polish_method=args.polish_method,
                    cem_samples=args.cem_samples,
                    cem_iterations=args.cem_iterations,
                    cem_elite_frac=args.cem_elite_frac,
                    cem_action_repeat=(args.cem_action_repeat
                                       if args.cem_action_repeat is not None
                                       else args.action_repeat),
                    reject_counts=reject_counts,
                    progress_cb=_cb,
                )
                if candidate_segments is not None:
                    new_total = sum(s.ticks for s in candidate_segments)
                    delta = old_total - new_total
                    old_pickups = segments[idx].pickups_before - segments[idx].pickups_after
                    new_pickups = segments[idx].pickups_before - candidate_segments[idx].pickups_after
                    bar.write(
                        f"[tas] pass {pass_i+1} seg {idx}: "
                        f"replacement {segments[idx].ticks} → {candidate_segments[idx].ticks} ticks "
                        f"for {old_pickups} → {new_pickups} pickups, "
                        f"total {old_total} → {new_total} (-{delta})"
                    )
                    segments = candidate_segments
                    if bar.total != len(segments):
                        bar.total = len(segments)
                        bar.refresh()
                    improved += 1
                    # Tail segments (idx onward) are new instances. Re-rank
                    # everything not yet tried so the next pick reflects the
                    # post-polish state — this is the main reason prioritized
                    # walking helps: an accepted polish can expose a different
                    # downstream segment as the new bottleneck.
                    queue = _enqueue_all()
                bar.update(1)
        finally:
            bar.close()
        total = sum(s.ticks for s in segments)
        print(f"[tas] pass {pass_i+1} done: improved {improved} segments, "
              f"total {start_total} → {total} "
              f"(-{start_total - total}, {total / 60.0:.2f}s)")
        if args.verbose and reject_counts:
            summary = ", ".join(
                f"{reason}={count}"
                for reason, count in reject_counts.most_common()
            )
            print(f"[tas] pass {pass_i+1} attempts: {summary}")
        if improved == 0:
            print("[tas] no more improvements — stopping early")
            break

    # ── Stitch + capture frames ───────────────────────────────────────────
    frames, final_ticks, completed = stitch_and_capture(env, segments)
    if not completed:
        print(f"[tas] ERROR: stitched run did not complete "
              f"(ticks={final_ticks}, remaining={env.get_pickups_remaining()})")
        return 2

    time_seconds = final_ticks / 60.0
    ghost_frames = build_ghost_frames(frames)
    print(f"[tas] final: {final_ticks} ticks, {time_seconds:.2f}s game-time "
          f"({len(ghost_frames)} ghost frames)")

    if args.dump_json:
        final_actions = flatten_segments(segments)
        dump_action_file(args.dump_json, level, final_actions, final_ticks)
        print(f"[tas] dumped action trace -> {args.dump_json}")

    # Report MCTS tree-reuse effectiveness across the whole run.
    try:
        hits, misses = mcts.get_reuse_stats()
        total = hits + misses
        if total:
            print(f"[tas] mcts tree reuse: {hits}/{total} hits "
                  f"({100.0 * hits / total:.1f}%) — cache is reset per segment, "
                  f"so reuse is intra-segment only")
    except Exception:
        pass

    if args.no_save:
        print("[tas] --no-save set; skipping DB write")
        return 0

    save_ghost(
        level, args.label, time_seconds, ghost_frames,
        action_indices=flatten_segments(segments),
        final_ticks=final_ticks,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
