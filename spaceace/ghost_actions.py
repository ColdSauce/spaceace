"""Exact per-tick action trace sidecars for ghost runs."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Optional

import numpy as np

from spaceace.strategies.actions import ALL_ACTIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GHOST_ACTIONS_DIR = PROJECT_ROOT / "ghost_actions"

ACTION_TO_INDEX = {
    tuple(int(x) for x in action.tolist()): idx
    for idx, action in enumerate(ALL_ACTIONS)
}


def sidecar_path(level: int, ghost_type: str) -> Path:
    """Return the canonical exact-action sidecar path for a ghost label."""
    return GHOST_ACTIONS_DIR / f"L{int(level)}_{ghost_type}.json"


def action_to_index(action) -> int:
    """Convert a raw action triplet into the canonical 0..5 action index."""
    if isinstance(action, np.ndarray):
        raw = tuple(int(x) for x in action.astype(int).tolist())
    elif isinstance(action, (list, tuple)) and len(action) == 3:
        raw = tuple(int(x) for x in action)
    else:
        raise ValueError(f"expected a length-3 action triplet, got {action!r}")
    idx = ACTION_TO_INDEX.get(raw)
    if idx is None:
        raise ValueError(f"action {list(raw)} is not in the 6-action set")
    return idx


def decode_action_item(item, idx: int) -> int:
    """Accept either an action index or a raw [left, right, thrust] triplet."""
    if isinstance(item, Integral) and not isinstance(item, bool):
        action_idx = int(item)
        if 0 <= action_idx < len(ALL_ACTIONS):
            return action_idx
        raise ValueError(
            f"action {idx}: index {action_idx} is outside 0..{len(ALL_ACTIONS) - 1}"
        )

    if isinstance(item, (list, tuple)) and len(item) == 3:
        try:
            return action_to_index(item)
        except ValueError as exc:
            raise ValueError(f"action {idx}: {exc}") from exc

    raise ValueError(f"action {idx}: expected an index or [left, right, thrust] triplet")


def load_action_file(path: Path | str) -> tuple[Optional[int], list[int]]:
    """Load a TAS/ghost sidecar and normalize it to action indices."""
    action_path = Path(path)
    data = json.loads(action_path.read_text())
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

    return level, [decode_action_item(item, i) for i, item in enumerate(raw_actions)]


def load_sidecar_actions(level: int, ghost_type: str) -> Optional[list[int]]:
    """Load exact sidecar actions for (level, ghost_type), if present."""
    path = sidecar_path(level, ghost_type)
    if not path.exists():
        return None
    sidecar_level, actions = load_action_file(path)
    if sidecar_level is not None and sidecar_level != int(level):
        raise ValueError(
            f"{path} declares level {sidecar_level}, expected level {int(level)}"
        )
    return actions


def dump_action_file(path: Path | str, level: int, action_indices: list[int], ticks: int) -> None:
    """Write an exact action trace in the sidecar schema."""
    normalized = [decode_action_item(a, i) for i, a in enumerate(action_indices)]
    payload = {
        "level": int(level),
        "ticks": int(ticks),
        "seconds": round(int(ticks) / 60.0, 3),
        "action_format": "indices",
        "actions": normalized,
        "raw_actions": [ALL_ACTIONS[a].astype(int).tolist() for a in normalized],
    }
    action_path = Path(path)
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_text(json.dumps(payload, indent=2) + "\n")


def write_sidecar_if_best(
    level: int,
    ghost_type: str,
    action_indices: list[int],
    ticks: int,
) -> bool:
    """Write the sidecar if it beats the existing sidecar for this label.

    This intentionally compares against the sidecar, not the dashboard ghost row.
    Older DB rows may lack exact actions; a slower-but-valid sidecar is still
    useful as a deterministic seed until a faster exact trace replaces it.
    """
    path = sidecar_path(level, ghost_type)
    seconds = int(ticks) / 60.0
    existing_seconds: Optional[float] = None
    if path.exists():
        try:
            existing_seconds = float(json.loads(path.read_text()).get("seconds"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing_seconds = None

    if existing_seconds is not None and existing_seconds <= seconds:
        print(
            f"  [ghost] existing {ghost_type} action sidecar for level {level} is faster "
            f"({existing_seconds:.2f}s <= {seconds:.2f}s), not saving"
        )
        return False

    dump_action_file(path, level, action_indices, ticks)
    prev = f" (prev {existing_seconds:.2f}s)" if existing_seconds is not None else ""
    print(
        f"  [ghost] saved {ghost_type} action sidecar for level {level}: "
        f"{len(action_indices)} actions, {seconds:.2f}s{prev}"
    )
    return True
