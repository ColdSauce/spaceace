"""Exact endpoint reachable-set polish for near-complete TAS traces.

This handles the common failure mode where a search gets into the final pickup
disk's neighborhood but stops just before the simulator flips the completion
bit. It does not score candidates by proximity. It enumerates short exact
continuations, replays each in the Rust engine, and accepts only validated level
completion.

Example:
    uv run python scripts/tas_endpoint_reachable_polish.py --level 7 \
      --input-json /tmp/spaceace_l7_phase15_final_best_effort_prefix.json \
      --max-append 8 --dump-json /tmp/spaceace_l7_endpoint_complete.json
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spaceace_rl  # noqa: E402,F401  # Imported so the PyO3 module is available.
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.ghost_actions import dump_action_file, load_action_file  # noqa: E402
from spaceace.strategies.actions import ACTION_NAMES, ALL_ACTIONS  # noqa: E402


@dataclass(frozen=True)
class ReplayResult:
    completed: bool
    crashed: bool
    ticks: int
    obs: tuple[float, float, float, float, float]
    pickup_states: tuple[bool, ...]


def _obs5(obs) -> tuple[float, float, float, float, float]:
    return (
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
    )


class ReplayHarness:
    def __init__(self, level: int, max_steps: int) -> None:
        self.env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self.max_steps = int(max_steps)

    def replay(self, actions: list[int]) -> ReplayResult:
        env = self.env
        env.reset()
        obs = env.get_observation()
        info = {}
        for tick, action_idx in enumerate(actions[: self.max_steps], start=1):
            obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
            if info.get("level_completed"):
                return ReplayResult(True, False, tick, _obs5(obs), tuple(env.get_pickup_states()))
            if info.get("ship_exploded") or terminated or truncated:
                return ReplayResult(
                    False,
                    bool(info.get("ship_exploded")),
                    tick,
                    _obs5(obs),
                    tuple(env.get_pickup_states()),
                )
        return ReplayResult(
            False,
            False,
            min(len(actions), self.max_steps),
            _obs5(obs),
            tuple(env.get_pickup_states()),
        )


def _action_indices(raw: Optional[str]) -> list[int]:
    if raw is None:
        return list(range(len(ALL_ACTIONS)))
    names = {name: idx for idx, name in enumerate(ACTION_NAMES)}
    indices: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
        else:
            idx = names[token]
        if not 0 <= idx < len(ALL_ACTIONS):
            raise ValueError(f"action {idx} outside 0..{len(ALL_ACTIONS) - 1}")
        indices.append(idx)
    return sorted(set(indices))


def find_short_completion(
    *,
    level: int,
    actions: list[int],
    max_append: int,
    action_indices: list[int],
    max_steps: int,
) -> tuple[Optional[list[int]], ReplayResult, int]:
    harness = ReplayHarness(level=level, max_steps=max_steps)
    base_result = harness.replay(actions)
    if base_result.completed:
        return list(actions[: base_result.ticks]), base_result, 0

    checked = 0
    t0 = time.time()
    for depth in range(1, max_append + 1):
        for suffix in itertools.product(action_indices, repeat=depth):
            checked += 1
            candidate = actions + list(suffix)
            result = harness.replay(candidate)
            if result.completed:
                names = " ".join(ACTION_NAMES[idx] for idx in suffix)
                print(
                    f"found completion: +{depth}t total={result.ticks} "
                    f"suffix=[{names}] checked={checked} elapsed={time.time() - t0:.2f}s"
                )
                return candidate[: result.ticks], result, checked
        print(f"depth={depth} checked={checked} elapsed={time.time() - t0:.2f}s")
    return None, base_result, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--max-append", type=int, default=8)
    parser.add_argument("--actions", default=None, help="Comma-separated action names or indices.")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--dump-json", type=Path, default=None)
    args = parser.parse_args()

    trace_level, actions = load_action_file(args.input_json)
    if trace_level is not None and trace_level != args.level:
        raise ValueError(f"{args.input_json} declares level {trace_level}, expected {args.level}")

    action_indices = _action_indices(args.actions)
    completed_actions, result, checked = find_short_completion(
        level=args.level,
        actions=actions,
        max_append=args.max_append,
        action_indices=action_indices,
        max_steps=args.max_steps,
    )

    speed = math.hypot(result.obs[2], result.obs[3])
    print(
        f"base/result: completed={result.completed} crashed={result.crashed} "
        f"ticks={result.ticks} checked={checked} "
        f"obs={[round(v, 3) for v in result.obs]} speed={speed:.1f} "
        f"pickups={list(result.pickup_states)}"
    )
    if completed_actions is None:
        print("no endpoint completion found")
        return 1

    if args.dump_json is not None:
        dump_action_file(args.dump_json, args.level, completed_actions, len(completed_actions))
        print(f"dumped completion -> {args.dump_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
