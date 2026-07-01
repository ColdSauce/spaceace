"""Exact-model local search for TAS action traces.

This is intentionally not a reward-shaping optimizer. It treats the Rust game
engine as the objective function:

    completed trace A is better than completed trace B iff A finishes in fewer
    physics ticks.

The search works on the run-length encoding of an incumbent exact action trace.
It tries deterministic duration edits and short action substitutions, replays
each candidate through the simulator, and accepts only validated faster
completions. This makes it a practical follow-up to MCTS/hand-seeded traces
without inventing another heuristic value function.

Examples:
    uv run python scripts/tas_local_search.py --level 7 --input-json /tmp/spaceace_l7_mcts_1367.json --passes 4 --dump-json /tmp/spaceace_l7_local.json
    uv run python scripts/tas_local_search.py --level 7 --input-json /tmp/spaceace_l7_local.json --segment 0 --target-ticks 1311
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.ghost_actions import dump_action_file, load_action_file  # noqa: E402
from spaceace.strategies.actions import ACTION_NAMES, ALL_ACTIONS  # noqa: E402


@dataclass(frozen=True)
class Run:
    action_idx: int
    ticks: int


@dataclass(frozen=True)
class PickupEvent:
    pickup_idx: int
    tick: int


@dataclass(frozen=True)
class ReplayResult:
    completed: bool
    crashed: bool
    ticks: int
    events: tuple[PickupEvent, ...]
    final_speed: float


@dataclass(frozen=True)
class Candidate:
    label: str
    actions: list[int]


class ReplayHarness:
    """Reusable exact simulator harness for candidate validation."""

    def __init__(self, level: int, max_steps: int) -> None:
        self.level = int(level)
        self.max_steps = int(max_steps)
        self.env = SpaceAceDirectEnv(level=level, max_steps=max_steps)

    def replay(self, actions: list[int]) -> ReplayResult:
        """Replay a candidate in the exact simulator and report the true outcome."""
        env = self.env
        env.reset()
        prev_pickups = list(env.get_pickup_states())
        events: list[PickupEvent] = []
        final_speed = 0.0

        for tick, action_idx in enumerate(actions[: self.max_steps], start=1):
            obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
            final_speed = math.hypot(float(obs[2]), float(obs[3]))
            pickup_states = list(env.get_pickup_states())
            for idx, (before, after) in enumerate(zip(prev_pickups, pickup_states)):
                if not before and after:
                    events.append(PickupEvent(idx, tick))
            prev_pickups = pickup_states

            if info.get("level_completed"):
                return ReplayResult(
                    completed=True,
                    crashed=False,
                    ticks=tick,
                    events=tuple(events),
                    final_speed=final_speed,
                )
            if info.get("ship_exploded") or terminated or truncated:
                return ReplayResult(
                    completed=False,
                    crashed=bool(info.get("ship_exploded")),
                    ticks=tick,
                    events=tuple(events),
                    final_speed=final_speed,
                )

        return ReplayResult(
            completed=False,
            crashed=False,
            ticks=min(len(actions), self.max_steps),
            events=tuple(events),
            final_speed=final_speed,
        )


def rle(actions: list[int]) -> list[Run]:
    """Return run-length encoding for a per-tick action trace."""
    if not actions:
        return []
    runs: list[Run] = []
    last = actions[0]
    count = 1
    for action_idx in actions[1:]:
        if action_idx == last:
            count += 1
            continue
        runs.append(Run(last, count))
        last = action_idx
        count = 1
    runs.append(Run(last, count))
    return runs


def flatten_runs(runs: Iterable[Run]) -> list[int]:
    out: list[int] = []
    for run in runs:
        if run.ticks > 0:
            out.extend([run.action_idx] * run.ticks)
    return out


def merge_runs(runs: Iterable[Run]) -> list[Run]:
    merged: list[Run] = []
    for run in runs:
        if run.ticks <= 0:
            continue
        if merged and merged[-1].action_idx == run.action_idx:
            prev = merged[-1]
            merged[-1] = Run(prev.action_idx, prev.ticks + run.ticks)
        else:
            merged.append(run)
    return merged


def replay(level: int, actions: list[int], *, max_steps: int) -> ReplayResult:
    return ReplayHarness(level=level, max_steps=max_steps).replay(actions)


def _run_starts(runs: list[Run]) -> list[int]:
    starts: list[int] = []
    total = 0
    for run in runs:
        starts.append(total)
        total += run.ticks
    return starts


def _segment_bounds(events: tuple[PickupEvent, ...], total_ticks: int) -> list[tuple[int, int]]:
    """Return zero-based half-open tick ranges for pickup-bounded segments."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for event in events:
        end = event.tick
        bounds.append((start, end))
        start = end
    if start < total_ticks:
        bounds.append((start, total_ticks))
    return bounds


def _run_overlaps_segment(start: int, end: int, segment: Optional[tuple[int, int]]) -> bool:
    if segment is None:
        return True
    seg_start, seg_end = segment
    return start < seg_end and end > seg_start


def deletion_candidates(
    actions: list[int],
    runs: list[Run],
    *,
    amounts: list[int],
    segment: Optional[tuple[int, int]],
) -> Iterable[Candidate]:
    """Remove ticks from individual action runs."""
    starts = _run_starts(runs)
    for i, run in enumerate(runs):
        run_start = starts[i]
        run_end = run_start + run.ticks
        if not _run_overlaps_segment(run_start, run_end, segment):
            continue
        for amount in amounts:
            if amount <= 0 or run.ticks <= amount:
                continue
            new_runs = list(runs)
            new_runs[i] = Run(run.action_idx, run.ticks - amount)
            yield Candidate(
                label=f"delete {amount}t from run {i} {ACTION_NAMES[run.action_idx]}",
                actions=flatten_runs(merge_runs(new_runs)),
            )


def boundary_shift_candidates(
    runs: list[Run],
    *,
    amounts: list[int],
    segment: Optional[tuple[int, int]],
) -> Iterable[Candidate]:
    """Move ticks across adjacent action-run boundaries, preserving length."""
    starts = _run_starts(runs)
    for i in range(len(runs) - 1):
        left = runs[i]
        right = runs[i + 1]
        boundary_start = starts[i]
        boundary_end = starts[i + 1] + right.ticks
        if not _run_overlaps_segment(boundary_start, boundary_end, segment):
            continue
        for amount in amounts:
            if amount <= 0:
                continue
            if left.ticks > amount:
                new_runs = list(runs)
                new_runs[i] = Run(left.action_idx, left.ticks - amount)
                new_runs[i + 1] = Run(right.action_idx, right.ticks + amount)
                yield Candidate(
                    label=(
                        f"shift {amount}t left->right at boundary {i} "
                        f"{ACTION_NAMES[left.action_idx]}->{ACTION_NAMES[right.action_idx]}"
                    ),
                    actions=flatten_runs(merge_runs(new_runs)),
                )
            if right.ticks > amount:
                new_runs = list(runs)
                new_runs[i] = Run(left.action_idx, left.ticks + amount)
                new_runs[i + 1] = Run(right.action_idx, right.ticks - amount)
                yield Candidate(
                    label=(
                        f"shift {amount}t right->left at boundary {i} "
                        f"{ACTION_NAMES[left.action_idx]}->{ACTION_NAMES[right.action_idx]}"
                    ),
                    actions=flatten_runs(merge_runs(new_runs)),
                )


def substitution_candidates(
    actions: list[int],
    runs: list[Run],
    *,
    amounts: list[int],
    segment: Optional[tuple[int, int]],
) -> Iterable[Candidate]:
    """Replace a short prefix/suffix of a run with another canonical action."""
    starts = _run_starts(runs)
    for i, run in enumerate(runs):
        run_start = starts[i]
        run_end = run_start + run.ticks
        if not _run_overlaps_segment(run_start, run_end, segment):
            continue
        for amount in amounts:
            if amount <= 0 or run.ticks < amount:
                continue
            for replacement in range(len(ALL_ACTIONS)):
                if replacement == run.action_idx:
                    continue
                # Prefix replacement.
                prefix_actions = list(actions)
                prefix_actions[run_start : run_start + amount] = [replacement] * amount
                yield Candidate(
                    label=(
                        f"replace first {amount}t of run {i} "
                        f"{ACTION_NAMES[run.action_idx]}->{ACTION_NAMES[replacement]}"
                    ),
                    actions=prefix_actions,
                )
                # Suffix replacement. Avoid duplicating the prefix candidate when
                # the whole run is replaced.
                suffix_start = run_end - amount
                if suffix_start != run_start:
                    suffix_actions = list(actions)
                    suffix_actions[suffix_start:run_end] = [replacement] * amount
                    yield Candidate(
                        label=(
                            f"replace last {amount}t of run {i} "
                            f"{ACTION_NAMES[run.action_idx]}->{ACTION_NAMES[replacement]}"
                        ),
                        actions=suffix_actions,
                    )


def _parse_amounts(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value > 0:
            values.append(value)
    # Large edits first often expose the useful deletions sooner.
    return sorted(set(values), reverse=True)


def _print_trace(label: str, result: ReplayResult, action_count: int) -> None:
    order = [event.pickup_idx for event in result.events]
    event_text = ", ".join(f"P{event.pickup_idx}@{event.tick}" for event in result.events)
    print(
        f"{label}: completed={result.completed} crashed={result.crashed} "
        f"ticks={result.ticks} actions={action_count} "
        f"order={order} events=[{event_text}]"
    )


def search(
    *,
    level: int,
    initial_actions: list[int],
    max_steps: int,
    passes: int,
    delete_amounts: list[int],
    shift_amounts: list[int],
    substitute_amounts: list[int],
    segment_idx: Optional[int],
    first_improvement: bool,
    target_ticks: Optional[int],
    checkpoint_path: Optional[Path],
) -> tuple[list[int], ReplayResult]:
    """Run deterministic local search and return the best validated trace."""
    harness = ReplayHarness(level=level, max_steps=max_steps)
    best_actions = list(initial_actions)
    best_result = harness.replay(best_actions)
    _print_trace("initial", best_result, len(best_actions))
    if not best_result.completed:
        raise ValueError("initial trace does not complete; local search needs a valid incumbent")

    attempts = 0
    accepted = 0
    t0 = time.time()

    for pass_idx in range(1, passes + 1):
        runs = rle(best_actions[: best_result.ticks])
        segment: Optional[tuple[int, int]] = None
        if segment_idx is not None:
            bounds = _segment_bounds(best_result.events, best_result.ticks)
            if not 0 <= segment_idx < len(bounds):
                raise ValueError(f"segment {segment_idx} outside 0..{len(bounds) - 1}")
            segment = bounds[segment_idx]

        print(
            f"\npass {pass_idx}: incumbent={best_result.ticks}t "
            f"runs={len(runs)} segment={segment if segment is not None else 'full'}"
        )

        candidate_streams = (
            deletion_candidates(
                best_actions[: best_result.ticks],
                runs,
                amounts=delete_amounts,
                segment=segment,
            ),
            boundary_shift_candidates(
                runs,
                amounts=shift_amounts,
                segment=segment,
            ),
            substitution_candidates(
                best_actions[: best_result.ticks],
                runs,
                amounts=substitute_amounts,
                segment=segment,
            ),
        )

        pass_best_actions: Optional[list[int]] = None
        pass_best_result: Optional[ReplayResult] = None
        pass_best_label = ""

        for stream in candidate_streams:
            for candidate in stream:
                attempts += 1
                result = harness.replay(candidate.actions)
                if not result.completed:
                    continue
                if result.ticks >= best_result.ticks:
                    continue
                if pass_best_result is None or result.ticks < pass_best_result.ticks:
                    pass_best_actions = candidate.actions[: result.ticks]
                    pass_best_result = result
                    pass_best_label = candidate.label
                    print(
                        f"  found {result.ticks}t ({best_result.ticks - result.ticks:+d}t): "
                        f"{candidate.label}"
                    )
                    if first_improvement:
                        break
            if first_improvement and pass_best_result is not None:
                break

        if pass_best_result is None or pass_best_actions is None:
            print(f"  no accepted edit in pass {pass_idx}")
            break

        best_actions = pass_best_actions
        best_result = pass_best_result
        accepted += 1
        print(
            f"  accepted pass {pass_idx}: {best_result.ticks}t "
            f"via {pass_best_label}"
        )

        if checkpoint_path is not None:
            dump_action_file(checkpoint_path, level, best_actions, best_result.ticks)
            print(f"  checkpoint -> {checkpoint_path}")

        if target_ticks is not None and best_result.ticks <= target_ticks:
            print(f"  target reached: {best_result.ticks} <= {target_ticks}")
            break

    print(
        f"\nsearch done: attempts={attempts} accepted={accepted} "
        f"best={best_result.ticks}t ({best_result.ticks / 60.0:.2f}s) "
        f"elapsed={time.time() - t0:.1f}s"
    )
    _print_trace("best", best_result, len(best_actions))
    return best_actions[: best_result.ticks], best_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--delete-amounts", default="32,16,8,4,2,1")
    parser.add_argument("--shift-amounts", default="24,12,6,3,1")
    parser.add_argument("--substitute-amounts", default="12,6,3,1")
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Only mutate one pickup-bounded segment by rank: 0=start->first pickup.",
    )
    parser.add_argument("--first-improvement", action="store_true")
    parser.add_argument("--target-ticks", type=int, default=None)
    parser.add_argument("--dump-json", type=Path, default=None)
    parser.add_argument("--checkpoint-json", type=Path, default=None)
    args = parser.parse_args()

    trace_level, actions = load_action_file(args.input_json)
    if trace_level is not None and trace_level != args.level:
        raise ValueError(f"{args.input_json} declares level {trace_level}, expected {args.level}")

    best_actions, best_result = search(
        level=args.level,
        initial_actions=actions,
        max_steps=args.max_steps,
        passes=args.passes,
        delete_amounts=_parse_amounts(args.delete_amounts),
        shift_amounts=_parse_amounts(args.shift_amounts),
        substitute_amounts=_parse_amounts(args.substitute_amounts),
        segment_idx=args.segment,
        first_improvement=args.first_improvement,
        target_ticks=args.target_ticks,
        checkpoint_path=args.checkpoint_json,
    )

    if args.dump_json is not None:
        dump_action_file(args.dump_json, args.level, best_actions, best_result.ticks)
        print(f"dumped best trace -> {args.dump_json}")

    print(json.dumps({
        "level": args.level,
        "ticks": best_result.ticks,
        "seconds": best_result.ticks / 60.0,
        "completed": best_result.completed,
        "events": [
            {"pickup_idx": event.pickup_idx, "tick": event.tick}
            for event in best_result.events
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
