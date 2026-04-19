"""Dump + compare scalar series from SB3 tensorboard event files.

Usage:
    uv run python scripts/compare_runs.py <run_dir_a> <run_dir_b> [<run_dir_c> ...]

Prints final / best / mean values for the metrics that matter:
rollout/ep_rew_mean, rollout/ep_len_mean, train/value_loss,
curriculum/smoothed_win_rate, episode/completed, episode/crashed, and FPS.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


KEYS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "curriculum/smoothed_win_rate",
    "episode/completed",
    "episode/crashed",
    "episode/pickups_collected",
    "time/fps",
]


def load(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    out = {}
    for k in KEYS:
        if k in tags:
            out[k] = [(e.step, e.value) for e in acc.Scalars(k)]
    return out


def summarize(series: list[tuple[int, float]]) -> dict[str, float]:
    vals = [v for _, v in series]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "first": vals[0],
        "last": vals[-1],
        "best": max(vals),
        "worst": min(vals),
        "mean": sum(vals) / len(vals),
        "last_step": series[-1][0],
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    dirs = [Path(p) for p in sys.argv[1:]]
    runs = {d.name: load(d) for d in dirs}

    # Print per-metric table
    all_keys = set()
    for r in runs.values():
        all_keys.update(r.keys())

    for key in KEYS:
        if key not in all_keys:
            continue
        print()
        print(f"=== {key} ===")
        header = f"{'run':<30s} {'n':>6s} {'first':>12s} {'last':>12s} {'best':>12s} {'mean':>12s} {'last_step':>12s}"
        print(header)
        print("-" * len(header))
        for name, r in runs.items():
            s = summarize(r.get(key, []))
            if not s:
                print(f"{name:<30s} (no data)")
                continue
            print(f"{name:<30s} {s['n']:>6d} {s['first']:>12.3f} {s['last']:>12.3f} "
                  f"{s['best']:>12.3f} {s['mean']:>12.3f} {int(s['last_step']):>12d}")


if __name__ == "__main__":
    main()
