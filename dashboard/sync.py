"""Scan tensorboard_logs/ and models/ directories, populate SQLite."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dashboard.db import get_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TB_DIR = PROJECT_ROOT / "tensorboard_logs"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------------

# hrl_pilot_lvl0_1_2_3_ar3_500k_1
_HRL_PILOT = re.compile(
    r"^hrl_pilot_lvl([\d_]+)_ar(\d+)_(\d+)k_(\d+)$"
)
# hrl_dqn_corridors_200k_1  /  hrl_ppo_corridors_500k_2
_HRL_ALGO = re.compile(
    r"^hrl_(dqn|ppo)_corridors_(\d+)k_(\d+)$"
)
# lvl1_ar5_500k_1
_PPO_SINGLE = re.compile(
    r"^lvl(\d+)_ar(\d+)_(\d+)k_(\d+)$"
)
# curriculum_500k  (if any)
_CURRICULUM = re.compile(
    r"^curriculum_(\d+)k(?:_(\d+))?$"
)


def _parse_run_name(name: str) -> dict:
    """Extract structured metadata from a TensorBoard run directory name."""
    m = _HRL_PILOT.match(name)
    if m:
        levels = [int(x) for x in m.group(1).split("_") if x]
        return dict(
            agent_type="hrl_pilot",
            levels=str(levels),
            action_repeat=int(m.group(2)),
            total_timesteps=int(m.group(3)) * 1000,
        )

    m = _HRL_ALGO.match(name)
    if m:
        return dict(
            agent_type=f"hrl_{m.group(1)}",
            levels='"corridors"',
            action_repeat=None,
            total_timesteps=int(m.group(2)) * 1000,
        )

    m = _PPO_SINGLE.match(name)
    if m:
        return dict(
            agent_type="ppo",
            levels=f"[{m.group(1)}]",
            action_repeat=int(m.group(2)),
            total_timesteps=int(m.group(3)) * 1000,
        )

    m = _CURRICULUM.match(name)
    if m:
        return dict(
            agent_type="ppo",
            levels='"curriculum"',
            action_repeat=None,
            total_timesteps=int(m.group(1)) * 1000,
        )

    # Fallback: store the raw name with unknown type
    return dict(
        agent_type="unknown",
        levels=None,
        action_repeat=None,
        total_timesteps=None,
    )


def _find_event_file(run_dir: Path) -> Path | None:
    for f in run_dir.iterdir():
        if f.name.startswith("events.out.tfevents."):
            return f
    return None


# ---------------------------------------------------------------------------
# TensorBoard metric extraction
# ---------------------------------------------------------------------------

def _extract_metrics(event_file_dir: str) -> list[tuple[str, int, float, float]]:
    """Return [(tag, step, wall_time, value), ...] from a TB run directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
            SCALARS,
        )
    except ImportError:
        print("Warning: tensorboard not available, skipping metric extraction")
        return []

    ea = EventAccumulator(event_file_dir, size_guidance={SCALARS: 0})
    ea.Reload()
    rows: list[tuple[str, int, float, float]] = []
    for tag in ea.Tags().get("scalars", []):
        for ev in ea.Scalars(tag):
            rows.append((tag, ev.step, ev.wall_time, ev.value))
    return rows


# ---------------------------------------------------------------------------
# Sync runs
# ---------------------------------------------------------------------------

def sync_runs() -> int:
    """Scan tensorboard_logs/, upsert into training_runs + metric_snapshots.

    Returns the number of runs synced (new or updated).
    """
    if not TB_DIR.is_dir():
        return 0

    db = get_db()
    synced = 0
    now = datetime.now(timezone.utc).isoformat()

    for entry in sorted(TB_DIR.iterdir()):
        if not entry.is_dir():
            continue
        run_name = entry.name
        event_file = _find_event_file(entry)
        if event_file is None:
            continue

        mtime = event_file.stat().st_mtime

        # Check if already synced and up-to-date
        row = db.execute(
            "SELECT id, event_file_mtime FROM training_runs WHERE run_name = ?",
            (run_name,),
        ).fetchone()

        if row and row["event_file_mtime"] and abs(row["event_file_mtime"] - mtime) < 1:
            continue  # unchanged

        meta = _parse_run_name(run_name)
        status = "running" if (time.time() - mtime) < 600 else "completed"

        if row:
            run_id = row["id"]
            db.execute(
                """UPDATE training_runs
                   SET agent_type=?, levels=?, action_repeat=?, total_timesteps=?,
                       status=?, event_file_path=?, event_file_mtime=?, synced_at=?
                   WHERE id=?""",
                (
                    meta["agent_type"], meta["levels"], meta["action_repeat"],
                    meta["total_timesteps"], status, str(event_file), mtime, now,
                    run_id,
                ),
            )
        else:
            # Extract created_at from event file timestamp embedded in name
            parts = event_file.name.split(".")
            created_at = None
            for p in parts:
                try:
                    ts = float(p)
                    if ts > 1_000_000_000:
                        created_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                        break
                except ValueError:
                    continue

            cur = db.execute(
                """INSERT INTO training_runs
                   (run_name, agent_type, levels, action_repeat, total_timesteps,
                    status, event_file_path, event_file_mtime, created_at, synced_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_name, meta["agent_type"], meta["levels"],
                    meta["action_repeat"], meta["total_timesteps"],
                    status, str(event_file), mtime, created_at, now,
                ),
            )
            run_id = cur.lastrowid

        # Re-extract metrics
        db.execute("DELETE FROM metric_snapshots WHERE run_id = ?", (run_id,))
        metrics = _extract_metrics(str(entry))
        if metrics:
            db.executemany(
                """INSERT OR IGNORE INTO metric_snapshots (run_id, tag, step, wall_time, value)
                   VALUES (?,?,?,?,?)""",
                [(run_id, tag, step, wt, val) for tag, step, wt, val in metrics],
            )

        db.commit()
        synced += 1

    db.close()
    return synced


# ---------------------------------------------------------------------------
# Sync models
# ---------------------------------------------------------------------------

# Expected obs dimensions per agent type
_EXPECTED_OBS = {
    "ppo": 24,         # PathAugmentedObs24 (sin/cos rotation)
    "hrl": 24,         # WaypointEnv (24-dim)
    "alphazero": None,  # .pt/.onnx — not SB3, skip check
    "unknown": None,
}


def _check_compatible(fpath: Path, agent_type: str) -> int:
    """Check if an SB3 .zip model's obs space matches the current code for its agent type."""
    if fpath.suffix != ".zip":
        return 1
    expected = _EXPECTED_OBS.get(agent_type)
    if expected is None:
        return 1
    try:
        from stable_baselines3 import PPO
        m = PPO.load(str(fpath), device="cpu")
        obs_dim = m.observation_space.shape[0]
        del m
        return 1 if obs_dim == expected else 0
    except Exception:
        return 0


def sync_models() -> int:
    """Scan models/ for checkpoints, upsert into model_checkpoints."""
    if not MODELS_DIR.is_dir():
        return 0

    db = get_db()
    synced = 0
    extensions = {".zip", ".pt", ".onnx"}

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix not in extensions:
                continue

            rel_path = str(fpath.relative_to(PROJECT_ROOT))
            stat = fpath.stat()

            # Infer agent type from path
            rel_lower = rel_path.lower()
            if "alphazero" in rel_lower:
                agent_type = "alphazero"
            elif "hrl" in rel_lower:
                agent_type = "hrl"
            elif "ppo" in rel_lower or fpath.suffix == ".zip":
                agent_type = "ppo"
            else:
                agent_type = "unknown"

            # Infer level from directory path
            parts = Path(root).relative_to(MODELS_DIR).parts
            level = None
            for p in parts:
                if p.isdigit():
                    level = p
                    break
                if p == "curriculum":
                    level = "curriculum"
                    break
                # e.g. "level_4" -> "4"
                lm = re.search(r"(\d+)", p)
                if lm and p not in ("_archive",):
                    level = lm.group(1)
                    break
            # Use parent dir path as label if still None
            if level is None:
                dir_rel = str(Path(root).relative_to(MODELS_DIR))
                if dir_rel != ".":
                    level = dir_rel

            # Parse iteration from filename
            iteration = None
            m = re.search(r"iter_(\d+)", fname)
            if m:
                iteration = int(m.group(1))

            is_best = 1 if "best" in fname.lower() else 0
            compatible = _check_compatible(fpath, agent_type)

            db.execute(
                """INSERT INTO model_checkpoints
                   (path, agent_type, level, filename, file_type,
                    file_size_bytes, file_mtime, is_best, iteration, compatible)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET
                    file_size_bytes=excluded.file_size_bytes,
                    file_mtime=excluded.file_mtime,
                    compatible=excluded.compatible""",
                (
                    rel_path, agent_type, level, fname, fpath.suffix.lstrip("."),
                    stat.st_size, stat.st_mtime, is_best, iteration, compatible,
                ),
            )
            synced += 1

    db.commit()
    db.close()
    return synced


def sync_all() -> dict:
    """Run all sync operations. Returns counts."""
    runs = sync_runs()
    models = sync_models()
    return {"synced_runs": runs, "synced_models": models}
