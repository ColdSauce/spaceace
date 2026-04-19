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

    t_iter = time.perf_counter()
    entries = sorted(TB_DIR.iterdir())
    print(f"[sync_runs] iterdir({len(entries)} entries) took {time.perf_counter()-t_iter:.2f}s")

    for entry in entries:
        if not entry.is_dir():
            continue
        run_name = entry.name
        t_run = time.perf_counter()

        t = time.perf_counter()
        event_file = _find_event_file(entry)
        t_find = time.perf_counter() - t
        if event_file is None:
            continue

        t = time.perf_counter()
        mtime = event_file.stat().st_mtime
        t_stat = time.perf_counter() - t

        # Check if already synced and up-to-date
        row = db.execute(
            "SELECT id, event_file_mtime FROM training_runs WHERE run_name = ?",
            (run_name,),
        ).fetchone()

        # Skip if unchanged. For active runs (mtime ticking every few seconds),
        # only re-parse if the file grew by >=30s of training to avoid
        # re-reading the entire ~20MB tfevents on every dashboard refresh.
        if row and row["event_file_mtime"]:
            delta = mtime - row["event_file_mtime"]
            if delta < 30:
                continue

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
        t = time.perf_counter()
        metrics = _extract_metrics(str(entry))
        t_extract = time.perf_counter() - t
        t = time.perf_counter()
        if metrics:
            db.executemany(
                """INSERT OR IGNORE INTO metric_snapshots (run_id, tag, step, wall_time, value)
                   VALUES (?,?,?,?,?)""",
                [(run_id, tag, step, wt, val) for tag, step, wt, val in metrics],
            )
        t_insert = time.perf_counter() - t
        size_mb = event_file.stat().st_size / 1e6
        print(
            f"[sync_runs] {run_name}: total={time.perf_counter()-t_run:.2f}s "
            f"find={t_find*1000:.0f}ms stat={t_stat*1000:.0f}ms "
            f"extract={t_extract:.2f}s ({len(metrics)} pts, {size_mb:.1f}MB) "
            f"insert={t_insert*1000:.0f}ms"
        )

        db.commit()
        synced += 1

    db.close()
    return synced


def sync_one_run(run_name: str) -> bool:
    """Sync a single tensorboard run by name. Cheap — used by polling endpoints.

    Returns True if the run was found and (re)synced, False if it doesn't exist
    on disk yet.
    """
    if not TB_DIR.is_dir():
        return False
    run_dir = TB_DIR / run_name
    if not run_dir.is_dir():
        return False
    event_file = _find_event_file(run_dir)
    if event_file is None:
        return False

    mtime = event_file.stat().st_mtime
    db = get_db()
    try:
        row = db.execute(
            "SELECT id, event_file_mtime FROM training_runs WHERE run_name = ?",
            (run_name,),
        ).fetchone()

        # Skip re-extraction if mtime hasn't moved meaningfully
        if row and row["event_file_mtime"] and (mtime - row["event_file_mtime"] < 10):
            return True

        meta = _parse_run_name(run_name)
        status = "running" if (time.time() - mtime) < 600 else "completed"
        now = datetime.now(timezone.utc).isoformat()

        if row:
            run_id = row["id"]
            db.execute(
                """UPDATE training_runs
                   SET agent_type=?, levels=?, action_repeat=?, total_timesteps=?,
                       status=?, event_file_path=?, event_file_mtime=?, synced_at=?
                   WHERE id=?""",
                (meta["agent_type"], meta["levels"], meta["action_repeat"],
                 meta["total_timesteps"], status, str(event_file), mtime, now, run_id),
            )
        else:
            cur = db.execute(
                """INSERT INTO training_runs
                   (run_name, agent_type, levels, action_repeat, total_timesteps,
                    status, event_file_path, event_file_mtime, created_at, synced_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (run_name, meta["agent_type"], meta["levels"], meta["action_repeat"],
                 meta["total_timesteps"], status, str(event_file), mtime, now, now),
            )
            run_id = cur.lastrowid

        db.execute("DELETE FROM metric_snapshots WHERE run_id = ?", (run_id,))
        metrics = _extract_metrics(str(run_dir))
        if metrics:
            db.executemany(
                """INSERT OR IGNORE INTO metric_snapshots (run_id, tag, step, wall_time, value)
                   VALUES (?,?,?,?,?)""",
                [(run_id, tag, step, wt, val) for tag, step, wt, val in metrics],
            )
        db.commit()
        return True
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Sync models
# ---------------------------------------------------------------------------

# Expected obs dimensions per agent type. PPO uses PathAugmentedObs23 which
# now writes 16 base + 8 derived + 16 fine raycasts = 40 dims. Older 24-dim
# checkpoints are marked incompatible so the dashboard hides them from Watch.
_EXPECTED_OBS = {
    "ppo": 40,
    "hrl": 24,         # WaypointEnv (24-dim, unchanged)
    "alphazero": None,  # .pt/.onnx — not SB3, skip check
    "unknown": None,
}


def _check_compatible(fpath: Path, agent_type: str) -> int:
    """Cheap obs-space compatibility check.

    SB3 stores `observation_space` in the zip's `data` JSON as a base64-pickled
    Box under `:serialized:`. We decode that and read `.shape[0]` directly.
    Avoids the ~3-5s cost of a full PPO.load (no torch import).
    """
    if fpath.suffix != ".zip":
        return 1
    expected = _EXPECTED_OBS.get(agent_type)
    if expected is None:
        return 1
    try:
        import base64
        import json
        import pickle
        import zipfile

        with zipfile.ZipFile(str(fpath)) as zf:
            with zf.open("data") as f:
                data = json.load(f)

        obs_space = data.get("observation_space")
        if not isinstance(obs_space, dict):
            return 1
        serialized = obs_space.get(":serialized:")
        if not serialized:
            return 1
        space = pickle.loads(base64.b64decode(serialized))
        shape = getattr(space, "shape", None)
        if not shape:
            return 1
        return 1 if int(shape[0]) == expected else 0
    except Exception:
        return 0


def sync_models() -> int:
    """Scan models/ for checkpoints, upsert into model_checkpoints."""
    if not MODELS_DIR.is_dir():
        return 0

    db = get_db()
    synced = 0
    extensions = {".zip", ".pt", ".onnx"}

    # Pre-load (path -> file_mtime) so we can skip unchanged checkpoints without
    # re-running the (very expensive) compatibility check.
    known = {
        row["path"]: row["file_mtime"]
        for row in db.execute("SELECT path, file_mtime FROM model_checkpoints").fetchall()
    }

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix not in extensions:
                continue

            rel_path = str(fpath.relative_to(PROJECT_ROOT))
            stat = fpath.stat()
            if rel_path in known and known[rel_path] is not None \
                    and abs(known[rel_path] - stat.st_mtime) < 1:
                continue  # unchanged, skip the costly PPO.load

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
