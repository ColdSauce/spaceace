"""Launch and manage training subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from dashboard.db import get_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "dashboard" / "job_logs"
LOGS_DIR.mkdir(exist_ok=True)

# Map trainer names to their python module commands
TRAINER_COMMANDS = {
    "ppo": ["-m", "spaceace.agents.ppo.train"],
    "ppo_curriculum": ["-m", "spaceace.agents.ppo.curriculum_train"],
    "alphazero": ["-m", "spaceace.agents.alphazero.train"],
    "hrl_pilot": ["-m", "spaceace.agents.hrl.train_pilot"],
}

# Which CLI args each trainer accepts (for building the command)
TRAINER_ARGS = {
    "ppo": {
        "level": {"flag": "--level", "type": "int"},
        "timesteps": {"flag": "--timesteps", "type": "int"},
        "max_steps": {"flag": "--max-steps", "type": "int"},
        "action_repeat": {"flag": "--action-repeat", "type": "int"},
        "obs": {"flag": "--obs", "type": "str"},
        "reward": {"flag": "--reward", "type": "str"},
        "seed": {"flag": "--seed", "type": "int"},
        "n_envs": {"flag": "--n-envs", "type": "int"},
        "resume": {"flag": "--resume", "type": "str"},
    },
    "ppo_curriculum": {
        "stages": {"flag": "--stages", "type": "str"},
        "timesteps": {"flag": "--timesteps", "type": "int"},
        "advance_win_rate": {"flag": "--advance-win-rate", "type": "float"},
        "min_steps_per_stage": {"flag": "--min-steps-per-stage", "type": "int"},
        "action_repeat": {"flag": "--action-repeat", "type": "int"},
        "seed": {"flag": "--seed", "type": "int"},
    },
    "alphazero": {
        "level": {"flag": "--level", "type": "int"},
        "iterations": {"flag": "--iterations", "type": "int"},
        "games_per_iter": {"flag": "--games-per-iter", "type": "int"},
        "num_sims": {"flag": "--num-sims", "type": "int"},
        "action_repeat": {"flag": "--action-repeat", "type": "int"},
        "epochs": {"flag": "--epochs", "type": "int"},
        "batch_size": {"flag": "--batch-size", "type": "int"},
        "lr": {"flag": "--lr", "type": "float"},
        "max_steps": {"flag": "--max-steps", "type": "int"},
    },
    "hrl_pilot": {
        "levels": {"flag": "--levels", "type": "str"},
        "timesteps": {"flag": "--timesteps", "type": "int"},
        "max_steps": {"flag": "--max-steps", "type": "int"},
        "action_repeat": {"flag": "--action-repeat", "type": "int"},
        "seed": {"flag": "--seed", "type": "int"},
    },
}

# Track running processes (pid -> Popen) so we can kill them
_processes: dict[int, subprocess.Popen] = {}
_lock = threading.Lock()


def _build_command(trainer: str, args: dict) -> list[str]:
    """Build the subprocess command list."""
    cmd = ["uv", "run", "python"] + TRAINER_COMMANDS[trainer]
    schema = TRAINER_ARGS.get(trainer, {})
    for key, value in args.items():
        if value is None or value == "":
            continue
        spec = schema.get(key)
        if spec:
            cmd.extend([spec["flag"], str(value)])
    return cmd


def _fmt_steps(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def _make_display_name(trainer: str, args: dict) -> str:
    """Build a human-readable name for the job."""
    labels = {"ppo": "PPO", "ppo_curriculum": "PPO Curriculum",
              "alphazero": "AlphaZero", "hrl_pilot": "HRL Pilot"}
    parts = [labels.get(trainer, trainer)]
    if "level" in args and args["level"] is not None:
        parts.append(f"level {args['level']}")
    if "levels" in args and args["levels"]:
        parts.append(f"levels {args['levels']}")
    if "stages" in args and args["stages"]:
        parts.append(f"stages {args['stages']}")
    if "timesteps" in args and args["timesteps"]:
        parts.append(f"{_fmt_steps(int(args['timesteps']))} steps")
    if "iterations" in args and args["iterations"]:
        parts.append(f"{args['iterations']} iters")
    return " / ".join(parts)


def _run_job(job_id: int, cmd: list[str], log_path: Path) -> None:
    """Run in a background thread: launch process, stream output, update DB on completion."""
    db = get_db()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with _lock:
            _processes[job_id] = proc

        db.execute(
            "UPDATE jobs SET pid=?, status='running', started_at=? WHERE id=?",
            (proc.pid, datetime.now(timezone.utc).isoformat(), job_id),
        )
        db.commit()

        proc.wait()

        status = "completed" if proc.returncode == 0 else "failed"
        error = None
        if proc.returncode != 0:
            try:
                # Read last 500 chars of log for error context
                text = log_path.read_text()
                error = text[-500:] if len(text) > 500 else text
            except Exception:
                error = f"Exit code {proc.returncode}"

        db.execute(
            "UPDATE jobs SET status=?, finished_at=?, exit_code=? , error=? WHERE id=?",
            (status, datetime.now(timezone.utc).isoformat(), proc.returncode, error, job_id),
        )
        db.commit()

        # Auto-sync to pick up the new run's metrics
        from dashboard.sync import sync_all
        sync_all()

    except Exception as exc:
        db.execute(
            "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), str(exc), job_id),
        )
        db.commit()
    finally:
        with _lock:
            _processes.pop(job_id, None)
        db.close()


def launch_job(trainer: str, args: dict) -> int:
    """Create a job row, spawn the training process. Returns job ID."""
    if trainer not in TRAINER_COMMANDS:
        raise ValueError(f"Unknown trainer: {trainer}. Must be one of {list(TRAINER_COMMANDS)}")

    cmd = _build_command(trainer, args)
    display_name = _make_display_name(trainer, args)

    db = get_db()
    cur = db.execute(
        "INSERT INTO jobs (trainer, args_json, display_name, status) VALUES (?, ?, ?, 'pending')",
        (trainer, json.dumps(args), display_name),
    )
    db.commit()
    job_id = cur.lastrowid

    log_path = LOGS_DIR / f"job_{job_id}.log"
    db.execute("UPDATE jobs SET log_path=? WHERE id=?", (str(log_path), job_id))
    db.commit()
    db.close()

    t = threading.Thread(target=_run_job, args=(job_id, cmd, log_path), daemon=True)
    t.start()
    return job_id


def stop_job(job_id: int) -> bool:
    """Kill a running job. Returns True if it was running."""
    # Try in-memory handle first
    with _lock:
        proc = _processes.get(job_id)
    if proc is not None:
        proc.terminate()
        return True

    # Fall back to killing by PID from the database
    db = get_db()
    row = db.execute("SELECT pid FROM jobs WHERE id=? AND status='running'", (job_id,)).fetchone()
    db.close()
    if row and row["pid"] and _pid_alive(row["pid"]):
        import signal
        os.kill(row["pid"], signal.SIGTERM)
        return True

    return False


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def reap_stale_jobs() -> int:
    """Mark any 'running' jobs whose PID is dead as completed/failed.

    This handles server restarts and daemon threads that got lost.
    Returns the number of jobs reaped.
    """
    db = get_db()
    rows = db.execute(
        "SELECT id, pid, log_path FROM jobs WHERE status = 'running'"
    ).fetchall()
    reaped = 0
    for row in rows:
        pid = row["pid"]
        # If PID is alive, skip
        if pid is not None and _pid_alive(pid):
            continue
        # PID is dead or was never set — mark as finished
        if True:
            # Try to determine exit status from the log
            log_path = Path(row["log_path"]) if row["log_path"] else None
            error = None
            status = "interrupted"
            if log_path and log_path.exists():
                text = log_path.read_text()
                if "Training complete" in text:
                    status = "completed"
                elif "Traceback" in text:
                    status = "failed"
                    error = text[-500:] if len(text) > 500 else text
                else:
                    # Process died mid-training
                    error = "Process exited before training finished"
            elif pid is None:
                status = "failed"
                error = "Job never started"

            db.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=? WHERE id=?",
                (status, datetime.now(timezone.utc).isoformat(), error, row["id"]),
            )
            reaped += 1
    db.commit()
    db.close()
    return reaped


def get_log_tail(job_id: int, lines: int = 80) -> str:
    """Read the last N lines of a job's log file."""
    db = get_db()
    row = db.execute("SELECT log_path FROM jobs WHERE id=?", (job_id,)).fetchone()
    db.close()
    if not row or not row["log_path"]:
        return ""
    path = Path(row["log_path"])
    if not path.exists():
        return ""
    try:
        all_lines = path.read_text().splitlines()
        return "\n".join(all_lines[-lines:])
    except Exception:
        return ""


def _read_log(log_path: str | None) -> str:
    if not log_path:
        return ""
    p = Path(log_path)
    if not p.exists():
        return ""
    try:
        return p.read_text()
    except Exception:
        return ""


import re

_METRIC_LINE = re.compile(r"^\|\s+(\S+)\s+\|\s+(\S+)\s+\|$")
_TIMESTEPS_TARGET = re.compile(r"Timesteps:\s+([\d,]+)")
_TB_RUN_NAME = re.compile(r"Logging to tensorboard_logs/(\S+)")
_STAGE_LINE = re.compile(r"Stage (\d+): levels \[([^\]]+)\], max_steps=(\d+)")
_ADVANCE_LINE = re.compile(r">>> Advancing to stage (\d+)/(\d+): levels \[([^\]]+)\]")


def parse_progress(log_path: str | None) -> dict | None:
    """Parse the latest SB3 metric block from a job log.

    Returns a dict like:
        {
            "total_timesteps": 98304,
            "target_timesteps": 2000000,
            "pct": 4.9,
            "reward": -14.7,
            "completed": 0,
            "crashed": 1,
            "ep_len": 51,
            "fps": 3068,
            "elapsed": 32,
        }
    or None if no metrics found.
    """
    text = _read_log(log_path)
    if not text:
        return None

    # Find the target timesteps from the header
    target = None
    m = _TIMESTEPS_TARGET.search(text)
    if m:
        target = int(m.group(1).replace(",", ""))

    # Find the TensorBoard run name
    run_name = None
    m = _TB_RUN_NAME.search(text)
    if m:
        run_name = m.group(1)

    # Parse curriculum stages from header
    stages_info: list[dict] = []
    for m in _STAGE_LINE.finditer(text):
        levels = [int(x.strip()) for x in m.group(2).split(",")]
        stages_info.append({"stage": int(m.group(1)), "levels": levels, "max_steps": int(m.group(3))})

    # Track the latest advancement
    last_advance_stage = None
    for m in _ADVANCE_LINE.finditer(text):
        last_advance_stage = int(m.group(1))

    # Parse all metric lines — we want the last value for each key
    metrics: dict[str, str] = {}
    for line in text.splitlines():
        m = _METRIC_LINE.match(line)
        if m:
            metrics[m.group(1)] = m.group(2)

    if not metrics:
        return None

    def _float(key: str) -> float | None:
        v = metrics.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    total = _float("total_timesteps")
    pct = None
    if total is not None and target:
        pct = round(total / target * 100, 1)

    stage = _float("stage")
    win_rate = _float("smoothed_win_rate")
    stage_idx = int(stage) if stage is not None else 0
    total_stages = len(stages_info) if stages_info else None

    # Current stage levels
    current_levels = None
    current_max_steps = None
    if stages_info and stage_idx < len(stages_info):
        current_levels = stages_info[stage_idx]["levels"]
        current_max_steps = stages_info[stage_idx]["max_steps"]

    # Conquered stages (everything before current)
    conquered = stage_idx

    return {
        "total_timesteps": int(total) if total else None,
        "target_timesteps": target,
        "pct": pct,
        "reward": _float("ep_rew_mean"),
        "completed": _float("completed"),
        "crashed": _float("crashed"),
        "ep_len": _float("ep_len_mean"),
        "fps": _float("fps"),
        "elapsed": _float("time_elapsed"),
        "run_name": run_name,
        "stage": stage_idx + 1 if stage is not None else None,
        "total_stages": total_stages,
        "win_rate": win_rate,
        "current_levels": current_levels,
        "current_max_steps": current_max_steps,
        "conquered": conquered,
    }
