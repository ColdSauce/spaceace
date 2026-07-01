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
# Trainer launching is retired: the AI is the offline ace solver
# (scripts/solve.py), which is run directly rather than through dashboard
# jobs. These tables stay (empty) so the jobs API and historical job
# records keep working.
TRAINER_COMMANDS: dict[str, list[str]] = {}

TRAINER_ARGS: dict[str, dict] = {}


def _build_command(trainer: str, args: dict) -> list[str]:
    """Build the subprocess command list."""
    cmd = ["uv", "run", "python"] + TRAINER_COMMANDS[trainer]
    schema = TRAINER_ARGS.get(trainer, {})
    for key, value in args.items():
        if value is None or value == "":
            continue
        spec = schema.get(key)
        if spec:
            if spec["type"] == "bool":
                if value:
                    cmd.append(spec["flag"])
            else:
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
              "alphazero": "AlphaZero", "alphazero_curriculum": "AlphaZero Curriculum",
              "hrl_pilot": "HRL Pilot"}
    parts = [labels.get(trainer, trainer)]
    if "level" in args and args["level"] is not None:
        parts.append(f"level {args['level']}")
    if "levels" in args and args["levels"]:
        parts.append(f"levels {args['levels']}")
    if "base_levels" in args and args["base_levels"]:
        parts.append(f"bases {args['base_levels']}")
    if "stages" in args and args["stages"]:
        parts.append(f"stages {args['stages']}")
    if "timesteps" in args and args["timesteps"]:
        parts.append(f"{_fmt_steps(int(args['timesteps']))} steps")
    if "iterations" in args and args["iterations"]:
        parts.append(f"{args['iterations']} iters")
    if args.get("fresh"):
        parts.append("fresh")
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
_TIMESTEPS_TARGET = re.compile(r"(?:Total )?[Tt]imesteps:\s+([\d,]+)")
_TB_RUN_NAME = re.compile(r"Logging to tensorboard_logs/(\S+)")
_STAGE_LINE = re.compile(r"Stage (\d+): levels \[([^\]]+)\], max_steps=(\d+)")
_ADVANCE_LINE = re.compile(r">>> Advancing to stage (\d+)/(\d+): levels \[([^\]]+)\]")

# AlphaZero-specific log parsing — AZ doesn't use SB3 or tensorboard.
_AZ_HEADER = re.compile(r"=== AlphaZero Training ===")
_AZ_BUDGET = re.compile(r"Total iteration budget:\s+(\d+)")
_AZ_STAGE_HEADER = re.compile(
    r"STAGE (\d+)/(\d+): levels=\[([^\]]+)\]\s+\(max_steps=(\d+),\s+sims=(\d+)\)"
)
_AZ_ITERATION = re.compile(
    r"Iteration (\d+)\s+\(stage (\d+),\s+L(\d+),\s+attempt (\d+)/(\d+)\)"
)
_AZ_EVAL = re.compile(
    r"Eval\s+\(L(\d+)\):\s+reward=([-\d.]+),\s+pickups=([\d.]+),"
    r"\s+wins=(\d+)/(\d+)\s+\(([\d.]+)%\),\s+smoothed=([\d.]+)%"
)
_AZ_LOSS = re.compile(r"Loss:\s+policy=([-\d.]+)\s+value=([-\d.]+)")
_AZ_GAMES = re.compile(
    r"Games:\s+(\d+)/(\d+),\s+examples:\s+(\d+),\s+wins:\s+(\d+),"
    r"\s+crashes:\s+(\d+),\s+mean_reward:\s+(-?[\d.]+),\s+mean_pickups:\s+([\d.]+)"
)


def parse_alphazero_series(log_path: str | None) -> dict[str, list[dict]]:
    """Per-iteration time series for an AlphaZero log.

    Walks the log sequentially, associating each metric block with the most
    recent `Iteration N` header. Returns a dict keyed by tag (matching the
    `alphazero/...` names the dashboard knows about).
    """
    text = _read_log(log_path)
    if not text or not _AZ_HEADER.search(text):
        return {}

    series: dict[str, list[dict]] = {
        "alphazero/policy_loss": [],
        "alphazero/value_loss": [],
        "alphazero/eval_reward": [],
        "alphazero/eval_pickups": [],
        "alphazero/eval_win_rate": [],
        "alphazero/smoothed_win_rate": [],
        "alphazero/curriculum_stage": [],
        "alphazero/self_play_completion": [],
        "alphazero/self_play_wins": [],
        "alphazero/self_play_crashes": [],
        "alphazero/self_play_reward": [],
        "alphazero/self_play_pickups": [],
    }

    current_iter: int | None = None
    self_play_seen = False  # first Games line per iter is self-play, second is eval

    for line in text.splitlines():
        m = _AZ_ITERATION.search(line)
        if m:
            current_iter = int(m.group(1))
            self_play_seen = False
            series["alphazero/curriculum_stage"].append({"step": current_iter, "value": int(m.group(2))})
            continue
        if current_iter is None:
            continue
        m = _AZ_GAMES.search(line)
        if m and not self_play_seen:
            self_play_seen = True
            step = current_iter
            total_games = int(m.group(2))
            wins = int(m.group(4))
            if total_games:
                series["alphazero/self_play_completion"].append({"step": step, "value": wins / total_games})
            series["alphazero/self_play_wins"].append({"step": step, "value": int(m.group(4))})
            series["alphazero/self_play_crashes"].append({"step": step, "value": int(m.group(5))})
            try:
                series["alphazero/self_play_reward"].append({"step": step, "value": float(m.group(6))})
            except ValueError:
                pass
            try:
                series["alphazero/self_play_pickups"].append({"step": step, "value": float(m.group(7))})
            except ValueError:
                pass
            continue
        m = _AZ_LOSS.search(line)
        if m:
            try:
                series["alphazero/policy_loss"].append({"step": current_iter, "value": float(m.group(1))})
                series["alphazero/value_loss"].append({"step": current_iter, "value": float(m.group(2))})
            except ValueError:
                pass
            continue
        m = _AZ_EVAL.search(line)
        if m:
            step = current_iter
            try:
                series["alphazero/eval_reward"].append({"step": step, "value": float(m.group(2))})
            except ValueError:
                pass
            try:
                series["alphazero/eval_pickups"].append({"step": step, "value": float(m.group(3))})
            except ValueError:
                pass
            try:
                wins = int(m.group(4))
                total = int(m.group(5))
                if total:
                    series["alphazero/eval_win_rate"].append({"step": step, "value": wins / total})
            except ValueError:
                pass
            try:
                series["alphazero/smoothed_win_rate"].append({"step": step, "value": float(m.group(7)) / 100.0})
            except ValueError:
                pass
            continue

    # Drop empty series so the frontend doesn't render blank charts.
    return {k: v for k, v in series.items() if v}


def _parse_alphazero_progress(text: str) -> dict | None:
    """Parse progress from an AlphaZero training log.

    Returns the same dict shape as `parse_progress` (so the dashboard UI
    works unchanged). `total_timesteps` / `target_timesteps` are iterations
    rather than env steps. `run_name` is None because AZ doesn't log to
    tensorboard; the dashboard should show a log-tail fallback.
    """
    budget = None
    m = _AZ_BUDGET.search(text)
    if m:
        budget = int(m.group(1))

    # Latest stage header seen (authoritative for current stage metadata).
    stage_matches = list(_AZ_STAGE_HEADER.finditer(text))
    total_stages = None
    stage_idx = None
    current_levels: list[int] | None = None
    current_max_steps: int | None = None
    if stage_matches:
        last = stage_matches[-1]
        stage_idx = int(last.group(1))  # 1-indexed in log
        total_stages = int(last.group(2))
        current_levels = [int(x.strip()) for x in last.group(3).split(",")]
        current_max_steps = int(last.group(4))

    # Latest iteration seen.
    iter_matches = list(_AZ_ITERATION.finditer(text))
    latest_iter = None
    if iter_matches:
        latest_iter = int(iter_matches[-1].group(1))

    # Latest eval line (reward, win rate, smoothed win rate).
    eval_matches = list(_AZ_EVAL.finditer(text))
    reward = None
    win_rate = None
    if eval_matches:
        last = eval_matches[-1]
        try:
            reward = float(last.group(2))
        except ValueError:
            pass
        try:
            win_rate = float(last.group(7)) / 100.0
        except ValueError:
            pass

    # Only treat this as an AZ job if we saw *something* (the header plus any
    # downstream line). The header alone is printed instantly at startup, so
    # we still want to surface a progress block even before iter 0 finishes.
    if latest_iter is None and stage_idx is None and budget is None:
        return None

    # Cumulative iteration count as a pseudo-timestep for the progress bar.
    total = (latest_iter + 1) if latest_iter is not None else 0
    pct = None
    if budget:
        pct = round(total / budget * 100, 1)

    return {
        "total_timesteps": total,
        "target_timesteps": budget,
        "pct": pct,
        "reward": reward,
        "completed": None,
        "crashed": None,
        "ep_len": None,
        "fps": None,
        "elapsed": None,
        "run_name": None,
        "trainer_kind": "alphazero",
        "stage": stage_idx,
        "total_stages": total_stages,
        "win_rate": win_rate,
        "current_levels": current_levels,
        "current_max_steps": current_max_steps,
        "conquered": (stage_idx - 1) if stage_idx else 0,
    }


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

    # Route AlphaZero logs to a dedicated parser — the SB3-shaped regexes
    # below won't match its output.
    if _AZ_HEADER.search(text):
        return _parse_alphazero_progress(text)

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
