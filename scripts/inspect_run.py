"""Inspect training jobs and runs from the command line.

A single entry point for LLM agents (and humans) to ask "what happened in
this run?" without spelunking through huge log files, the SQLite schema,
and tensorboard event files separately.

Usage:
    uv run python scripts/inspect_run.py list                       # both jobs and runs (compact)
    uv run python scripts/inspect_run.py jobs [--limit N] [--json]
    uv run python scripts/inspect_run.py runs [--limit N] [--json]
    uv run python scripts/inspect_run.py job  <id>     [--json] [--log-lines N]
    uv run python scripts/inspect_run.py run  <name>   [--json]
    uv run python scripts/inspect_run.py errors <job-id>            # tracebacks/errors only
    uv run python scripts/inspect_run.py tail   <job-id> [-n N]     # last N lines of job log

Default output is concise plain text optimized for LLM consumption. Pass
``--json`` for machine-parseable output.

The script reads from ``dashboard/spaceace_dashboard.db`` and the
``dashboard/job_logs/`` directory. It is read-only — it never mutates the
database. If the DB is missing or empty, a clear hint is printed instead
of a stack trace.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "dashboard" / "spaceace_dashboard.db"
JOB_LOGS_DIR = PROJECT_ROOT / "dashboard" / "job_logs"
TB_DIR = PROJECT_ROOT / "tensorboard_logs"

# Metrics worth surfacing in a one-shot run summary, in display order.
SUMMARY_TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "episode/completed",
    "episode/crashed",
    "episode/pickups_collected",
    "curriculum/smoothed_win_rate",
    "curriculum/stage",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "time/fps",
]


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _missing_db_hint() -> None:
    print(f"No dashboard DB found at {DB_PATH}.", file=sys.stderr)
    print(
        "Run the dashboard once (`uv run python -m dashboard`) or call "
        "`dashboard.sync.sync_all()` to populate it.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------

_TRACEBACK_START = re.compile(r"^Traceback \(most recent call last\):")
_GENERIC_ERROR = re.compile(r"^(Error|FATAL|CRITICAL|.*?Error: )")
_TB_RUN_NAME = re.compile(r"Logging to tensorboard_logs/(\S+)")


def extract_errors(log_path: Path, max_chars: int = 4000) -> list[str]:
    """Pull tracebacks and obvious error lines out of a log.

    A traceback block is everything from a ``Traceback (...)`` line until the
    blank line or non-indented terminating exception line that follows. We
    trim each block at ``max_chars`` so a runaway error log doesn't dominate
    output.
    """
    if not log_path.exists():
        return []
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _TRACEBACK_START.match(line):
            j = i + 1
            # A traceback ends at the first non-indented line that isn't blank;
            # that line is the exception type/message and is part of the block.
            while j < n and (lines[j].startswith((" ", "\t")) or not lines[j]):
                j += 1
            if j < n:
                j += 1  # include the exception summary line
            block = "\n".join(lines[i:j])
            if len(block) > max_chars:
                block = block[:max_chars] + "\n... (truncated)"
            blocks.append(block)
            i = j
            continue
        # Standalone error lines (no preceding Traceback) are still useful.
        if _GENERIC_ERROR.match(line):
            blocks.append(line)
        i += 1
    return blocks


def detect_tb_run_name(log_path: Path) -> str | None:
    """Return the tensorboard run name a job log is writing to, if any."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    m = _TB_RUN_NAME.search(text)
    return m.group(1) if m else None


def tail_lines(log_path: Path, n: int) -> list[str]:
    if not log_path.exists():
        return []
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return []
    return text.splitlines()[-n:]


# ---------------------------------------------------------------------------
# Metric summaries
# ---------------------------------------------------------------------------

def summarize_run_metrics(conn: sqlite3.Connection, run_id: int) -> dict[str, dict]:
    """Return per-tag {first,last,best,worst,mean,n,last_step} summaries.

    Only includes tags in ``SUMMARY_TAGS`` plus any tag that already has data
    for this run — keeps output compact while remaining honest if a metric
    we care about wasn't logged.
    """
    rows = conn.execute(
        """SELECT tag, step, value FROM metric_snapshots
           WHERE run_id = ? ORDER BY tag, step""",
        (run_id,),
    ).fetchall()
    by_tag: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append((r["step"], r["value"]))

    out: dict[str, dict] = {}
    # Stable order: SUMMARY_TAGS first, then anything else alphabetically.
    ordered = [t for t in SUMMARY_TAGS if t in by_tag]
    ordered.extend(sorted(t for t in by_tag if t not in SUMMARY_TAGS))
    for tag in ordered:
        series = by_tag[tag]
        vals = [v for _, v in series]
        out[tag] = {
            "n": len(vals),
            "first": vals[0],
            "last": vals[-1],
            "best": max(vals),
            "worst": min(vals),
            "mean": sum(vals) / len(vals),
            "last_step": series[-1][0],
        }
    return out


# ---------------------------------------------------------------------------
# Builders: turn DB rows into agent-friendly dicts
# ---------------------------------------------------------------------------

def _job_log_path(row: sqlite3.Row) -> Path:
    """Resolve a job's log path, preferring whichever copy actually exists.

    The DB sometimes records an absolute path from the machine that ran the
    job (e.g. a different host's homedir). When that path is missing locally
    we fall back to the conventional ``dashboard/job_logs/job_<id>.log`` so
    the script keeps working on a fresh checkout.
    """
    fallback = JOB_LOGS_DIR / f"job_{row['id']}.log"
    if row["log_path"]:
        stored = Path(row["log_path"])
        if stored.exists():
            return stored
        if fallback.exists():
            return fallback
        return stored
    return fallback


def build_job_summary(conn: sqlite3.Connection, job_id: int, log_lines: int = 40) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    log_path = _job_log_path(row)

    args: dict[str, Any] = {}
    try:
        args = json.loads(row["args_json"]) if row["args_json"] else {}
    except (TypeError, json.JSONDecodeError):
        args = {"_raw": row["args_json"]}

    log_size = log_path.stat().st_size if log_path.exists() else 0
    errors = extract_errors(log_path)
    tb_run = detect_tb_run_name(log_path)

    # If we found a TB run name and it's been synced into the DB, attach a
    # metric summary so the LLM doesn't have to chase a second command.
    tb_metrics: dict[str, dict] = {}
    tb_run_id: int | None = None
    if tb_run:
        tb_row = conn.execute(
            "SELECT id FROM training_runs WHERE run_name = ?", (tb_run,)
        ).fetchone()
        if tb_row:
            tb_run_id = tb_row["id"]
            tb_metrics = summarize_run_metrics(conn, tb_run_id)

    return {
        "id": row["id"],
        "trainer": row["trainer"],
        "display_name": row["display_name"],
        "status": row["status"],
        "exit_code": row["exit_code"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "pid": row["pid"],
        "args": args,
        "run_name": row["run_name"],
        "log_path": str(log_path),
        "log_size_bytes": log_size,
        "log_exists": log_path.exists(),
        "log_tail": tail_lines(log_path, log_lines),
        "errors": errors,
        "stored_error": row["error"],
        "tb_run_name": tb_run,
        "tb_run_id": tb_run_id,
        "tb_metrics": tb_metrics,
    }


def build_run_summary(conn: sqlite3.Connection, identifier: str) -> dict[str, Any] | None:
    if identifier.isdigit():
        row = conn.execute(
            "SELECT * FROM training_runs WHERE id = ?", (int(identifier),)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM training_runs WHERE run_name = ?", (identifier,)
        ).fetchone()
    if row is None:
        return None

    metrics = summarize_run_metrics(conn, row["id"])

    # Best-effort link back to the job that produced this run.
    job_row = conn.execute(
        "SELECT id, status, exit_code FROM jobs WHERE run_name = ?",
        (row["run_name"],),
    ).fetchone()
    job_link = dict(job_row) if job_row else None

    # Checkpoints whose path looks related to this run (substring match on
    # run_name — coarse but works for the existing naming conventions).
    ckpts = conn.execute(
        """SELECT path, agent_type, level, filename, file_type, file_size_bytes,
                  is_best, iteration, compatible
           FROM model_checkpoints
           WHERE path LIKE ? OR filename LIKE ?
           ORDER BY file_mtime DESC LIMIT 10""",
        (f"%{row['run_name']}%", f"%{row['run_name']}%"),
    ).fetchall()

    return {
        "id": row["id"],
        "run_name": row["run_name"],
        "agent_type": row["agent_type"],
        "levels": row["levels"],
        "action_repeat": row["action_repeat"],
        "total_timesteps": row["total_timesteps"],
        "obs_strategy": row["obs_strategy"],
        "reward_strategy": row["reward_strategy"],
        "status": row["status"],
        "created_at": row["created_at"],
        "synced_at": row["synced_at"],
        "event_file_path": row["event_file_path"],
        "metrics": metrics,
        "linked_job": job_link,
        "checkpoints": [dict(c) for c in ckpts],
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _print_kv(items: Iterable[tuple[str, Any]], indent: int = 0) -> None:
    pad = " " * indent
    for k, v in items:
        if v is None or v == "":
            v = "-"
        print(f"{pad}{k}: {v}")


def render_job_text(summary: dict[str, Any]) -> None:
    print(f"=== job {summary['id']} — {summary['display_name'] or summary['trainer']} ===")
    _print_kv(
        [
            ("trainer", summary["trainer"]),
            ("status", summary["status"]),
            ("exit_code", summary["exit_code"]),
            ("started_at", summary["started_at"]),
            ("finished_at", summary["finished_at"]),
            ("pid", summary["pid"]),
            ("run_name", summary["run_name"] or summary["tb_run_name"]),
            ("log_path", summary["log_path"]),
            ("log_size", _fmt_bytes(summary["log_size_bytes"]) if summary["log_exists"] else "(missing)"),
        ]
    )
    if summary["args"]:
        print("\nargs:")
        for k, v in sorted(summary["args"].items()):
            print(f"  {k} = {v}")
    if summary["errors"]:
        print(f"\nerrors ({len(summary['errors'])}):")
        for block in summary["errors"][:5]:
            print("---")
            print(block)
        if len(summary["errors"]) > 5:
            print(f"... ({len(summary['errors']) - 5} more)")
    if summary["tb_metrics"]:
        print(f"\ntensorboard run: {summary['tb_run_name']}  (id={summary['tb_run_id']})")
        _print_metrics_table(summary["tb_metrics"])
    elif summary["tb_run_name"]:
        print(
            f"\ntensorboard run: {summary['tb_run_name']}"
            "  (not yet synced — run dashboard.sync.sync_all to refresh)"
        )
    if summary["log_tail"]:
        print(f"\nlog tail ({len(summary['log_tail'])} lines):")
        for line in summary["log_tail"]:
            print(f"  {line}")


def _print_metrics_table(metrics: dict[str, dict]) -> None:
    if not metrics:
        print("  (no metric snapshots)")
        return
    print(
        f"  {'tag':<32s} {'n':>5s} {'first':>10s} {'last':>10s} "
        f"{'best':>10s} {'last_step':>10s}"
    )
    for tag, s in metrics.items():
        print(
            f"  {tag:<32s} {s['n']:>5d} {s['first']:>10.3f} {s['last']:>10.3f} "
            f"{s['best']:>10.3f} {int(s['last_step']):>10d}"
        )


def render_run_text(summary: dict[str, Any]) -> None:
    print(f"=== run {summary['id']} — {summary['run_name']} ===")
    _print_kv(
        [
            ("agent_type", summary["agent_type"]),
            ("levels", summary["levels"]),
            ("action_repeat", summary["action_repeat"]),
            ("total_timesteps", summary["total_timesteps"]),
            ("obs_strategy", summary["obs_strategy"]),
            ("reward_strategy", summary["reward_strategy"]),
            ("status", summary["status"]),
            ("created_at", summary["created_at"]),
            ("synced_at", summary["synced_at"]),
            ("event_file_path", summary["event_file_path"]),
        ]
    )
    if summary["linked_job"]:
        j = summary["linked_job"]
        print(f"\nlinked job: id={j['id']} status={j['status']} exit={j['exit_code']}")
    print("\nmetrics:")
    _print_metrics_table(summary["metrics"])
    if summary["checkpoints"]:
        print(f"\ncheckpoints ({len(summary['checkpoints'])}):")
        for c in summary["checkpoints"]:
            best = " [best]" if c["is_best"] else ""
            iter_s = f" iter={c['iteration']}" if c["iteration"] is not None else ""
            size = _fmt_bytes(c["file_size_bytes"] or 0)
            print(f"  {c['path']} ({size}){iter_s}{best}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_jobs(args: argparse.Namespace) -> int:
    conn = _connect()
    if conn is None:
        _missing_db_hint()
        return 1
    rows = conn.execute(
        """SELECT id, trainer, display_name, status, exit_code,
                  started_at, finished_at, run_name
           FROM jobs ORDER BY id DESC LIMIT ?""",
        (args.limit,),
    ).fetchall()
    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2, default=str))
        return 0
    if not rows:
        print("(no jobs in dashboard DB)")
        return 0
    print(f"{'id':>4s}  {'status':<10s} {'exit':>4s}  {'trainer':<22s} {'name'}")
    for r in rows:
        exit_s = "" if r["exit_code"] is None else str(r["exit_code"])
        print(
            f"{r['id']:>4d}  {(r['status'] or '-'):<10s} {exit_s:>4s}  "
            f"{(r['trainer'] or '-'):<22s} {r['display_name'] or r['run_name'] or '-'}"
        )
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    conn = _connect()
    if conn is None:
        _missing_db_hint()
        return 1
    rows = conn.execute(
        """SELECT id, run_name, agent_type, levels, action_repeat,
                  total_timesteps, status, created_at
           FROM training_runs ORDER BY created_at DESC LIMIT ?""",
        (args.limit,),
    ).fetchall()
    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2, default=str))
        return 0
    if not rows:
        print("(no training runs in dashboard DB)")
        return 0
    print(f"{'id':>4s}  {'status':<10s} {'agent':<14s} {'steps':>10s}  {'name'}")
    for r in rows:
        steps = f"{r['total_timesteps']:,}" if r["total_timesteps"] else "-"
        print(
            f"{r['id']:>4d}  {(r['status'] or '-'):<10s} "
            f"{(r['agent_type'] or '-'):<14s} {steps:>10s}  {r['run_name']}"
        )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    conn = _connect()
    if conn is None:
        _missing_db_hint()
        return 1
    jobs = conn.execute(
        """SELECT id, trainer, display_name, status, exit_code,
                  started_at, finished_at, run_name
           FROM jobs ORDER BY id DESC LIMIT ?""",
        (args.limit,),
    ).fetchall()
    runs = conn.execute(
        """SELECT id, run_name, agent_type, levels, action_repeat,
                  total_timesteps, status, created_at
           FROM training_runs ORDER BY created_at DESC LIMIT ?""",
        (args.limit,),
    ).fetchall()

    if args.json:
        print(
            json.dumps(
                {
                    "jobs": [dict(r) for r in jobs],
                    "runs": [dict(r) for r in runs],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print("# Jobs")
    if not jobs:
        print("(no jobs in dashboard DB)")
    else:
        print(f"{'id':>4s}  {'status':<10s} {'exit':>4s}  {'trainer':<22s} {'name'}")
        for r in jobs:
            exit_s = "" if r["exit_code"] is None else str(r["exit_code"])
            print(
                f"{r['id']:>4d}  {(r['status'] or '-'):<10s} {exit_s:>4s}  "
                f"{(r['trainer'] or '-'):<22s} {r['display_name'] or r['run_name'] or '-'}"
            )
    print("\n# Runs")
    if not runs:
        print("(no training runs in dashboard DB)")
    else:
        print(f"{'id':>4s}  {'status':<10s} {'agent':<14s} {'steps':>10s}  {'name'}")
        for r in runs:
            steps = f"{r['total_timesteps']:,}" if r["total_timesteps"] else "-"
            print(
                f"{r['id']:>4d}  {(r['status'] or '-'):<10s} "
                f"{(r['agent_type'] or '-'):<14s} {steps:>10s}  {r['run_name']}"
            )
    return 0


def cmd_job(args: argparse.Namespace) -> int:
    conn = _connect()
    if conn is None:
        _missing_db_hint()
        return 1
    summary = build_job_summary(conn, args.job_id, log_lines=args.log_lines)
    if summary is None:
        print(f"job {args.job_id} not found", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        render_job_text(summary)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    conn = _connect()
    if conn is None:
        _missing_db_hint()
        return 1
    summary = build_run_summary(conn, args.identifier)
    if summary is None:
        print(f"run {args.identifier!r} not found", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        render_run_text(summary)
    return 0


def cmd_errors(args: argparse.Namespace) -> int:
    log_path = _resolve_job_log(args.job_id)
    if log_path is None:
        return 1
    blocks = extract_errors(log_path)
    if args.json:
        print(json.dumps({"job_id": args.job_id, "errors": blocks}, indent=2))
        return 0
    if not blocks:
        print(f"(no errors detected in {log_path.name})")
        return 0
    print(f"# {len(blocks)} error block(s) in {log_path.name}")
    for i, block in enumerate(blocks, 1):
        print(f"\n--- {i} ---")
        print(block)
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    log_path = _resolve_job_log(args.job_id)
    if log_path is None:
        return 1
    lines = tail_lines(log_path, args.lines)
    if args.json:
        print(json.dumps({"job_id": args.job_id, "lines": lines}, indent=2))
        return 0
    for line in lines:
        print(line)
    return 0


def _resolve_job_log(job_id: int) -> Path | None:
    """Find a job log file. Prefers DB-recorded path when it exists locally,
    otherwise falls back to the conventional ``job_<id>.log``."""
    fallback = JOB_LOGS_DIR / f"job_{job_id}.log"
    conn = _connect()
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT log_path FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
        if row and row["log_path"]:
            stored = Path(row["log_path"])
            if stored.exists():
                return stored
    if fallback.exists():
        return fallback
    print(f"no log found for job {job_id}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inspect_run",
        description="Introspect SpaceAce training jobs and tensorboard runs.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list recent jobs and runs")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_jobs = sub.add_parser("jobs", help="list recent jobs")
    p_jobs.add_argument("--limit", type=int, default=20)
    p_jobs.add_argument("--json", action="store_true")
    p_jobs.set_defaults(func=cmd_jobs)

    p_runs = sub.add_parser("runs", help="list recent training runs")
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.add_argument("--json", action="store_true")
    p_runs.set_defaults(func=cmd_runs)

    p_job = sub.add_parser("job", help="full picture of one job")
    p_job.add_argument("job_id", type=int)
    p_job.add_argument("--log-lines", type=int, default=40)
    p_job.add_argument("--json", action="store_true")
    p_job.set_defaults(func=cmd_job)

    p_run = sub.add_parser("run", help="full picture of one tensorboard run")
    p_run.add_argument("identifier", help="run name or numeric id")
    p_run.add_argument("--json", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_err = sub.add_parser("errors", help="extract tracebacks from a job log")
    p_err.add_argument("job_id", type=int)
    p_err.add_argument("--json", action="store_true")
    p_err.set_defaults(func=cmd_errors)

    p_tail = sub.add_parser("tail", help="tail of a job log")
    p_tail.add_argument("job_id", type=int)
    p_tail.add_argument("-n", "--lines", type=int, default=80)
    p_tail.add_argument("--json", action="store_true")
    p_tail.set_defaults(func=cmd_tail)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
