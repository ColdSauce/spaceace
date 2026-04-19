"""SQLite schema and connection helpers."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# SQLite requires reliable fsync + POSIX file locking. RunPod's FUSE network
# volume doesn't provide either, which causes "database disk image is malformed"
# under concurrent writes. Override DB_PATH to a local-disk path on the pod.
DB_PATH = Path(os.environ.get(
    "DASHBOARD_DB_PATH",
    str(Path(__file__).parent / "spaceace_dashboard.db"),
))

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY,
    run_name        TEXT UNIQUE NOT NULL,
    agent_type      TEXT NOT NULL,
    levels          TEXT,               -- JSON array or descriptive string
    action_repeat   INTEGER,
    total_timesteps INTEGER,
    obs_strategy    TEXT DEFAULT 'path_augmented',
    reward_strategy TEXT DEFAULT 'dense_shaped',
    status          TEXT DEFAULT 'unknown',
    event_file_path TEXT,
    event_file_mtime REAL,
    created_at      TEXT,
    synced_at       TEXT
);

CREATE TABLE IF NOT EXISTS metric_snapshots (
    id      INTEGER PRIMARY KEY,
    run_id  INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    tag     TEXT NOT NULL,
    step    INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    value   REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_metric_run_tag_step
    ON metric_snapshots(run_id, tag, step);

CREATE TABLE IF NOT EXISTS assessments (
    id                INTEGER PRIMARY KEY,
    agent_type        TEXT NOT NULL,
    model_path        TEXT,
    levels            TEXT NOT NULL,     -- JSON array
    episodes_per_level INTEGER NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    started_at        TEXT,
    completed_at      TEXT,
    completion_rate   REAL,
    crash_rate        REAL,
    timeout_rate      REAL,
    mean_reward       REAL,
    mean_steps        REAL,
    report_json       TEXT,
    error_message     TEXT
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id              INTEGER PRIMARY KEY,
    path            TEXT UNIQUE NOT NULL,
    agent_type      TEXT NOT NULL,
    level           TEXT,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    file_size_bytes INTEGER,
    file_mtime      REAL,
    is_best         INTEGER DEFAULT 0,
    iteration       INTEGER
);
"""


def get_db() -> sqlite3.Connection:
    # 30s busy_timeout so the background sync thread doesn't fail with
    # "database is locked" when an API request happens to hold a lock.
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


MIGRATIONS = [
    "ALTER TABLE training_runs ADD COLUMN notes TEXT DEFAULT ''",
    "ALTER TABLE model_checkpoints ADD COLUMN compatible INTEGER DEFAULT 1",
    """CREATE TABLE IF NOT EXISTS leaderboard (
        id          INTEGER PRIMARY KEY,
        level       INTEGER NOT NULL,
        agent_type  TEXT NOT NULL,
        model_path  TEXT,
        outcome     TEXT NOT NULL,
        steps       INTEGER NOT NULL,
        pickups_collected INTEGER NOT NULL,
        pickups_total     INTEGER NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )""",
    """CREATE TABLE IF NOT EXISTS ghost_replays (
        id           INTEGER PRIMARY KEY,
        level        INTEGER NOT NULL,
        ghost_type   TEXT NOT NULL,
        steps        INTEGER NOT NULL,
        time_seconds REAL NOT NULL,
        frames_json  TEXT NOT NULL,
        created_at   TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(level, ghost_type)
    )""",
    """CREATE TABLE IF NOT EXISTS jobs (
        id          INTEGER PRIMARY KEY,
        trainer     TEXT NOT NULL,
        args_json   TEXT NOT NULL,
        display_name TEXT,
        pid         INTEGER,
        status      TEXT NOT NULL DEFAULT 'pending',
        started_at  TEXT,
        finished_at TEXT,
        exit_code   INTEGER,
        run_name    TEXT,
        log_path    TEXT,
        error       TEXT
    )""",
]


def init_db() -> None:
    conn = get_db()
    conn.executescript(SCHEMA)
    for sql in MIGRATIONS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()
