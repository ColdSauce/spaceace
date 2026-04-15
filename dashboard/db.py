"""SQLite schema and connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "spaceace_dashboard.db"

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
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


MIGRATIONS = [
    "ALTER TABLE training_runs ADD COLUMN notes TEXT DEFAULT ''",
    "ALTER TABLE model_checkpoints ADD COLUMN compatible INTEGER DEFAULT 1",
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
