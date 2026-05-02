"""Tests for the inspect_run CLI helpers.

These exercise the pure-function parts (log parsing, metric summaries,
fallback behavior) so they can run without a populated dashboard DB.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inspect_run.py"


@pytest.fixture(scope="module")
def inspect_run():
    spec = importlib.util.spec_from_file_location("inspect_run", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_errors_picks_up_traceback(tmp_path, inspect_run):
    log = tmp_path / "job_test.log"
    log.write_text(
        "Curriculum: 165 levels x 4 stages = 660 total stages\n"
        "Progression per level: base -> +1 -> +2 -> +3 pickups\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "<frozen runpy>", line 198, in _run_module_as_main\n'
        '  File "/foo/bar.py", line 49, in calibrate_max_steps\n'
        "    disk_cache = _load_cache(cache)\n"
        "json.decoder.JSONDecodeError: Expecting property name in double quotes\n"
        "\n"
        "(some unrelated noise after)\n"
    )
    blocks = inspect_run.extract_errors(log)
    assert len(blocks) == 1
    assert blocks[0].startswith("Traceback (most recent call last):")
    assert "JSONDecodeError" in blocks[0]


def test_extract_errors_returns_empty_for_clean_log(tmp_path, inspect_run):
    log = tmp_path / "clean.log"
    log.write_text("Stage 1: levels [0]\nIteration 5\nGames: 100/100, examples: 50, ...\n")
    assert inspect_run.extract_errors(log) == []


def test_extract_errors_handles_missing_file(tmp_path, inspect_run):
    assert inspect_run.extract_errors(tmp_path / "does_not_exist.log") == []


def test_detect_tb_run_name(tmp_path, inspect_run):
    log = tmp_path / "job.log"
    log.write_text(
        "Using cpu device\n"
        "Starting training (run: lvl0_ar3_500k)...\n"
        "Logging to tensorboard_logs/lvl0_ar3_500k_2\n"
    )
    assert inspect_run.detect_tb_run_name(log) == "lvl0_ar3_500k_2"


def test_detect_tb_run_name_returns_none_when_absent(tmp_path, inspect_run):
    log = tmp_path / "noisy.log"
    log.write_text("nothing relevant in here\n")
    assert inspect_run.detect_tb_run_name(log) is None


def test_tail_lines(tmp_path, inspect_run):
    log = tmp_path / "tail.log"
    log.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")
    assert inspect_run.tail_lines(log, 3) == ["line 8", "line 9", "line 10"]


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE training_runs (
            id INTEGER PRIMARY KEY,
            run_name TEXT UNIQUE NOT NULL,
            agent_type TEXT,
            levels TEXT,
            action_repeat INTEGER,
            total_timesteps INTEGER,
            obs_strategy TEXT,
            reward_strategy TEXT,
            status TEXT,
            event_file_path TEXT,
            event_file_mtime REAL,
            created_at TEXT,
            synced_at TEXT,
            notes TEXT DEFAULT ''
        );
        CREATE TABLE metric_snapshots (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            step INTEGER NOT NULL,
            wall_time REAL NOT NULL,
            value REAL NOT NULL
        );
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            trainer TEXT,
            args_json TEXT,
            display_name TEXT,
            pid INTEGER,
            status TEXT,
            started_at TEXT,
            finished_at TEXT,
            exit_code INTEGER,
            run_name TEXT,
            log_path TEXT,
            error TEXT
        );
        CREATE TABLE model_checkpoints (
            id INTEGER PRIMARY KEY,
            path TEXT,
            agent_type TEXT,
            level TEXT,
            filename TEXT,
            file_type TEXT,
            file_size_bytes INTEGER,
            file_mtime REAL,
            is_best INTEGER DEFAULT 0,
            iteration INTEGER,
            compatible INTEGER DEFAULT 1
        );
        """
    )
    return conn


def test_summarize_run_metrics_orders_known_tags_first(tmp_path, inspect_run):
    db = _make_db(tmp_path / "test.db")
    db.execute(
        "INSERT INTO training_runs (id, run_name) VALUES (1, 'run_a')"
    )
    snapshots = [
        ("custom/zzz", 0, 0.0, 1.0),
        ("custom/zzz", 1, 0.0, 2.0),
        ("rollout/ep_rew_mean", 0, 0.0, -10.0),
        ("rollout/ep_rew_mean", 1, 0.0, 5.0),
        ("rollout/ep_rew_mean", 2, 0.0, 8.0),
    ]
    db.executemany(
        "INSERT INTO metric_snapshots (run_id, tag, step, wall_time, value) VALUES (1,?,?,?,?)",
        snapshots,
    )
    db.commit()
    summary = inspect_run.summarize_run_metrics(db, 1)
    keys = list(summary.keys())
    assert keys == ["rollout/ep_rew_mean", "custom/zzz"]
    rew = summary["rollout/ep_rew_mean"]
    assert rew["n"] == 3
    assert rew["first"] == -10.0
    assert rew["last"] == 8.0
    assert rew["best"] == 8.0
    assert rew["worst"] == -10.0
    assert rew["last_step"] == 2


def test_build_job_summary_links_tb_run_when_present(tmp_path, inspect_run, monkeypatch):
    db_path = tmp_path / "test.db"
    db = _make_db(db_path)
    log_path = tmp_path / "job_42.log"
    log_path.write_text(
        "Logging to tensorboard_logs/lvl0_ar3_500k_1\n"
        "Iteration 5\n"
    )
    db.execute(
        "INSERT INTO training_runs (id, run_name) VALUES (1, 'lvl0_ar3_500k_1')"
    )
    db.execute(
        "INSERT INTO metric_snapshots (run_id, tag, step, wall_time, value) "
        "VALUES (1, 'rollout/ep_rew_mean', 0, 0.0, 1.5)"
    )
    db.execute(
        "INSERT INTO jobs (id, trainer, args_json, display_name, status, log_path) "
        "VALUES (42, 'ppo', '{\"level\": 0}', 'PPO / level 0', 'completed', ?)",
        (str(log_path),),
    )
    db.commit()

    summary = inspect_run.build_job_summary(db, 42)
    assert summary is not None
    assert summary["id"] == 42
    assert summary["trainer"] == "ppo"
    assert summary["args"] == {"level": 0}
    assert summary["tb_run_name"] == "lvl0_ar3_500k_1"
    assert summary["tb_run_id"] == 1
    assert "rollout/ep_rew_mean" in summary["tb_metrics"]


def test_build_job_summary_returns_none_for_missing_id(tmp_path, inspect_run):
    db = _make_db(tmp_path / "test.db")
    assert inspect_run.build_job_summary(db, 999) is None
