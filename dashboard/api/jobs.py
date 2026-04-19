"""API blueprint for training jobs."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, request

from dashboard.db import get_db
from dashboard.job_runner import (
    TRAINER_ARGS,
    TRAINER_COMMANDS,
    get_log_tail,
    launch_job,
    parse_alphazero_series,
    parse_progress,
    reap_stale_jobs,
    stop_job,
)
from dashboard.sync import sync_all

jobs_bp = Blueprint("jobs", __name__)


@jobs_bp.route("/jobs")
def list_jobs():
    reap_stale_jobs()
    db = get_db()
    rows = db.execute(
        """SELECT id, trainer, display_name, pid, status,
                  started_at, finished_at, exit_code, error, log_path
           FROM jobs ORDER BY id DESC"""
    ).fetchall()
    db.close()
    jobs = []
    for r in rows:
        j = dict(r)
        j["progress"] = parse_progress(j.pop("log_path", None))
        jobs.append(j)
    return jsonify({"jobs": jobs})


@jobs_bp.route("/jobs/<int:job_id>")
def get_job(job_id: int):
    reap_stale_jobs()
    db = get_db()
    row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    result = dict(row)
    result["args"] = json.loads(result.pop("args_json", "{}"))
    result["progress"] = parse_progress(result.get("log_path"))
    return jsonify({"job": result})


@jobs_bp.route("/jobs/<int:job_id>/log")
def get_job_log(job_id: int):
    lines = request.args.get("lines", 80, type=int)
    return jsonify({"log": get_log_tail(job_id, lines)})


@jobs_bp.route("/jobs/<int:job_id>/az-series")
def get_job_az_series(job_id: int):
    """Per-iteration AlphaZero metrics parsed from the job log."""
    db = get_db()
    row = db.execute("SELECT log_path FROM jobs WHERE id=?", (job_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify({"tags": parse_alphazero_series(row["log_path"])})


@jobs_bp.route("/jobs/<int:job_id>/sync", methods=["POST"])
def sync_job_metrics(job_id: int):
    """Re-sync tensorboard data (useful while a job is running)."""
    result = sync_all()
    return jsonify(result)


@jobs_bp.route("/jobs", methods=["POST"])
def create_job():
    data = request.get_json(force=True)
    trainer = data.get("trainer")
    args = data.get("args", {})

    if not trainer or trainer not in TRAINER_COMMANDS:
        return jsonify({
            "error": f"trainer must be one of: {list(TRAINER_COMMANDS)}",
        }), 400

    try:
        job_id = launch_job(trainer, args)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"job_id": job_id, "status": "running"}), 201


@jobs_bp.route("/jobs/<int:job_id>/stop", methods=["POST"])
def stop_job_route(job_id: int):
    stopped = stop_job(job_id)
    if not stopped:
        return jsonify({"error": "Job not running or not found"}), 404
    return jsonify({"ok": True, "status": "stopping"})


@jobs_bp.route("/trainers")
def list_trainers():
    """Return available trainer types and their configurable args."""
    result = {}
    for name in TRAINER_COMMANDS:
        result[name] = {
            k: {"flag": v["flag"], "type": v["type"]}
            for k, v in TRAINER_ARGS.get(name, {}).items()
        }
    return jsonify({"trainers": result})
