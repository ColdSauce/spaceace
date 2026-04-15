"""API blueprint for on-demand agent assessments."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, request

from dashboard.assess_runner import start_assessment
from dashboard.db import get_db

assessments_bp = Blueprint("assessments", __name__)


@assessments_bp.route("/assessments")
def list_assessments():
    db = get_db()
    rows = db.execute(
        """SELECT id, agent_type, model_path, levels, episodes_per_level,
                  status, started_at, completed_at,
                  completion_rate, crash_rate, timeout_rate,
                  mean_reward, mean_steps
           FROM assessments ORDER BY id DESC"""
    ).fetchall()
    db.close()
    return jsonify({"assessments": [dict(r) for r in rows]})


@assessments_bp.route("/assessments/<int:aid>")
def get_assessment(aid: int):
    db = get_db()
    row = db.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()
    db.close()
    if row is None:
        return jsonify({"error": "not found"}), 404
    result = dict(row)
    if result.get("report_json"):
        result["report"] = json.loads(result["report_json"])
        del result["report_json"]
    return jsonify({"assessment": result})


@assessments_bp.route("/assessments", methods=["POST"])
def create_assessment():
    data = request.get_json(force=True)
    agent_type = data.get("agent_type")
    levels = data.get("levels", [0])
    episodes = data.get("episodes", 5)
    model_path = data.get("model_path")

    if not agent_type:
        return jsonify({"error": "agent_type required"}), 400

    if isinstance(levels, str):
        levels = [int(x.strip()) for x in levels.split(",")]

    db = get_db()
    cur = db.execute(
        """INSERT INTO assessments (agent_type, model_path, levels, episodes_per_level, status)
           VALUES (?, ?, ?, ?, 'pending')""",
        (agent_type, model_path, json.dumps(levels), episodes),
    )
    db.commit()
    aid = cur.lastrowid
    db.close()

    started = start_assessment(aid)
    if not started:
        return jsonify({"assessment_id": aid, "status": "pending",
                        "warning": "Another assessment is running; queued."}), 202

    return jsonify({"assessment_id": aid, "status": "running"}), 201
