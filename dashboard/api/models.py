"""API blueprint for model checkpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from dashboard.db import get_db

models_bp = Blueprint("models", __name__)


@models_bp.route("/models")
def list_models():
    db = get_db()
    agent_filter = request.args.get("agent_type")

    base = """SELECT id, path, agent_type, level, filename, file_type,
                     file_size_bytes, file_mtime, is_best, iteration, compatible
              FROM model_checkpoints WHERE compatible = 1"""

    if agent_filter:
        rows = db.execute(
            base + " AND agent_type = ? ORDER BY agent_type, level, iteration",
            (agent_filter,),
        ).fetchall()
    else:
        rows = db.execute(
            base + " ORDER BY agent_type, level, iteration"
        ).fetchall()
    db.close()
    return jsonify({"models": [dict(r) for r in rows]})
