"""API blueprint for ghost replay data (AI and human best runs per level)."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, request

from dashboard.db import get_db

ghosts_bp = Blueprint("ghosts", __name__)


@ghosts_bp.route("/ghosts")
def get_ghosts():
    """Return ghost data for a level.

    Query params: ?level=N (required)
    Returns: { ai: {time, frames} | null, human: {time, frames} | null }
    """
    level = request.args.get("level")
    if level is None:
        return jsonify({"error": "level query param required"}), 400

    db = get_db()
    rows = db.execute(
        "SELECT ghost_type, time_seconds, frames_json FROM ghost_replays WHERE level = ?",
        (int(level),),
    ).fetchall()
    db.close()

    result = {"ai": None, "human": None, "goofy": None, "eager": None, "rewind": None}
    for row in rows:
        result[row["ghost_type"]] = {
            "time": row["time_seconds"],
            "frames": json.loads(row["frames_json"]),
        }
    return jsonify(result)


@ghosts_bp.route("/ghosts", methods=["POST"])
def save_ghost():
    """Upload a ghost recording. Only saves if faster than existing."""
    data = request.get_json(force=True)
    for field in ("level", "ghost_type", "time_seconds", "frames"):
        if field not in data:
            return jsonify({"error": f"{field} required"}), 400

    ghost_type = data["ghost_type"]
    if ghost_type not in ("ai", "human", "goofy", "eager", "rewind"):
        return jsonify({"error": "ghost_type must be 'ai', 'human', 'goofy', 'eager', or 'rewind'"}), 400

    level = int(data["level"])
    time_seconds = float(data["time_seconds"])
    frames = data["frames"]

    db = get_db()
    existing = db.execute(
        "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = ?",
        (level, ghost_type),
    ).fetchone()

    if existing and existing["time_seconds"] <= time_seconds:
        db.close()
        return jsonify({"updated": False, "reason": "existing ghost is faster"})

    db.execute(
        """INSERT OR REPLACE INTO ghost_replays
           (level, ghost_type, steps, time_seconds, frames_json)
           VALUES (?, ?, ?, ?, ?)""",
        (level, ghost_type, len(frames), time_seconds, json.dumps(frames)),
    )
    db.commit()
    db.close()
    return jsonify({"updated": True}), 201
