"""API blueprint for per-level leaderboard."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from dashboard.db import get_db

leaderboard_bp = Blueprint("leaderboard", __name__)


@leaderboard_bp.route("/leaderboard")
def get_leaderboard():
    """Return leaderboard entries, optionally filtered by level.

    Query params: ?level=N (optional)
    """
    db = get_db()
    level = request.args.get("level")
    if level is not None:
        rows = db.execute(
            """SELECT * FROM leaderboard WHERE level = ?
               ORDER BY steps ASC LIMIT 50""",
            (int(level),),
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT * FROM leaderboard ORDER BY level ASC, steps ASC""",
        ).fetchall()
    db.close()
    return jsonify({"entries": [dict(r) for r in rows]})


@leaderboard_bp.route("/leaderboard", methods=["POST"])
def add_entry():
    """Manually submit a leaderboard entry."""
    data = request.get_json(force=True)
    required = ("level", "agent_type", "outcome", "steps", "pickups_collected", "pickups_total")
    for field in required:
        if field not in data:
            return jsonify({"error": f"{field} required"}), 400

    db = get_db()
    cur = db.execute(
        """INSERT INTO leaderboard (level, agent_type, model_path, outcome, steps,
                                    pickups_collected, pickups_total)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            int(data["level"]),
            data["agent_type"],
            data.get("model_path"),
            data["outcome"],
            int(data["steps"]),
            int(data["pickups_collected"]),
            int(data["pickups_total"]),
        ),
    )
    db.commit()
    entry_id = cur.lastrowid
    db.close()
    return jsonify({"id": entry_id}), 201


@leaderboard_bp.route("/leaderboard/<int:eid>", methods=["DELETE"])
def delete_entry(eid: int):
    db = get_db()
    db.execute("DELETE FROM leaderboard WHERE id = ?", (eid,))
    db.commit()
    db.close()
    return jsonify({"ok": True})
