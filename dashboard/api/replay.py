"""API blueprint for game replays."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

replay_bp = Blueprint("replay", __name__)


@replay_bp.route("/replay", methods=["POST"])
def create_replay():
    """Run an agent on a level and return the full replay.

    Body: { agent_type, level, max_steps?, action_repeat?, model_path?, num_simulations? }
    """
    data = request.get_json(force=True)
    agent_type = data.get("agent_type")
    level = data.get("level", 0)

    if not agent_type:
        return jsonify({"error": "agent_type required"}), 400

    try:
        from dashboard.replay import capture_replay

        replay = capture_replay(
            agent_type=agent_type,
            level=int(level),
            max_steps=int(data.get("max_steps", 3000)),
            action_repeat=int(data.get("action_repeat", 5)),
            model_path=data.get("model_path"),
            num_simulations=int(data["num_simulations"]) if data.get("num_simulations") else None,
        )

        # Auto-submit to leaderboard and save AI ghost
        try:
            import json as _json
            from dashboard.db import get_db
            pickups_total = len(replay.get("pickups_initial", []))
            last_frame = replay["frames"][-1] if replay["frames"] else {}
            pickups_collected = pickups_total - last_frame.get("pickups_remaining", pickups_total)
            db = get_db()
            db.execute(
                """INSERT INTO leaderboard (level, agent_type, model_path, outcome, steps,
                                            pickups_collected, pickups_total)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (int(level), agent_type, data.get("model_path"),
                 replay["outcome"], replay["total_steps"],
                 pickups_collected, pickups_total),
            )

            # Save AI ghost if this is a completed run faster than existing
            if replay["outcome"] == "completed":
                time_seconds = replay["total_steps"] / 60.0
                existing = db.execute(
                    "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = 'ai'",
                    (int(level),),
                ).fetchone()
                if existing is None or time_seconds < existing["time_seconds"]:
                    # Downsample 60fps replay to ~10fps for compact ghost data
                    ghost_frames = []
                    for i, f in enumerate(replay["frames"]):
                        if i % 6 == 0 or i == len(replay["frames"]) - 1:
                            ghost_frames.append({
                                "x": f["x"], "y": f["y"],
                                "rotation": f["rotation"],
                                "thrusting": f["action"][2] > 0,
                                "time": round(i / 60.0, 3),
                            })
                    db.execute(
                        """INSERT OR REPLACE INTO ghost_replays
                           (level, ghost_type, steps, time_seconds, frames_json)
                           VALUES (?, 'ai', ?, ?, ?)""",
                        (int(level), len(ghost_frames), time_seconds,
                         _json.dumps(ghost_frames)),
                    )

            db.commit()
            db.close()
        except Exception:
            pass  # don't fail the replay if leaderboard/ghost write fails

        return jsonify({"replay": replay})
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500
