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
        return jsonify({"replay": replay})
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500
