"""API blueprint for training runs."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from dashboard.db import get_db
from dashboard.sync import sync_all

runs_bp = Blueprint("runs", __name__)


def _attach_final_metrics(runs: list[dict], db) -> list[dict]:
    """For each run, fetch the last value of key metrics for inline display."""
    if not runs:
        return runs
    ids = [r["id"] for r in runs]
    placeholders = ",".join("?" * len(ids))
    rows = db.execute(
        f"""SELECT run_id, tag, value FROM metric_snapshots
            WHERE run_id IN ({placeholders})
              AND tag IN ('rollout/ep_rew_mean', 'episode/completed', 'episode/crashed')
              AND step = (
                  SELECT MAX(m2.step) FROM metric_snapshots m2
                  WHERE m2.run_id = metric_snapshots.run_id AND m2.tag = metric_snapshots.tag
              )""",
        ids,
    ).fetchall()

    lookup: dict[int, dict] = {}
    for row in rows:
        lookup.setdefault(row["run_id"], {})[row["tag"]] = row["value"]

    for r in runs:
        m = lookup.get(r["id"], {})
        r["final_reward"] = round(m.get("rollout/ep_rew_mean", 0), 1) if "rollout/ep_rew_mean" in m else None
        r["final_completion"] = round(m.get("episode/completed", 0), 3) if "episode/completed" in m else None
        r["final_crash"] = round(m.get("episode/crashed", 0), 3) if "episode/crashed" in m else None
    return runs


@runs_bp.route("/runs")
def list_runs():
    db = get_db()
    rows = db.execute(
        """SELECT id, run_name, agent_type, levels, action_repeat,
                  total_timesteps, obs_strategy, reward_strategy,
                  status, created_at, synced_at, notes
           FROM training_runs ORDER BY created_at DESC"""
    ).fetchall()
    runs = [dict(r) for r in rows]
    runs = _attach_final_metrics(runs, db)
    db.close()
    return jsonify({"runs": runs})


@runs_bp.route("/runs/<int:run_id>")
def get_run(run_id: int):
    db = get_db()
    row = db.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    db.close()
    if row is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"run": dict(row)})


@runs_bp.route("/runs/<int:run_id>/notes", methods=["PUT"])
def update_notes(run_id: int):
    data = request.get_json(force=True)
    notes = data.get("notes", "")
    db = get_db()
    db.execute("UPDATE training_runs SET notes = ? WHERE id = ?", (notes, run_id))
    db.commit()
    db.close()
    return jsonify({"ok": True})


@runs_bp.route("/runs/sync", methods=["POST"])
def trigger_sync():
    result = sync_all()
    return jsonify(result)
