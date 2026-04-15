"""API blueprint for training metrics."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from dashboard.db import get_db

metrics_bp = Blueprint("metrics", __name__)

MAX_POINTS = 2000  # downsample to keep charts responsive


def _downsample(points: list[dict], max_points: int = MAX_POINTS) -> list[dict]:
    """Take every Nth point to cap at max_points."""
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [points[int(i * step)] for i in range(max_points)]


@metrics_bp.route("/runs/<int:run_id>/metrics")
def get_metrics(run_id: int):
    db = get_db()
    tag_filter = request.args.get("tag")

    if tag_filter:
        rows = db.execute(
            """SELECT tag, step, wall_time, value
               FROM metric_snapshots WHERE run_id = ? AND tag = ?
               ORDER BY step""",
            (run_id, tag_filter),
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT tag, step, wall_time, value
               FROM metric_snapshots WHERE run_id = ?
               ORDER BY tag, step""",
            (run_id,),
        ).fetchall()
    db.close()

    tags: dict[str, list] = {}
    for r in rows:
        tags.setdefault(r["tag"], []).append(
            {"step": r["step"], "wall_time": r["wall_time"], "value": r["value"]}
        )
    # Downsample each tag to keep responses manageable
    for tag in tags:
        tags[tag] = _downsample(tags[tag])
    return jsonify({"tags": tags})


@metrics_bp.route("/metrics/by-run-name/<run_name>")
def get_metrics_by_name(run_name: str):
    """Fetch metrics by TensorBoard run name. Syncs first to pick up latest data."""
    from dashboard.sync import sync_runs
    sync_runs()

    db = get_db()
    row = db.execute("SELECT id FROM training_runs WHERE run_name = ?", (run_name,)).fetchone()
    if not row:
        db.close()
        return jsonify({"tags": {}})
    run_id = row["id"]
    rows = db.execute(
        "SELECT tag, step, wall_time, value FROM metric_snapshots WHERE run_id = ? ORDER BY tag, step",
        (run_id,),
    ).fetchall()
    db.close()

    tags: dict[str, list] = {}
    for r in rows:
        tags.setdefault(r["tag"], []).append(
            {"step": r["step"], "wall_time": r["wall_time"], "value": r["value"]}
        )
    for tag in tags:
        tags[tag] = _downsample(tags[tag])
    return jsonify({"tags": tags})


@metrics_bp.route("/runs/<int:run_id>/metrics/tags")
def get_metric_tags(run_id: int):
    db = get_db()
    rows = db.execute(
        "SELECT DISTINCT tag FROM metric_snapshots WHERE run_id = ? ORDER BY tag",
        (run_id,),
    ).fetchall()
    db.close()
    return jsonify({"tags": [r["tag"] for r in rows]})
