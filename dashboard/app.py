"""Flask application factory."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from flask import Flask, send_from_directory

STATIC_DIR = Path(__file__).parent / "static"
WEB_DIR = Path(__file__).parent.parent / "web"
DATA_DIR = Path(__file__).parent.parent / "data"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    from dashboard.db import init_db
    from dashboard.sync import sync_all

    with app.app_context():
        init_db()

        from dashboard.job_runner import reap_stale_jobs
        reaped = reap_stale_jobs()
        if reaped:
            print(f"Reaped {reaped} stale job(s)")

    # Run the (potentially slow) tensorboard/model sync in a background thread
    # so the server starts listening immediately. Set DASHBOARD_SKIP_SYNC=1 to
    # disable entirely (useful when tensorboard_logs/ is huge).
    if os.environ.get("DASHBOARD_SKIP_SYNC") != "1":
        def _bg_sync():
            try:
                with app.app_context():
                    result = sync_all()
                    print(f"[sync] {result['synced_runs']} runs, {result['synced_models']} models")
            except Exception as e:
                print(f"[sync] failed: {e}")
        threading.Thread(target=_bg_sync, daemon=True).start()

    from dashboard.api.runs import runs_bp
    from dashboard.api.metrics import metrics_bp
    from dashboard.api.assessments import assessments_bp
    from dashboard.api.models import models_bp
    from dashboard.api.jobs import jobs_bp
    from dashboard.api.replay import replay_bp
    from dashboard.api.leaderboard import leaderboard_bp
    from dashboard.api.ghosts import ghosts_bp

    app.register_blueprint(runs_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api")
    app.register_blueprint(assessments_bp, url_prefix="/api")
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(jobs_bp, url_prefix="/api")
    app.register_blueprint(replay_bp, url_prefix="/api")
    app.register_blueprint(leaderboard_bp, url_prefix="/api")
    app.register_blueprint(ghosts_bp, url_prefix="/api")

    @app.route("/play")
    def play_game():
        return send_from_directory(str(WEB_DIR), "index.html")

    @app.route("/mapData.js")
    def map_data():
        return send_from_directory(str(DATA_DIR), "mapData.js")

    @app.route("/")
    @app.route("/<path:path>")
    def spa(path=""):
        return send_from_directory(str(STATIC_DIR), "index.html")

    return app
