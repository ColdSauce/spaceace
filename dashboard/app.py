"""Flask application factory."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, send_from_directory

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    from dashboard.db import init_db
    from dashboard.sync import sync_all

    with app.app_context():
        init_db()
        result = sync_all()
        print(f"Synced {result['synced_runs']} runs, {result['synced_models']} models")

        from dashboard.job_runner import reap_stale_jobs
        reaped = reap_stale_jobs()
        if reaped:
            print(f"Reaped {reaped} stale job(s)")

    from dashboard.api.runs import runs_bp
    from dashboard.api.metrics import metrics_bp
    from dashboard.api.assessments import assessments_bp
    from dashboard.api.models import models_bp
    from dashboard.api.jobs import jobs_bp
    from dashboard.api.replay import replay_bp

    app.register_blueprint(runs_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api")
    app.register_blueprint(assessments_bp, url_prefix="/api")
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(jobs_bp, url_prefix="/api")
    app.register_blueprint(replay_bp, url_prefix="/api")

    @app.route("/")
    @app.route("/<path:path>")
    def spa(path=""):
        return send_from_directory(str(STATIC_DIR), "index.html")

    return app
