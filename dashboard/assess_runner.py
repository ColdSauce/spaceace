"""Background assessment runner — wraps spaceace.tools.assess."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone

from dashboard.db import get_db

_lock = threading.Lock()


def run_assessment(assessment_id: int) -> None:
    """Run an assessment in the current thread. Updates the DB row as it goes."""
    db = get_db()
    row = db.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
    if row is None:
        return

    db.execute(
        "UPDATE assessments SET status='running', started_at=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), assessment_id),
    )
    db.commit()

    try:
        import spaceace.agents  # noqa: F401 — trigger registration
        from spaceace.agents.base import AGENT_REGISTRY
        from spaceace.tools.assess import run_episode, analyze, build_report

        agent_type = row["agent_type"]
        levels = json.loads(row["levels"])
        episodes_per_level = row["episodes_per_level"]
        model_path = row["model_path"]

        agent_cls = AGENT_REGISTRY[agent_type]
        agent = agent_cls()

        # Build setup kwargs from the assessment row
        agent_kwargs: dict = {}
        if model_path:
            agent_kwargs["model_path"] = model_path

        all_summaries = []
        all_steps = []

        for level in levels:
            setup_kw = dict(agent_kwargs)
            agent.setup(level=level, max_steps=3000, **setup_kw)

            for ep in range(episodes_per_level):
                summary, steps, _ = run_episode(agent, level, ep)
                all_summaries.append(summary)
                all_steps.append(steps)

        issues = analyze(all_summaries, all_steps)
        report = build_report(
            agent_name=agent_type,
            levels=levels,
            episodes_per_level=episodes_per_level,
            agent_kwargs=agent_kwargs,
            all_summaries=all_summaries,
            all_steps=all_steps,
            all_issues=issues,
        )

        summary = report["summary"]
        db.execute(
            """UPDATE assessments
               SET status='completed', completed_at=?,
                   completion_rate=?, crash_rate=?, timeout_rate=?,
                   mean_reward=?, mean_steps=?, report_json=?
               WHERE id=?""",
            (
                datetime.now(timezone.utc).isoformat(),
                summary["completion_rate"],
                summary["crash_rate"],
                summary["timeout_rate"],
                summary["mean_reward"],
                summary["mean_steps"],
                json.dumps(report),
                assessment_id,
            ),
        )
        db.commit()

    except Exception:
        tb = traceback.format_exc()
        db.execute(
            "UPDATE assessments SET status='failed', completed_at=?, error_message=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), tb, assessment_id),
        )
        db.commit()
    finally:
        db.close()


def start_assessment(assessment_id: int) -> bool:
    """Launch assessment in a background thread. Returns False if one is already running."""
    if not _lock.acquire(blocking=False):
        return False

    def _run():
        try:
            run_assessment(assessment_id)
        finally:
            _lock.release()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True
