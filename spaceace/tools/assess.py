#!/usr/bin/env python3
"""Agent assessment tool — runs any agent, records telemetry, analyzes behavior, flags issues.

Produces a structured JSON report consumable by humans and LLMs. The report
contains aggregate statistics, per-episode summaries, behavioral issue flags
with severity and evidence, and raw telemetry for deep-dives.

Usage:
    # Quick assessment of MCTS on level 0
    uv run python -m spaceace.tools.assess --agent mcts --level 0 --episodes 5

    # Compare agents on the same level
    uv run python -m spaceace.tools.assess --agent random --level 0 --episodes 20
    uv run python -m spaceace.tools.assess --agent mcts --level 0 --episodes 5 --num-simulations 2000

    # Assess across multiple levels
    uv run python -m spaceace.tools.assess --agent mcts --levels 0,1,2,3 --episodes 3

    # Save full report to file
    uv run python -m spaceace.tools.assess --agent mcts --level 0 --episodes 10 --output report.json

    # JSON-only (for piping to another tool or LLM)
    uv run python -m spaceace.tools.assess --agent mcts --level 0 --episodes 5 --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field

import numpy as np

import spaceace.agents  # noqa: F401 — eager imports
from spaceace.agents.base import AGENT_REGISTRY
from spaceace.agents import load_agent_module
from spaceace.strategies.actions import ACTION_NAMES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One physics step worth of telemetry."""

    step: int
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    rotation_deg: float
    action_idx: int
    action_name: str
    reward: float
    cumulative_reward: float
    pickups_remaining: int
    wall_distances: list[float]
    min_wall_dist: float


@dataclass
class EpisodeSummary:
    """Aggregate stats for one episode."""

    episode: int
    level: int
    outcome: str  # "completed", "crashed", "truncated"
    steps: int
    total_reward: float
    pickups_collected: int
    pickups_total: int
    duration_sec: float

    # Behavioral stats
    action_counts: dict[str, int]
    thrust_ratio: float
    mean_speed: float
    max_speed: float
    mean_min_wall_dist: float
    closest_wall_approach: float
    distance_traveled: float
    stuck_steps: int  # steps where position barely changed
    stuck_ratio: float
    oscillation_count: int  # rapid action switches

    # Crash-specific
    crash_speed: float | None = None
    crash_position: tuple[float, float] | None = None

    # MCTS/AlphaZero-specific (if available)
    mean_root_value: float | None = None
    mean_simulations: float | None = None


@dataclass
class Issue:
    """A flagged behavioral issue."""

    severity: str  # "critical", "warning", "info"
    category: str  # e.g. "stuck", "crash", "action_bias", "speed", "timeout"
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class AssessmentReport:
    """Full assessment output."""

    agent: str
    levels: list[int]
    episodes_per_level: int
    timestamp: str
    agent_kwargs: dict

    # Aggregate
    total_episodes: int
    completion_rate: float
    crash_rate: float
    timeout_rate: float
    mean_steps: float
    mean_reward: float
    mean_pickups: float

    # Per-level breakdown
    per_level: dict[int, dict]

    # Issues
    issues: list[dict]

    # Episode details
    episodes: list[dict]


# ---------------------------------------------------------------------------
# Telemetry collection
# ---------------------------------------------------------------------------

ALL_ACTIONS = [
    np.array([0, 0, 0], dtype=np.int32),
    np.array([0, 0, 1], dtype=np.int32),
    np.array([1, 0, 0], dtype=np.int32),
    np.array([1, 0, 1], dtype=np.int32),
    np.array([0, 1, 0], dtype=np.int32),
    np.array([0, 1, 1], dtype=np.int32),
]


def _action_to_idx(action: np.ndarray) -> int:
    for i, a in enumerate(ALL_ACTIONS):
        if np.array_equal(action, a):
            return i
    return -1


def run_episode(agent, level: int, episode_num: int, diagnose: bool = False) -> tuple[EpisodeSummary, list[StepRecord], list[dict] | None]:
    """Run one episode and collect full telemetry.

    If diagnose=True and the agent has MCTS-style debug_info, also returns
    a list of per-decision diagnostic snapshots with heuristic breakdowns,
    pathfinder state, and pickup distances.
    """
    agent.reset()

    steps: list[StepRecord] = []
    action_counter: Counter = Counter()
    speeds: list[float] = []
    min_wall_dists: list[float] = []
    positions: list[tuple[float, float]] = []
    root_values: list[float] = []
    sim_counts: list[int] = []
    cumulative_reward = 0.0
    thrust_steps = 0
    total_steps = 0
    oscillation_count = 0
    last_action_idx = -1
    prev_action_idx = -1

    raw_env = agent.get_raw_env()
    initial_obs = raw_env.get_observation()
    pickups_total = int(initial_obs[16])
    diagnostics: list[dict] = [] if diagnose else None
    prev_pickups_rem = pickups_total

    t0 = time.monotonic()

    while True:
        action, reward, terminated, truncated, info = agent.step()
        total_steps += 1
        cumulative_reward += reward

        raw_obs = raw_env.get_observation()
        x, y = float(raw_obs[0]), float(raw_obs[1])
        vx, vy = float(raw_obs[2]), float(raw_obs[3])
        speed = math.sqrt(vx ** 2 + vy ** 2)
        rot = float(raw_obs[4])
        wall_dists = [float(raw_obs[8 + i]) for i in range(8)]
        min_wd = min(wall_dists)
        pickups_rem = int(raw_obs[16])

        action_idx = _action_to_idx(action)
        action_name = ACTION_NAMES[action_idx] if action_idx >= 0 else "UNKNOWN"
        action_counter[action_name] += 1

        if int(action[2]) > 0:
            thrust_steps += 1

        speeds.append(speed)
        min_wall_dists.append(min_wd)
        positions.append((x, y))

        # Oscillation: A→B→A pattern
        if action_idx != last_action_idx and last_action_idx == prev_action_idx and total_steps > 2:
            pass  # not oscillation — just changed action
        if total_steps >= 3 and action_idx == prev_action_idx and action_idx != last_action_idx:
            oscillation_count += 1
        prev_action_idx = last_action_idx
        last_action_idx = action_idx

        # Collect MCTS/AlphaZero debug info
        if hasattr(agent, "debug_info") and agent.debug_info:
            di = agent.debug_info
            if "root_heuristic" in di:
                root_values.append(float(di["root_heuristic"]))
            if "num_simulations" in di:
                sim_counts.append(int(di["num_simulations"]))

            # Deep diagnostics: capture heuristic breakdown per decision
            if diagnose and di:
                pickup_just_collected = pickups_rem < prev_pickups_rem
                diag_entry = {
                    "step": total_steps,
                    "position": [round(x, 1), round(y, 1)],
                    "speed": round(speed, 1),
                    "min_wall_dist": round(min_wd, 1),
                    "pickups_remaining": pickups_rem,
                    "pickup_collected_this_step": pickup_just_collected,
                    "action": action_name,
                    "reward": round(reward, 4),
                }
                if "heuristic" in di:
                    diag_entry["heuristic"] = {
                        k: round(v, 2) if isinstance(v, float) and v != float("inf") else v
                        for k, v in di["heuristic"].items()
                    }
                if "target" in di:
                    diag_entry["target"] = {
                        k: round(v, 1) if isinstance(v, float) else v
                        for k, v in di["target"].items()
                    }
                if "action_stats" in di:
                    diag_entry["top_actions"] = di["action_stats"][:3]
                if "root_heuristic" in di:
                    diag_entry["root_value"] = round(float(di["root_heuristic"]), 1)
                diagnostics.append(diag_entry)

        prev_pickups_rem = pickups_rem

        rec = StepRecord(
            step=total_steps,
            x=round(x, 1),
            y=round(y, 1),
            vx=round(vx, 1),
            vy=round(vy, 1),
            speed=round(speed, 1),
            rotation_deg=round(math.degrees(rot), 1),
            action_idx=action_idx,
            action_name=action_name,
            reward=round(reward, 4),
            cumulative_reward=round(cumulative_reward, 2),
            pickups_remaining=pickups_rem,
            wall_distances=[round(d, 1) for d in wall_dists],
            min_wall_dist=round(min_wd, 1),
        )
        steps.append(rec)

        if terminated or truncated:
            break

    elapsed = time.monotonic() - t0

    # Compute derived stats
    if info.get("level_completed"):
        outcome = "completed"
    elif info.get("ship_exploded"):
        outcome = "crashed"
    else:
        outcome = "truncated"

    # Distance traveled
    dist_traveled = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dist_traveled += math.sqrt(dx ** 2 + dy ** 2)

    # Stuck detection: windows where position spread < threshold
    stuck_steps = 0
    window = 50
    spread_threshold = 20.0
    for i in range(window, len(positions)):
        recent = positions[i - window : i]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        spread = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        if spread < spread_threshold:
            stuck_steps += 1

    pickups_collected = pickups_total - int(raw_obs[16])

    summary = EpisodeSummary(
        episode=episode_num,
        level=level,
        outcome=outcome,
        steps=total_steps,
        total_reward=round(cumulative_reward, 2),
        pickups_collected=pickups_collected,
        pickups_total=pickups_total,
        duration_sec=round(elapsed, 2),
        action_counts=dict(action_counter),
        thrust_ratio=round(thrust_steps / max(total_steps, 1), 3),
        mean_speed=round(float(np.mean(speeds)), 1) if speeds else 0.0,
        max_speed=round(float(np.max(speeds)), 1) if speeds else 0.0,
        mean_min_wall_dist=round(float(np.mean(min_wall_dists)), 1) if min_wall_dists else 0.0,
        closest_wall_approach=round(float(np.min(min_wall_dists)), 1) if min_wall_dists else 0.0,
        distance_traveled=round(dist_traveled, 1),
        stuck_steps=stuck_steps,
        stuck_ratio=round(stuck_steps / max(total_steps - window, 1), 3) if total_steps > window else 0.0,
        oscillation_count=oscillation_count,
        crash_speed=round(speeds[-1], 1) if outcome == "crashed" and speeds else None,
        crash_position=(round(positions[-1][0], 1), round(positions[-1][1], 1)) if outcome == "crashed" and positions else None,
        mean_root_value=round(float(np.mean(root_values)), 3) if root_values else None,
        mean_simulations=round(float(np.mean(sim_counts)), 0) if sim_counts else None,
    )
    return summary, steps, diagnostics


# ---------------------------------------------------------------------------
# Analysis & issue detection
# ---------------------------------------------------------------------------


def analyze(summaries: list[EpisodeSummary], all_steps: list[list[StepRecord]]) -> list[Issue]:
    """Analyze episode data and flag issues."""
    issues: list[Issue] = []
    n = len(summaries)
    if n == 0:
        return issues

    outcomes = Counter(s.outcome for s in summaries)
    completion_rate = outcomes["completed"] / n
    crash_rate = outcomes["crashed"] / n
    timeout_rate = outcomes["truncated"] / n

    # --- Completion rate ---
    if completion_rate == 0:
        issues.append(Issue(
            severity="critical",
            category="completion",
            message=f"Agent never completed the level in {n} episodes",
            evidence={"outcomes": dict(outcomes)},
        ))
    elif completion_rate < 0.3:
        issues.append(Issue(
            severity="warning",
            category="completion",
            message=f"Low completion rate: {completion_rate:.0%} ({outcomes['completed']}/{n})",
            evidence={"outcomes": dict(outcomes)},
        ))

    # --- Crash rate ---
    if crash_rate > 0.5:
        crash_eps = [s for s in summaries if s.outcome == "crashed"]
        crash_speeds = [s.crash_speed for s in crash_eps if s.crash_speed is not None]
        crash_positions = [s.crash_position for s in crash_eps if s.crash_position is not None]
        issues.append(Issue(
            severity="critical" if crash_rate > 0.7 else "warning",
            category="crash",
            message=f"High crash rate: {crash_rate:.0%} ({outcomes['crashed']}/{n})",
            evidence={
                "mean_crash_speed": round(float(np.mean(crash_speeds)), 1) if crash_speeds else None,
                "crash_positions": crash_positions[:5],
                "suggestion": "Agent may be approaching walls too fast or failing to steer away",
            },
        ))

    # --- Timeout rate ---
    if timeout_rate > 0.5:
        timeout_eps = [s for s in summaries if s.outcome == "truncated"]
        avg_pickups = np.mean([s.pickups_collected for s in timeout_eps])
        issues.append(Issue(
            severity="warning",
            category="timeout",
            message=f"High timeout rate: {timeout_rate:.0%} — agent runs out of steps",
            evidence={
                "mean_pickups_at_timeout": round(float(avg_pickups), 1),
                "suggestion": "Agent may be stuck, oscillating, or too slow to reach pickups",
            },
        ))

    # --- Stuck behavior ---
    stuck_ratios = [s.stuck_ratio for s in summaries]
    mean_stuck = float(np.mean(stuck_ratios))
    if mean_stuck > 0.3:
        worst = max(summaries, key=lambda s: s.stuck_ratio)
        issues.append(Issue(
            severity="critical" if mean_stuck > 0.5 else "warning",
            category="stuck",
            message=f"Agent frequently stuck: {mean_stuck:.0%} of steps in low-movement windows",
            evidence={
                "mean_stuck_ratio": round(mean_stuck, 3),
                "worst_episode": worst.episode,
                "worst_stuck_ratio": round(worst.stuck_ratio, 3),
                "suggestion": "Agent may be orbiting, hovering, or unable to navigate past obstacles",
            },
        ))

    # --- Action bias ---
    total_actions: Counter = Counter()
    for s in summaries:
        total_actions.update(s.action_counts)
    total_action_count = sum(total_actions.values())
    if total_action_count > 0:
        most_common_name, most_common_count = total_actions.most_common(1)[0]
        dominance = most_common_count / total_action_count
        if dominance > 0.6:
            issues.append(Issue(
                severity="warning",
                category="action_bias",
                message=f"Action '{most_common_name}' dominates at {dominance:.0%} of all actions",
                evidence={
                    "distribution": {k: round(v / total_action_count, 3) for k, v in total_actions.most_common()},
                    "suggestion": "Healthy agents typically show a mix of thrust, coast, and turning",
                },
            ))
        # No thrust
        thrust_names = {"THRUST", "LEFT+THR", "RIGHT+THR"}
        thrust_total = sum(total_actions.get(n, 0) for n in thrust_names)
        thrust_frac = thrust_total / total_action_count
        if thrust_frac < 0.1:
            issues.append(Issue(
                severity="warning",
                category="action_bias",
                message=f"Very low thrust usage: {thrust_frac:.0%} — agent may be coasting/falling",
                evidence={"thrust_fraction": round(thrust_frac, 3)},
            ))

    # --- Speed issues ---
    mean_speeds = [s.mean_speed for s in summaries]
    avg_speed = float(np.mean(mean_speeds))
    if avg_speed < 10:
        issues.append(Issue(
            severity="warning",
            category="speed",
            message=f"Very low average speed ({avg_speed:.0f}) — agent may be nearly stationary",
            evidence={"mean_speed": round(avg_speed, 1)},
        ))

    max_speeds = [s.max_speed for s in summaries]
    avg_max_speed = float(np.mean(max_speeds))
    if avg_max_speed > 300:
        issues.append(Issue(
            severity="info",
            category="speed",
            message=f"High max speed ({avg_max_speed:.0f}) — may be hard to control near walls",
            evidence={"mean_max_speed": round(avg_max_speed, 1)},
        ))

    # --- Wall proximity ---
    closest_approaches = [s.closest_wall_approach for s in summaries]
    avg_closest = float(np.mean(closest_approaches))
    if avg_closest < 15 and crash_rate > 0.2:
        issues.append(Issue(
            severity="warning",
            category="wall_proximity",
            message=f"Consistently close wall approaches (avg closest: {avg_closest:.0f}px) with crashes",
            evidence={
                "mean_closest_approach": round(avg_closest, 1),
                "suggestion": "Agent may need more conservative wall avoidance",
            },
        ))

    # --- Oscillation ---
    osc_counts = [s.oscillation_count for s in summaries]
    mean_osc = float(np.mean(osc_counts))
    mean_steps = float(np.mean([s.steps for s in summaries]))
    osc_rate = mean_osc / max(mean_steps, 1)
    if osc_rate > 0.15:
        issues.append(Issue(
            severity="warning",
            category="oscillation",
            message=f"High action oscillation rate ({osc_rate:.0%}) — rapid A-B-A switching",
            evidence={
                "mean_oscillations": round(mean_osc, 1),
                "osc_per_step": round(osc_rate, 3),
                "suggestion": "Agent may be indecisive or reacting to noise",
            },
        ))

    # --- Pickup progress ---
    if completion_rate < 1.0:
        non_complete = [s for s in summaries if s.outcome != "completed"]
        pickup_fracs = [s.pickups_collected / max(s.pickups_total, 1) for s in non_complete]
        mean_frac = float(np.mean(pickup_fracs)) if pickup_fracs else 0
        if mean_frac < 0.5 and len(non_complete) >= 2:
            issues.append(Issue(
                severity="info",
                category="pickup_progress",
                message=f"Failed episodes average only {mean_frac:.0%} of pickups collected",
                evidence={
                    "mean_pickup_fraction": round(mean_frac, 3),
                    "per_episode": [
                        {"ep": s.episode, "collected": s.pickups_collected, "total": s.pickups_total}
                        for s in non_complete[:5]
                    ],
                },
            ))

    # --- Positive observations ---
    if completion_rate == 1.0:
        mean_steps_val = float(np.mean([s.steps for s in summaries]))
        issues.append(Issue(
            severity="info",
            category="completion",
            message=f"Perfect completion rate across {n} episodes (avg {mean_steps_val:.0f} steps)",
            evidence={},
        ))

    return issues


def analyze_diagnostics(
    summaries: list[EpisodeSummary],
    all_diagnostics: list[list[dict] | None],
) -> list[Issue]:
    """Analyze MCTS/AlphaZero diagnostic data to produce causal explanations."""
    issues: list[Issue] = []

    for ep_idx, (summary, diags) in enumerate(zip(summaries, all_diagnostics)):
        if not diags:
            continue

        # --- Detect hovering near unreachable pickup ---
        if summary.outcome in ("truncated", "crashed") and len(diags) > 20:
            # Look at last 30% of decisions
            tail_start = int(len(diags) * 0.7)
            tail = diags[tail_start:]
            if tail:
                path_dists = [d["heuristic"]["path_dist"] for d in tail if "heuristic" in d and "path_dist" in d.get("heuristic", {})]
                positions_tail = [d["position"] for d in tail]
                pickups_tail = [d["pickups_remaining"] for d in tail]

                if path_dists and pickups_tail:
                    # Check: constant pickups + constant path_dist = stuck near unreachable pickup
                    if len(set(pickups_tail)) == 1 and pickups_tail[0] > 0:
                        mean_pd = float(np.mean(path_dists))
                        std_pd = float(np.std(path_dists))
                        if std_pd < 10 and mean_pd < 100:
                            target = tail[-1].get("target", {})
                            issues.append(Issue(
                                severity="critical",
                                category="stuck_near_pickup",
                                message=(f"Ep {summary.episode}: Agent orbiting {mean_pd:.0f}px from pickup "
                                         f"for last {len(tail)} decisions without collecting"),
                                evidence={
                                    "episode": summary.episode,
                                    "pickups_remaining": pickups_tail[0],
                                    "mean_path_dist": round(mean_pd, 1),
                                    "path_dist_std": round(std_pd, 1),
                                    "target_pickup": target,
                                    "agent_position": positions_tail[-1],
                                    "cause": "Pickup may be inside pathfinder wall inflation zone, "
                                             "or heuristic value of collecting is lower than staying nearby "
                                             "(next pickup is much farther away, reducing proximity/velocity scores)",
                                    "suggestion": "Check if pickup is near a wall. If path_dist is stable ~35-60px, "
                                                  "the pathfinder grid can't route closer. The euclidean fallback "
                                                  "should handle this — if not, the collection radius may need tuning.",
                                },
                            ))

        # --- Detect crash cause from heuristic ---
        if summary.outcome == "crashed" and len(diags) >= 5:
            last_5 = diags[-5:]
            ttis = [d["heuristic"].get("min_tti", float("inf")) for d in last_5 if "heuristic" in d]
            tti_pens = [d["heuristic"].get("tti_penalty", 0) for d in last_5 if "heuristic" in d]
            speeds_diag = [d["speed"] for d in last_5]
            walls_diag = [d["min_wall_dist"] for d in last_5]
            root_vals = [d.get("root_value", 0) for d in last_5]

            if ttis:
                min_tti = min(t for t in ttis if t != float("inf")) if any(t != float("inf") for t in ttis) else None
                cause_parts = []
                if min_tti is not None and min_tti < 0.3:
                    cause_parts.append(f"TTI dropped to {min_tti:.2f}s (imminent collision)")
                if speeds_diag and max(speeds_diag) > 250:
                    cause_parts.append(f"High speed ({max(speeds_diag):.0f}) made avoidance difficult")
                if walls_diag and min(walls_diag) < 30:
                    cause_parts.append(f"Wall distance bottomed at {min(walls_diag):.0f}px")
                if root_vals and all(v < 0 for v in root_vals):
                    cause_parts.append("Heuristic was negative — agent knew crash was likely but had no escape")

                pickup_score = last_5[-1].get("heuristic", {}).get("pickups_score", 0)
                tti_pen = min(tti_pens) if tti_pens else 0
                if pickup_score > 0 and abs(tti_pen) < pickup_score * 0.5:
                    cause_parts.append(
                        f"TTI penalty ({tti_pen:.0f}) was small relative to pickup reward ({pickup_score:.0f}) — "
                        f"agent may have prioritized progress over safety"
                    )

                if cause_parts:
                    issues.append(Issue(
                        severity="warning",
                        category="crash_cause",
                        message=f"Ep {summary.episode} crash analysis: {cause_parts[0]}",
                        evidence={
                            "episode": summary.episode,
                            "causes": cause_parts,
                            "final_state": {
                                "speed": last_5[-1]["speed"],
                                "min_wall_dist": last_5[-1]["min_wall_dist"],
                                "position": last_5[-1]["position"],
                                "action": last_5[-1]["action"],
                            },
                        },
                    ))

        # --- Detect reward disincentive for collection ---
        if diags and summary.outcome != "completed":
            # Find decisions right after a pickup was collected
            for i, d in enumerate(diags):
                if d.get("pickup_collected_this_step") and i > 0 and i < len(diags) - 1:
                    before_h = diags[i - 1].get("root_value")
                    after_h = diags[i + 1].get("root_value") if i + 1 < len(diags) else None
                    if before_h is not None and after_h is not None:
                        drop = before_h - after_h
                        if drop > 100:
                            issues.append(Issue(
                                severity="warning",
                                category="collection_disincentive",
                                message=(f"Ep {summary.episode}: Heuristic dropped {drop:.0f} after pickup "
                                         f"at step {d['step']} (from {before_h:.0f} to {after_h:.0f})"),
                                evidence={
                                    "episode": summary.episode,
                                    "step": d["step"],
                                    "heuristic_before": round(before_h, 1),
                                    "heuristic_after": round(after_h, 1),
                                    "drop": round(drop, 1),
                                    "cause": "Next pickup is much farther away, causing proximity/velocity "
                                             "scores to collapse. If the drop exceeds the pickup collection "
                                             "bonus, the agent is incentivized NOT to collect.",
                                },
                            ))

    return issues


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(
    agent_name: str,
    levels: list[int],
    episodes_per_level: int,
    agent_kwargs: dict,
    all_summaries: list[EpisodeSummary],
    all_steps: list[list[StepRecord]],
    all_issues: list[Issue],
) -> dict:
    """Build the final JSON-serializable report."""
    from datetime import datetime

    n = len(all_summaries)
    outcomes = Counter(s.outcome for s in all_summaries)

    # Per-level breakdown
    per_level: dict[int, dict] = {}
    for lvl in levels:
        lvl_summaries = [s for s in all_summaries if s.level == lvl]
        if not lvl_summaries:
            continue
        lvl_outcomes = Counter(s.outcome for s in lvl_summaries)
        ln = len(lvl_summaries)
        per_level[lvl] = {
            "episodes": ln,
            "completion_rate": round(lvl_outcomes["completed"] / ln, 3),
            "crash_rate": round(lvl_outcomes["crashed"] / ln, 3),
            "timeout_rate": round(lvl_outcomes["truncated"] / ln, 3),
            "mean_steps": round(float(np.mean([s.steps for s in lvl_summaries])), 1),
            "mean_reward": round(float(np.mean([s.total_reward for s in lvl_summaries])), 2),
            "mean_pickups": round(float(np.mean([s.pickups_collected for s in lvl_summaries])), 1),
            "mean_speed": round(float(np.mean([s.mean_speed for s in lvl_summaries])), 1),
            "mean_thrust_ratio": round(float(np.mean([s.thrust_ratio for s in lvl_summaries])), 3),
        }

    report = {
        "agent": agent_name,
        "levels": levels,
        "episodes_per_level": episodes_per_level,
        "timestamp": datetime.now().isoformat(),
        "agent_kwargs": {k: v for k, v in agent_kwargs.items() if v is not None},
        "summary": {
            "total_episodes": n,
            "completion_rate": round(outcomes["completed"] / max(n, 1), 3),
            "crash_rate": round(outcomes["crashed"] / max(n, 1), 3),
            "timeout_rate": round(outcomes["truncated"] / max(n, 1), 3),
            "mean_steps": round(float(np.mean([s.steps for s in all_summaries])), 1),
            "std_steps": round(float(np.std([s.steps for s in all_summaries])), 1),
            "mean_reward": round(float(np.mean([s.total_reward for s in all_summaries])), 2),
            "mean_pickups": round(float(np.mean([s.pickups_collected for s in all_summaries])), 1),
            "mean_speed": round(float(np.mean([s.mean_speed for s in all_summaries])), 1),
            "mean_thrust_ratio": round(float(np.mean([s.thrust_ratio for s in all_summaries])), 3),
        },
        "per_level": per_level,
        "issues": [asdict(i) for i in all_issues],
        "episodes": [asdict(s) for s in all_summaries],
    }
    return report


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


_SEVERITY_MARKERS = {"critical": "X", "warning": "!", "info": "-"}


def print_report(report: dict) -> None:
    """Print a concise human-readable summary."""
    s = report["summary"]
    print()
    print(f"=== Assessment: {report['agent']} on level(s) {report['levels']} ===")
    print(f"Episodes: {s['total_episodes']}  |  "
          f"Completed: {s['completion_rate']:.0%}  |  "
          f"Crashed: {s['crash_rate']:.0%}  |  "
          f"Timeout: {s['timeout_rate']:.0%}")
    print(f"Steps: {s['mean_steps']:.0f} avg (+/- {s['std_steps']:.0f})  |  "
          f"Reward: {s['mean_reward']:.1f} avg  |  "
          f"Pickups: {s['mean_pickups']:.1f} avg")
    print(f"Speed: {s['mean_speed']:.0f} avg  |  "
          f"Thrust: {s['mean_thrust_ratio']:.0%}")

    if len(report["per_level"]) > 1:
        print()
        print("Per-level breakdown:")
        for lvl, d in sorted(report["per_level"].items(), key=lambda x: int(x[0])):
            print(f"  Level {lvl}: "
                  f"complete={d['completion_rate']:.0%} "
                  f"crash={d['crash_rate']:.0%} "
                  f"steps={d['mean_steps']:.0f} "
                  f"reward={d['mean_reward']:.1f} "
                  f"pickups={d['mean_pickups']:.1f}")

    issues = report["issues"]
    if issues:
        print()
        print("Issues:")
        for i in sorted(issues, key=lambda x: {"critical": 0, "warning": 1, "info": 2}[x["severity"]]):
            marker = _SEVERITY_MARKERS.get(i["severity"], "?")
            print(f"  [{marker}] {i['severity'].upper()}: {i['message']}")
            ev = i.get("evidence", {})
            if "suggestion" in ev:
                print(f"      -> {ev['suggestion']}")
    else:
        print()
        print("No issues detected.")

    print()
    # Episode table
    print("Episodes:")
    print(f"  {'#':>3}  {'Lvl':>3}  {'Outcome':<10}  {'Steps':>6}  {'Reward':>8}  {'Pickups':>8}  {'Thrust':>7}  {'AvgSpd':>6}  {'Stuck%':>6}")
    print(f"  {'---':>3}  {'---':>3}  {'-------':<10}  {'-----':>6}  {'------':>8}  {'-------':>8}  {'------':>7}  {'------':>6}  {'------':>6}")
    for ep in report["episodes"]:
        pstr = f"{ep['pickups_collected']}/{ep['pickups_total']}"
        print(f"  {ep['episode']:>3}  {ep['level']:>3}  {ep['outcome']:<10}  {ep['steps']:>6}  "
              f"{ep['total_reward']:>8.1f}  {pstr:>8}  {ep['thrust_ratio']:>6.0%}  "
              f"{ep['mean_speed']:>6.0f}  {ep['stuck_ratio']:>5.0%}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assess any SpaceAce agent: run episodes, analyze behavior, flag issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--agent", type=str, required=True,
                   help=f"Agent type. Available: {', '.join(sorted(AGENT_REGISTRY))}")
    p.add_argument("--agent-module", type=str, default=None,
                   help="Dotted path to a module that registers a custom agent")
    p.add_argument("--level", type=int, default=None, help="Single level to assess")
    p.add_argument("--levels", type=str, default=None,
                   help="Comma-separated levels (e.g. '0,1,2,3')")
    p.add_argument("--episodes", type=int, default=5, help="Episodes per level (default 5)")
    p.add_argument("--max-steps", type=int, default=3000)

    # Agent-specific
    p.add_argument("--num-simulations", type=int, default=200)
    p.add_argument("--exploration", type=float, default=1.41)
    p.add_argument("--momentum-pathfinder", action="store_true")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--action-repeat", type=int, default=None)

    # Output
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Save full JSON report to this file")
    p.add_argument("--json", action="store_true",
                   help="Output JSON only (no human-readable text)")
    p.add_argument("--diagnose", action="store_true",
                   help="Deep introspection: capture MCTS heuristic breakdowns, "
                        "pathfinder state, and generate causal explanations for behavior")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.agent_module:
        load_agent_module(args.agent_module)

    if args.agent not in AGENT_REGISTRY:
        print(f"Unknown agent '{args.agent}'. Available: {', '.join(sorted(AGENT_REGISTRY))}", file=sys.stderr)
        sys.exit(1)

    # Resolve levels
    if args.levels:
        levels = [int(x.strip()) for x in args.levels.split(",")]
    elif args.level is not None:
        levels = [args.level]
    else:
        levels = [0]

    agent_kwargs = {
        "num_simulations": args.num_simulations,
        "exploration_constant": args.exploration,
        "momentum_pathfinder": args.momentum_pathfinder,
    }
    if args.model:
        agent_kwargs["model_path"] = args.model
    if args.action_repeat is not None:
        agent_kwargs["action_repeat"] = args.action_repeat

    all_summaries: list[EpisodeSummary] = []
    all_steps: list[list[StepRecord]] = []
    all_diagnostics: list[list[dict] | None] = []
    episode_counter = 0
    do_diagnose = getattr(args, "diagnose", False)

    if not args.json:
        mode = " (with diagnostics)" if do_diagnose else ""
        print(f"Assessing '{args.agent}' on level(s) {levels}, {args.episodes} episodes each{mode}...")

    for level in levels:
        agent = AGENT_REGISTRY[args.agent]()
        agent.setup(level=level, max_steps=args.max_steps, **agent_kwargs)

        for ep in range(args.episodes):
            if not args.json:
                print(f"  Level {level}, episode {ep + 1}/{args.episodes}...", end="", flush=True)
            summary, steps, diagnostics = run_episode(agent, level, episode_counter, diagnose=do_diagnose)
            if not args.json:
                print(f" {summary.outcome} ({summary.steps} steps, {summary.pickups_collected}/{summary.pickups_total} pickups)")
            all_summaries.append(summary)
            all_steps.append(steps)
            all_diagnostics.append(diagnostics)
            episode_counter += 1

        agent.close()

    # Analyze
    all_issues = analyze(all_summaries, all_steps)

    # Deep diagnostic analysis (MCTS heuristic breakdowns)
    if do_diagnose:
        diag_issues = analyze_diagnostics(all_summaries, all_diagnostics)
        all_issues.extend(diag_issues)

    # Build report
    report = build_report(
        agent_name=args.agent,
        levels=levels,
        episodes_per_level=args.episodes,
        agent_kwargs=agent_kwargs,
        all_summaries=all_summaries,
        all_steps=all_steps,
        all_issues=all_issues,
    )

    # Attach diagnostic snapshots to failed episodes for deep-dive
    if do_diagnose:
        for ep_report, diags in zip(report["episodes"], all_diagnostics):
            if diags and ep_report["outcome"] != "completed":
                # Include last 20 decisions for failed episodes
                ep_report["diagnostic_snapshots"] = diags[-20:]

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        if not args.json:
            print(f"Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
