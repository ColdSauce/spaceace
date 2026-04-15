"""Beam search solver for SpaceAce — finds short action sequences that complete a level."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS
from spaceace.strategies.pathfinder import RustPathfinder

NUM_ACTIONS = len(ALL_ACTIONS)


@dataclass
class LiveBeam:
    """A live beam entry with state snapshot for expansion."""

    state: object  # PyGameState snapshot
    pickups_remaining: int
    pickup_states: list[bool]
    obs: np.ndarray  # raw 20-dim observation
    physics_steps: int  # total physics steps taken so far


@dataclass
class TraceEntry:
    """Lightweight parent pointer for trajectory reconstruction."""

    parent_idx: int
    action_idx: int  # action index (0-5), held for action_repeat frames


class BeamSearchSolver:
    """Phase 1: beam search to find a complete trajectory.

    Uses stochastic beam selection (softmax sampling instead of hard top-K)
    to maintain diversity. Each expansion holds one action for `action_repeat`
    physics frames.
    """

    PICKUP_WEIGHT = 10000.0
    SPEED_CAP = 150.0      # speed above this is penalized
    SPEED_PENALTY = 0.01   # quadratic penalty coefficient for excess speed

    def __init__(
        self,
        env: SpaceAceDirectEnv,
        level: int,
        beam_width: int = 1000,
        max_steps: int = 3000,
        step_penalty: float = 0.01,
        action_repeat: int = 3,
        temperature: float = 5.0,
    ) -> None:
        self._env = env
        self._level = level
        self._beam_width = beam_width
        self._max_steps = max_steps
        self._step_penalty = step_penalty
        self._action_repeat = action_repeat
        self._temperature = temperature
        self._pathfinder = RustPathfinder(level, backend="grid")

    def _score(
        self, obs: np.ndarray, pickup_states: list[bool],
        pickups_remaining: int, physics_steps: int,
    ) -> float:
        self._pathfinder.clear_cache()
        if pickups_remaining > 0:
            path_dist, dir_x, dir_y = self._pathfinder.nearest_pickup_info(
                float(obs[0]), float(obs[1]), pickup_states,
            )
        else:
            path_dist = 0.0

        speed = float(math.sqrt(obs[2]**2 + obs[3]**2))
        speed_penalty = 0.0
        if speed > self.SPEED_CAP:
            excess = speed - self.SPEED_CAP
            speed_penalty = excess * excess * self.SPEED_PENALTY

        return (
            -pickups_remaining * self.PICKUP_WEIGHT
            - path_dist
            - speed_penalty
            - physics_steps * self._step_penalty
        )

    def _stochastic_select(
        self, scores: np.ndarray, k: int,
    ) -> np.ndarray:
        """Select k indices using softmax sampling (without replacement).

        Top half of the beam is kept greedily (exploitation).
        Bottom half is sampled proportional to softmax(scores) (exploration).
        """
        n = len(scores)
        if n <= k:
            return np.arange(n)

        # Keep top half greedily
        greedy_k = k // 2
        sorted_idx = np.argsort(scores)[::-1]
        greedy = sorted_idx[:greedy_k]

        # Sample remaining from the rest
        remaining_idx = sorted_idx[greedy_k:]
        remaining_scores = scores[remaining_idx]

        # Softmax for sampling probabilities
        shifted = remaining_scores - remaining_scores.max()
        exp_scores = np.exp(shifted / max(self._temperature, 0.1))
        probs = exp_scores / exp_scores.sum()

        sample_k = min(k - greedy_k, len(remaining_idx))
        # Ensure enough non-zero probabilities for sampling
        nonzero_count = np.count_nonzero(probs)
        sample_k = min(sample_k, nonzero_count)
        if sample_k == 0:
            return greedy
        sampled = np.random.choice(
            remaining_idx, size=sample_k, replace=False, p=probs,
        )

        return np.concatenate([greedy, sampled])

    def _expand_action(
        self, env: SpaceAceDirectEnv, parent_state: object,
        action_idx: int, action_repeat: int,
    ) -> tuple[np.ndarray | None, list[bool] | None, int, bool, bool]:
        """Execute one action for action_repeat physics frames."""
        env.load_state(parent_state)
        action = ALL_ACTIONS[action_idx]

        for _ in range(action_repeat):
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("ship_exploded", False):
                return None, None, 0, False, True
            if info.get("level_completed", False):
                pickup_states = list(env.get_pickup_states())
                return obs, pickup_states, 0, True, False
            if terminated or truncated:
                return None, None, 0, False, True

        pickup_states = list(env.get_pickup_states())
        pickups_remaining = sum(1 for p in pickup_states if not p)
        return obs, pickup_states, pickups_remaining, False, False

    def solve(self) -> list[int]:
        """Run beam search. Returns list of action indices (per-frame)."""
        env = self._env
        env.reset()

        initial_state = env.save_state()
        initial_obs = env.get_observation()
        initial_pickups = list(env.get_pickup_states())
        total_pickups = sum(1 for p in initial_pickups if not p)

        current_beam: list[LiveBeam] = [
            LiveBeam(
                state=initial_state,
                pickups_remaining=total_pickups,
                pickup_states=initial_pickups,
                obs=initial_obs,
                physics_steps=0,
            )
        ]

        trace_history: list[list[TraceEntry]] = []
        solutions: list[tuple[int, int, int]] = []  # (beam_step, trace_idx, physics_steps)

        t_start = time.time()
        best_pickups_remaining = total_pickups
        max_beam_steps = self._max_steps // self._action_repeat + 1

        for beam_step in range(1, max_beam_steps + 1):
            if not current_beam:
                print(f"  Beam step {beam_step}: beam empty, stopping")
                break

            cand_states: list[object] = []
            cand_obs: list[np.ndarray] = []
            cand_pickups: list[list[bool]] = []
            cand_remaining: list[int] = []
            cand_traces: list[TraceEntry] = []
            cand_scores_list: list[float] = []
            cand_physics_steps: list[int] = []
            completed_traces: list[tuple[TraceEntry, int]] = []

            for beam_idx, beam in enumerate(current_beam):
                for action_idx in range(NUM_ACTIONS):
                    obs, pstates, remaining, completed, crashed = \
                        self._expand_action(
                            env, beam.state, action_idx, self._action_repeat,
                        )

                    if crashed:
                        continue

                    new_physics = beam.physics_steps + self._action_repeat
                    trace = TraceEntry(parent_idx=beam_idx, action_idx=action_idx)

                    if completed:
                        completed_traces.append((trace, new_physics))
                        continue

                    score = self._score(obs, pstates, remaining, new_physics)
                    cand_states.append(env.save_state())
                    cand_obs.append(obs)
                    cand_pickups.append(pstates)
                    cand_remaining.append(remaining)
                    cand_traces.append(trace)
                    cand_scores_list.append(score)
                    cand_physics_steps.append(new_physics)

            # Stochastic beam selection
            if cand_scores_list:
                scores_arr = np.array(cand_scores_list)
                selected = self._stochastic_select(scores_arr, self._beam_width)
            else:
                selected = np.array([], dtype=int)

            step_traces: list[TraceEntry] = []
            new_beam: list[LiveBeam] = []
            for idx in selected:
                step_traces.append(cand_traces[idx])
                new_beam.append(LiveBeam(
                    state=cand_states[idx],
                    pickups_remaining=cand_remaining[idx],
                    pickup_states=cand_pickups[idx],
                    obs=cand_obs[idx],
                    physics_steps=cand_physics_steps[idx],
                ))

            for trace, phys_steps in completed_traces:
                sol_trace_idx = len(step_traces)
                step_traces.append(trace)
                solutions.append((beam_step, sol_trace_idx, phys_steps))

            trace_history.append(step_traces)
            current_beam = new_beam

            # Track progress
            if new_beam:
                min_remaining = min(b.pickups_remaining for b in new_beam)
                if min_remaining < best_pickups_remaining:
                    best_pickups_remaining = min_remaining
                    elapsed = time.time() - t_start
                    phys = new_beam[0].physics_steps
                    print(
                        f"  Step {beam_step} (frame {phys}): pickup collected! "
                        f"remaining={min_remaining}/{total_pickups} "
                        f"elapsed={elapsed:.1f}s"
                    )

            if beam_step % 100 == 0 or completed_traces:
                elapsed = time.time() - t_start
                best_score = max(cand_scores_list) if cand_scores_list else float("-inf")
                leader_info = ""
                if new_beam:
                    ldr = new_beam[0]
                    ldr_speed = float(math.sqrt(ldr.obs[2]**2 + ldr.obs[3]**2))
                    leader_info = (
                        f" leader=({ldr.obs[0]:.0f},{ldr.obs[1]:.0f}) "
                        f"speed={ldr_speed:.1f} "
                        f"frame={ldr.physics_steps}"
                    )
                print(
                    f"  Step {beam_step}: beam={len(new_beam)} "
                    f"best_score={best_score:.1f} "
                    f"remaining={best_pickups_remaining}/{total_pickups} "
                    f"solutions={len(solutions)} "
                    f"elapsed={elapsed:.1f}s{leader_info}"
                )

            if not cand_scores_list and not completed_traces:
                print(f"  Step {beam_step}: no viable candidates, stopping")
                break
            if solutions and not cand_scores_list:
                break

            # Early termination: stop if we have solutions and
            # the best remaining beam can't possibly beat the best solution
            if solutions:
                best_sol_frames = min(s[2] for s in solutions)
                if new_beam:
                    # Current beams are already at more physics steps than best solution
                    min_beam_frames = min(b.physics_steps for b in new_beam)
                    if min_beam_frames >= best_sol_frames:
                        print(f"  Step {beam_step}: all beams past best solution "
                              f"({best_sol_frames} frames), stopping")
                        break

        if not solutions:
            print("WARNING: No solution found! Returning empty trajectory.")
            return []

        best_beam_step, best_trace_idx, best_phys = min(
            solutions, key=lambda x: x[2],
        )
        print(
            f"Best solution: {best_phys} physics frames "
            f"({best_beam_step} beam steps, "
            f"{len(solutions)} solutions found)"
        )

        macro_actions = self._reconstruct(trace_history, best_beam_step, best_trace_idx)
        print(f"Reconstructed: {len(macro_actions)} macro-actions")

        # Expand to per-frame actions
        frame_actions = []
        for action_idx in macro_actions:
            frame_actions.extend([action_idx] * self._action_repeat)

        self._validate(frame_actions)
        return frame_actions

    def _reconstruct(
        self, trace_history: list[list[TraceEntry]], step: int, idx: int,
    ) -> list[int]:
        actions: list[int] = []
        current_step = step
        current_idx = idx
        while current_step > 0:
            entry = trace_history[current_step - 1][current_idx]
            actions.append(entry.action_idx)
            current_idx = entry.parent_idx
            current_step -= 1
        actions.reverse()
        return actions

    def _validate(self, actions: list[int]) -> bool:
        env = self._env
        env.reset()
        for i, action_idx in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
            if info.get("ship_exploded", False):
                print(f"VALIDATION FAILED: crash at frame {i}")
                return False
            if info.get("level_completed", False):
                print(f"VALIDATION OK: level completed at frame {i + 1}")
                return True
            if terminated or truncated:
                print(f"VALIDATION FAILED: terminated/truncated at frame {i}")
                return False
        print("VALIDATION FAILED: ran out of actions without completing")
        return False
