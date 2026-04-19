"""MCTS agent with bounded-stack rewind.

Keeps a deque of (game_snapshot, tree_checkpoint) taken just before each
commit. If a committed macro-action turns out badly (ship dies, expected
value collapses, or no pickup progress for a while), pop back to an earlier
checkpoint, mask the action we already tried, and search again — reusing
the tree so the re-search is nearly free.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Deque

import numpy as np

import spaceace_rl

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS, ACTION_NAMES


@dataclass
class _Checkpoint:
    game_state: Any            # opaque PyGameState
    tree_cp: Any               # opaque PyMCTSTreeCheckpoint
    expected_value: float      # root baseline when we committed
    step_idx: int
    pickups_remaining: int
    tried: set = field(default_factory=set)


@register_agent("mcts_rewind")
class MCTSRewindAgent(BaseAgent):
    """MCTS with a bounded history of tree checkpoints for cheap rewinds."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._num_simulations = kwargs.get("num_simulations", 5000)
        self._exploration = kwargs.get("exploration_constant", 1.41)
        self._gamma = kwargs.get("gamma", 0.99)
        self._action_repeat = kwargs.get("action_repeat", 5)
        self._ar_depth_bonus = kwargs.get("action_repeat_depth_bonus", 0)
        self._ar_max = kwargs.get("action_repeat_max", 20)
        self._widen_k = kwargs.get("widen_k", 0.0)
        self._thrust_bias = float(kwargs.get("thrust_bias", 0.0))
        self._thrust_bias_safe_dist = float(kwargs.get("thrust_bias_safe_dist", 0.0))

        # Rewind knobs
        self._history_cap = int(kwargs.get("rewind_history", 40))
        self._regret_drop = float(kwargs.get("rewind_regret", 0.35))
        self._stuck_steps = int(kwargs.get("rewind_stuck", 180))
        self._max_rewinds = int(kwargs.get("rewind_budget", 8))
        self._rewind_sims = int(kwargs.get("rewind_num_simulations", self._num_simulations))

        use_momentum = kwargs.get("momentum_pathfinder", False)
        self._mcts = spaceace_rl.PyMCTSEngine(level, max_steps, use_momentum)
        print(f"Pathfinder: {self._mcts.get_pathfinder_info()}")

        self._history: Deque[_Checkpoint] = deque(maxlen=self._history_cap)
        self._rewinds_used = 0
        self._step_idx = 0
        self._last_progress_step = 0
        self._last_pickup_count: Optional[int] = None
        self._rewound_last_step = False

        self.debug_info: Dict[str, Any] = {}

    def reset(self) -> None:
        self._env.reset()
        self._mcts.reset_tree_cache()
        self._history.clear()
        self._rewinds_used = 0
        self._step_idx = 0
        self._last_progress_step = 0
        self._last_pickup_count = None
        self._rewound_last_step = False
        self.debug_info = {}

    # -------- helpers --------

    def _dynamic_scaling(self) -> Tuple[int, int]:
        obs = self._env.get_observation()
        speed = float((obs[2] ** 2 + obs[3] ** 2) ** 0.5)
        min_wall_dist = float(min(obs[8:16]))
        action_repeat = self._action_repeat + int(speed / 50.0)
        num_sims = self._num_simulations
        if min_wall_dist < 150.0:
            num_sims = int(num_sims * (1.0 + (150.0 - min_wall_dist) / 150.0))
        num_sims = int(num_sims * (1.0 + speed / 300.0))
        return num_sims, action_repeat

    def _commit(self, action_idx: int, action_repeat: int) -> Tuple[float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        terminated = truncated = False
        info: Dict[str, Any] = {}
        action = ALL_ACTIONS[action_idx]
        for _ in range(action_repeat):
            _, r, terminated, truncated, info = self._env.step(action)
            total_reward += float(r)
            self._step_idx += 1
            if terminated or truncated:
                break
        return total_reward, terminated, truncated, info

    def _should_rewind(self, current_value: float, info: Dict[str, Any], terminated: bool) -> bool:
        if self._rewinds_used >= self._max_rewinds or not self._history:
            return False
        # Death = always rewind if budget remains
        if terminated and not info.get("level_completed", False):
            return True
        expected = self._history[-1].expected_value
        if current_value < expected - self._regret_drop:
            return True
        if self._step_idx - self._last_progress_step > self._stuck_steps:
            return True
        return False

    def _try_rewind(self, ar: int, sims: int) -> Optional[int]:
        """Walk back the history, masking previously tried actions, until a
        checkpoint yields a new candidate. Returns the new action to commit, or
        None if history is exhausted."""
        while self._history:
            cp = self._history[-1]
            masked = list(cp.tried)
            best_action, _stats, root_baseline = self._mcts.search_from_checkpoint_masked(
                cp.tree_cp, masked,
                sims, ar, self._exploration, self._gamma,
                0.5, False, self._thrust_bias, self._ar_depth_bonus, self._ar_max, self._widen_k,
                self._thrust_bias_safe_dist,
            )
            if best_action != 255:
                cp.tried.add(best_action)
                cp.expected_value = root_baseline
                self._env.load_state(cp.game_state)
                self._step_idx = cp.step_idx
                self._last_pickup_count = cp.pickups_remaining
                self._last_progress_step = cp.step_idx
                self._rewinds_used += 1
                return int(best_action)
            self._history.pop()
        self._mcts.reset_tree_cache()
        return None

    # -------- main step --------

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_state = self._env.save_state()
        num_sims, action_repeat = self._dynamic_scaling()

        action_idx, action_stats, root_baseline = self._mcts.search_with_reuse(
            current_state, num_sims, action_repeat,
            self._exploration, self._gamma,
            0.5, False, self._thrust_bias,
            self._ar_depth_bonus, self._ar_max, self._widen_k,
            self._thrust_bias_safe_dist,
        )

        # Snapshot BEFORE committing — the checkpoint captures the tree as it
        # was when this decision was made, so a rewind can explore siblings.
        tree_cp = self._mcts.checkpoint_tree()
        if tree_cp is not None:
            pickups_remaining = self._env.get_pickups_remaining()
            self._history.append(_Checkpoint(
                game_state=current_state,
                tree_cp=tree_cp,
                expected_value=float(root_baseline),
                step_idx=self._step_idx,
                pickups_remaining=pickups_remaining,
                tried={int(action_idx)},
            ))

        # Commit the action
        self._env.load_state(current_state)
        total_reward, terminated, truncated, info = self._commit(int(action_idx), action_repeat)

        # Track pickup progress
        remaining = int(info.get("pickups_remaining", self._env.get_pickups_remaining()))
        if self._last_pickup_count is None or remaining < self._last_pickup_count:
            self._last_pickup_count = remaining
            self._last_progress_step = self._step_idx

        # Rewind decision. We don't rewind on a completed level.
        self._rewound_last_step = False
        if not (terminated and info.get("level_completed", False)) \
                and self._should_rewind(float(root_baseline), info, terminated):
            alt = self._try_rewind(action_repeat, self._rewind_sims)
            if alt is not None:
                self._rewound_last_step = True
                action_idx = alt
                total_reward, terminated, truncated, info = self._commit(alt, action_repeat)

        self.debug_info = {
            "action_stats": [
                {"name": ACTION_NAMES[a], "visits": v, "mean_value": mv}
                for a, v, mv in sorted(action_stats, key=lambda x: -x[1])
            ],
            "root_heuristic": float(root_baseline),
            "num_simulations": num_sims,
            "action_repeat": action_repeat,
            "history_len": len(self._history),
            "rewinds_used": self._rewinds_used,
            "rewound": self._rewound_last_step,
        }

        return ALL_ACTIONS[int(action_idx)], total_reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        hits, misses = self._mcts.get_reuse_stats()
        total = hits + misses
        if total > 0:
            print(f"[mcts_rewind] tree reuse: {hits}/{total} hits "
                  f"({100.0 * hits / total:.1f}%); rewinds used: {self._rewinds_used}")
        self._env.close()
