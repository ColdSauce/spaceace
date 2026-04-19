"""Reusable SB3 callbacks. Agnostic to the specific RL algorithm."""

from __future__ import annotations

from collections import deque

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

# Feature names for the 40-dim path_augmented observation.
_PA_LABELS = [
    "vx/300", "vy/300", "sin_rot", "cos_rot",
    "pickup_dist/1k",
    "wall_fwd", "wall_fwd_r", "wall_r", "wall_bk_r",
    "wall_bk", "wall_bk_l", "wall_l", "wall_fwd_l",
    "pickups_rem", "norm_x", "norm_y",
    "path_dist", "dir_x", "dir_y", "speed",
    "speed_toward", "heading_align", "min_tti", "time_rem",
] + [f"fine_{i}" for i in range(16)]


class MetricsCallback(BaseCallback):
    """Pulls `episode_metrics` out of info dicts and records them to the SB3 logger.

    Any RewardShaper that populates `episode_metrics()` flows through here for
    free — no per-agent code needed.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue
            for key, value in metrics.items():
                self.logger.record(f"episode/{key}", float(value))
        return True


class CurriculumCallback(BaseCallback):
    """Advance through curriculum stages based on rolling completion rate.

    Reads ``info["episode_metrics"]["completed"]`` (already populated by
    DenseShapedReward). When the smoothed rate clears the stage threshold,
    swaps the VecEnv to the next stage's levels while preserving
    VecNormalize statistics.
    """

    # Entropy-kick parameters (see _maybe_entropy_kick).
    _KICK_AFTER_STEPS = 500_000       # steps in stage before we consider a kick
    _KICK_WIN_RATE_BELOW = 0.3        # stuck = smoothed win rate under this
    _KICK_RESTORE_WIN_RATE = 0.5      # recovered = smoothed win rate over this
    _KICK_MULTIPLIER = 5.0            # how much to scale ent_coef while kicking

    # MCTS kickstart: when totally stuck, run MCTS to demonstrate the solution
    # and do behavioral cloning on the policy to teach it.
    _MCTS_AFTER_STEPS = 500_000       # steps at 0% before triggering MCTS demo
    _MCTS_WIN_RATE_BELOW = 0.01       # effectively 0%
    _MCTS_DEMO_EPISODES = 5           # number of MCTS episodes to collect
    _MCTS_SIMULATIONS = 2000          # MCTS sims per decision
    _MCTS_BC_EPOCHS = 3               # behavioral cloning epochs per kickstart
    _MCTS_BC_LR = 1e-3                # BC learning rate (higher than PPO for fast imitation)

    # Force-skip: if still stuck after MCTS kickstart + this many more steps, skip.
    _SKIP_AFTER_STEPS = 1_000_000     # total steps in stage before force-skip
    _SKIP_WIN_RATE_BELOW = 0.01       # effectively 0%

    def __init__(
        self,
        stages: list,
        make_env_fn,
        obs: str,
        reward: str,
        action_repeat: int,
        n_envs: int,
        pathfinder_backend: str = "grid",
        eval_env=None,
        window: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._stages = stages
        self._make_env_fn = make_env_fn
        self._obs = obs
        self._reward = reward
        self._action_repeat = action_repeat
        self._n_envs = n_envs
        self._pathfinder_backend = pathfinder_backend
        self._eval_env = eval_env
        self._window = window
        self._stage_idx = 0
        # Per-episode completion window: each entry is one finished episode.
        # Previous implementation batched in chunks of n_envs, which biased
        # smoothing toward the fastest-finishing (usually crashing) envs.
        self._recent: deque[float] = deque(maxlen=max(window * n_envs, 32))
        self._stage_start_step = 0
        # Entropy-kick state
        self._base_ent_coef: float | None = None
        self._kick_active = False
        # MCTS kickstart state
        self._mcts_attempted = False  # only try once per stage

    @property
    def current_stage(self):
        return self._stages[self._stage_idx]

    def _on_training_start(self) -> None:
        self._stage_start_step = 0
        self._pending_advance = False
        # Capture the baseline ent_coef so the kick can always restore it.
        self._base_ent_coef = float(self.model.ent_coef)

    def _on_step(self) -> bool:
        new_episodes = 0
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue
            self._recent.append(float(metrics.get("completed", 0)))
            new_episodes += 1

        if new_episodes and self._recent:
            smoothed = sum(self._recent) / len(self._recent)
            self.logger.record("curriculum/stage", self._stage_idx)
            self.logger.record("curriculum/smoothed_win_rate", smoothed)
            self.logger.record("curriculum/recent_n", len(self._recent))

        self._maybe_entropy_kick()
        self._maybe_mcts_kickstart()
        self._maybe_force_skip()
        self._maybe_log_stuck_diagnostics()

        if self._should_advance():
            self._pending_advance = True

        return True

    def _maybe_entropy_kick(self) -> None:
        """Shake the policy out of a rut when a stage has stalled.

        Trigger: stage has been running >_KICK_AFTER_STEPS and smoothed win
        rate is below _KICK_WIN_RATE_BELOW. We bump ent_coef by _KICK_MULTIPLIER
        until the win rate recovers past _KICK_RESTORE_WIN_RATE, then put it
        back. The mastery gate (advance_win_rate) is unchanged — we're just
        letting the agent re-explore on the current level geometry.
        """
        if self._base_ent_coef is None or len(self._recent) < max(self._window, self._n_envs):
            return
        smoothed = sum(self._recent) / len(self._recent)
        steps_in_stage = self.num_timesteps - self._stage_start_step
        stuck = (
            steps_in_stage >= self._KICK_AFTER_STEPS
            and smoothed < self._KICK_WIN_RATE_BELOW
        )
        recovered = smoothed >= self._KICK_RESTORE_WIN_RATE

        if stuck and not self._kick_active:
            self._kick_active = True
            self.model.ent_coef = self._base_ent_coef * self._KICK_MULTIPLIER
            print(
                f"\n[entropy-kick ON] stage {self._stage_idx + 1}: "
                f"win_rate={smoothed:.2f} after {steps_in_stage:,} steps, "
                f"ent_coef {self._base_ent_coef} -> {self.model.ent_coef}"
            )
        elif recovered and self._kick_active:
            self._kick_active = False
            self.model.ent_coef = self._base_ent_coef
            print(
                f"\n[entropy-kick OFF] stage {self._stage_idx + 1}: "
                f"win_rate={smoothed:.2f} recovered, "
                f"ent_coef restored to {self._base_ent_coef}"
            )

        self.logger.record("curriculum/ent_coef", float(self.model.ent_coef))
        self.logger.record("curriculum/entropy_kick_active", int(self._kick_active))

    def _on_rollout_start(self) -> None:
        """Swap env between rollouts so training never sees mixed data."""
        if self._pending_advance:
            self._pending_advance = False
            self._advance_stage()

    def _should_advance(self) -> bool:
        if self._stage_idx >= len(self._stages) - 1:
            return False
        stage = self.current_stage
        steps_in_stage = self.num_timesteps - self._stage_start_step
        if steps_in_stage < stage.min_steps:
            return False
        if len(self._recent) < max(self._window, self._n_envs):
            return False
        smoothed = sum(self._recent) / len(self._recent)
        return smoothed >= stage.advance_win_rate

    def _advance_stage(self) -> None:
        prev_stage_idx = self._stage_idx
        self._stage_idx += 1
        if self._stage_idx >= len(self._stages):
            return
        stage = self.current_stage
        print(
            f"\n>>> Advancing to stage {self._stage_idx + 1}/{len(self._stages)}: "
            f"levels {stage.levels}, max_steps={stage.max_episode_steps}"
        )

        # Snapshot the LAST observation from the old stage (env 0) for comparison.
        prev_obs = self.model._last_obs[0].copy() if self.model._last_obs is not None else None

        # Update the level pool on each RandomLevelEnv inside the VecEnv.
        # No env teardown/rebuild — just change what levels get sampled on reset.
        # Uses env_method() which works with both DummyVecEnv and SubprocVecEnv.
        vec_env = self.model.get_env()
        raw_vec = vec_env.venv if hasattr(vec_env, 'venv') else vec_env
        raw_vec.env_method("set_curriculum", stage.levels, stage.max_episode_steps)

        # Keep the eval env in sync so EvalCallback evaluates on current-stage levels.
        if self._eval_env is not None:
            raw_eval = self._eval_env.venv if hasattr(self._eval_env, 'venv') else self._eval_env
            raw_eval.env_method("set_curriculum", stage.levels, stage.max_episode_steps)

        # Reset so the new levels take effect immediately
        self.model._last_obs = vec_env.reset()
        self.model._last_episode_starts = np.ones((self._n_envs,), dtype=bool)

        # --- Stage transition diagnostics ---
        self._log_stage_transition(prev_stage_idx, prev_obs)

        self._recent.clear()
        self._episode_completions.clear()
        self._stage_start_step = self.num_timesteps
        self._mcts_attempted = False
        # Fresh stage: always restore baseline entropy — the new stage gets a
        # clean evaluation at the normal exploration level before we consider
        # another kick.
        if self._base_ent_coef is not None and self._kick_active:
            self.model.ent_coef = self._base_ent_coef
            self._kick_active = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _maybe_mcts_kickstart(self) -> None:
        """When totally stuck, run MCTS to demonstrate the solution and do BC."""
        if self._mcts_attempted:
            return
        if self._base_ent_coef is None or len(self._recent) < max(self._window, self._n_envs):
            return
        smoothed = sum(self._recent) / len(self._recent)
        steps_in_stage = self.num_timesteps - self._stage_start_step
        if steps_in_stage < self._MCTS_AFTER_STEPS or smoothed >= self._MCTS_WIN_RATE_BELOW:
            return

        self._mcts_attempted = True
        stage = self.current_stage
        level = stage.levels[0]
        max_steps = stage.max_episode_steps or 3000

        print(
            f"\n[mcts-kickstart] stage {self._stage_idx + 1} level {level}: "
            f"win_rate={smoothed:.3f} after {steps_in_stage:,} steps — "
            f"running {self._MCTS_DEMO_EPISODES} MCTS demos"
        )

        demos = self._collect_mcts_demos(level, max_steps)
        if not demos:
            print("[mcts-kickstart] MCTS failed to complete any episodes, skipping BC")
            return

        obs_all, actions_all = zip(*demos)
        print(
            f"[mcts-kickstart] collected {len(obs_all)} demo steps "
            f"from {self._MCTS_DEMO_EPISODES} episodes"
        )
        self._behavioral_cloning(obs_all, actions_all)

    def _maybe_force_skip(self) -> None:
        """Force-advance past a hopelessly stuck stage to prevent catastrophic forgetting."""
        if self._stage_idx >= len(self._stages) - 1:
            return
        if len(self._recent) < max(self._window, self._n_envs):
            return
        smoothed = sum(self._recent) / len(self._recent)
        steps_in_stage = self.num_timesteps - self._stage_start_step
        if steps_in_stage < self._SKIP_AFTER_STEPS or smoothed >= self._SKIP_WIN_RATE_BELOW:
            return

        stage = self.current_stage
        print(
            f"\n[force-skip] stage {self._stage_idx + 1} levels={stage.levels}: "
            f"win_rate={smoothed:.3f} after {steps_in_stage:,} steps — "
            f"skipping to avoid catastrophic forgetting"
        )
        self.logger.record("curriculum/force_skipped", 1)
        self._pending_advance = True

    def _collect_mcts_demos(
        self, level: int, max_steps: int
    ) -> list[tuple[np.ndarray, int]]:
        """Run MCTS episodes and return (obs, action_index) pairs for BC.

        Observations are path_augmented (matching what the PPO policy sees).
        Actions are Discrete(6) indices matching StrategyWrapper's action space.
        Exploration constant varies per episode so deterministic MCTS doesn't
        produce 5 identical trajectories.
        """
        import spaceace_rl
        from spaceace.core.gym_wrapper import SpaceAceGymWrapper
        from spaceace.training.envs import StrategyWrapper, _build_strategies

        demos: list[tuple[np.ndarray, int]] = []

        # Vary exploration to force trajectory diversity.
        exploration_schedule = [1.0, 1.41, 1.8, 2.2, 2.8]

        for ep in range(self._MCTS_DEMO_EPISODES):
            # Create fresh env + MCTS engine for this episode
            base_env = SpaceAceGymWrapper(level=level, max_steps=max_steps)
            obs_strategy, reward_strategy, pf = _build_strategies(
                level, max_steps, self._obs, self._reward, self._pathfinder_backend
            )
            wrapped = StrategyWrapper(
                base_env, obs_strategy, reward_strategy,
                action_repeat=self._action_repeat, pathfinder=pf,
            )

            mcts = spaceace_rl.PyMCTSEngine(level, max_steps, False)
            raw_env = base_env.env  # SpaceAceDirectEnv

            # Reset both
            pa_obs = wrapped.reset()[0]  # path_augmented obs
            raw_env.reset()  # sync the raw env

            done = False
            ep_steps = 0
            c_puct = exploration_schedule[ep % len(exploration_schedule)]
            while not done:
                # Get MCTS action using the raw env's state
                state = raw_env.save_state()
                action_idx = mcts.search(
                    state,
                    self._MCTS_SIMULATIONS,
                    self._action_repeat,
                    c_puct,
                    0.99,  # gamma
                )

                # Record the demo: current obs + MCTS action index (Discrete(6))
                demos.append((pa_obs.copy(), int(action_idx)))

                # Step the wrapped env with the Discrete action index
                pa_obs, reward, terminated, truncated, info = wrapped.step(action_idx)
                done = terminated or truncated
                ep_steps += 1

                if ep_steps > max_steps // self._action_repeat:
                    break

            completed = info.get("episode_metrics", {}).get("completed", False)
            print(
                f"  MCTS ep {ep + 1}: {'COMPLETED' if completed else 'FAILED'} "
                f"in {ep_steps} steps"
            )

        return demos

    def _behavioral_cloning(
        self,
        obs_list: tuple[np.ndarray, ...],
        actions_list: tuple[np.ndarray, ...],
    ) -> None:
        """Do a few epochs of supervised learning on the PPO policy."""
        import torch
        import torch.nn.functional as F

        policy = self.model.policy
        device = policy.device

        obs_tensor = torch.tensor(
            np.array(obs_list), dtype=torch.float32, device=device
        )
        actions_tensor = torch.tensor(
            np.array(actions_list), dtype=torch.long, device=device
        )

        # Use a separate optimizer so we don't mess with PPO's optimizer state
        bc_optimizer = torch.optim.Adam(policy.parameters(), lr=self._MCTS_BC_LR)

        for epoch in range(self._MCTS_BC_EPOCHS):
            # Shuffle
            indices = torch.randperm(len(obs_tensor), device=device)
            batch_size = 256
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_obs = obs_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]

                # evaluate_actions returns values, log_prob, entropy
                _, log_prob, entropy = policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Maximize log-prob of expert actions
                bc_loss = -log_prob.mean()

                bc_optimizer.zero_grad()
                bc_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                bc_loss_val = bc_loss.item()
                bc_optimizer.step()

                total_loss += bc_loss_val
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"  BC epoch {epoch + 1}/{self._MCTS_BC_EPOCHS}: loss={avg_loss:.4f}")

        print("[mcts-kickstart] behavioral cloning complete, resuming PPO")

    _DIAG_INTERVAL = 500_000  # log diagnostics every N steps while stuck

    def _log_stage_transition(self, prev_stage_idx: int, prev_obs) -> None:
        """Dump observation comparison at a stage transition for debugging."""
        new_obs = self.model._last_obs[0]
        labels = _PA_LABELS if len(new_obs) == 40 else [f"f{i}" for i in range(len(new_obs))]

        prev_levels = self._stages[prev_stage_idx].levels if prev_stage_idx < len(self._stages) else []
        new_levels = self.current_stage.levels

        print(f"\n[diag] stage {prev_stage_idx}→{self._stage_idx}  "
              f"levels {prev_levels}→{new_levels}  step={self.num_timesteps:,}")

        # Show features that changed meaningfully between old and new obs.
        if prev_obs is not None and len(prev_obs) == len(new_obs):
            diffs = []
            for i, (old, new, lbl) in enumerate(zip(prev_obs, new_obs, labels)):
                delta = abs(float(new) - float(old))
                if delta > 0.001:
                    diffs.append((delta, i, lbl, float(old), float(new)))
            diffs.sort(reverse=True)
            if diffs:
                print(f"[diag] obs diffs (top {min(10, len(diffs))}):")
                for delta, i, lbl, old, new in diffs[:10]:
                    print(f"  [{i:2d}] {lbl:>16s}: {old:+.4f} → {new:+.4f}  (Δ{delta:.4f})")
            else:
                print("[diag] obs: no meaningful changes")

        # Show the key pathfinder features on the new level.
        if len(new_obs) >= 24:
            print(f"[diag] new obs key features: "
                  f"dir=({new_obs[17]:+.4f},{new_obs[18]:+.4f}) "
                  f"path_dist={new_obs[16]:.4f} "
                  f"pickups_rem={new_obs[13]:.1f} "
                  f"speed={new_obs[19]:.2f}")

        # Query the policy for its action on the new obs (deterministic).
        try:
            obs_tensor = np.expand_dims(new_obs, 0)
            action, _ = self.model.predict(obs_tensor, deterministic=True)
            action_names = ["coast", "thrust", "left", "left+thr", "right", "right+thr"]
            act_idx = int(action[0]) if np.isscalar(action[0]) else int(action[0].item()) if hasattr(action[0], 'item') else action[0]
            act_str = action_names[act_idx] if isinstance(act_idx, int) and act_idx < len(action_names) else str(action)
            print(f"[diag] policy deterministic action: {act_str} (raw={action})")
        except Exception as e:
            print(f"[diag] policy query failed: {e}")

    def _maybe_log_stuck_diagnostics(self) -> None:
        """Periodically dump diagnostics while stuck on a stage."""
        steps_in_stage = self.num_timesteps - self._stage_start_step
        if steps_in_stage < self._DIAG_INTERVAL:
            return
        if steps_in_stage % self._DIAG_INTERVAL > self._n_envs * 10:
            return  # only fire once near the boundary

        if len(self._recent) < max(self._window, self._n_envs):
            return
        smoothed = sum(self._recent) / len(self._recent)

        stage = self.current_stage
        obs = self.model._last_obs[0] if self.model._last_obs is not None else None
        if obs is None:
            return

        print(f"\n[diag-stuck] stage {self._stage_idx + 1} levels={stage.levels} "
              f"step={self.num_timesteps:,} in_stage={steps_in_stage:,} "
              f"win_rate={smoothed:.3f}")

        if len(obs) >= 24:
            print(f"  obs: dir=({obs[17]:+.4f},{obs[18]:+.4f}) "
                  f"path_dist={obs[16]:.4f} "
                  f"pickups_rem={obs[13]:.1f} "
                  f"speed={obs[19]:.2f} "
                  f"wall_fwd={obs[5]:.3f} wall_r={obs[7]:.3f}")

        # Sample the last few episode infos for crash/pickup data.
        for info in self.locals.get("infos", [])[:1]:
            metrics = info.get("episode_metrics")
            if metrics:
                print(f"  last_episode: {metrics}")
