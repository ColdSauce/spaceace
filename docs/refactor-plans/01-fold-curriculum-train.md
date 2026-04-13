# Plan 1: Fold `curriculum_train.py` into `Sb3Trainer`

## Context

`spaceace/agents/ppo/curriculum_train.py` (537 lines) runs PPO across a progression of levels grouped into stages. It predates the `Trainer` abstraction and duplicates most of `Sb3Trainer`: env construction, VecNormalize, eval callback, SB3 model setup. The curriculum-specific pieces are:

- `CURRICULUM` list of levels sorted by difficulty (lines 24–26).
- **Calibration**: before each stage, run `PyMCTSEngine` for a few levels to pick `max_steps` (lines 60–113). Results cached in `data/calibration_cache.json` (lines 28–57).
- **Stage transitions**: watch a rolling win rate (`deque(maxlen=3)`); once smoothed win rate clears a threshold, swap the `SubprocVecEnv` to the next stage's levels while preserving `VecNormalize` stats (lines 408–517).
- **Per-stage config**: `max_steps`, sims budget, stage size (5 levels).

Everything else is ordinary PPO training. The value of folding is: one code path for SB3 training, curriculum expressed declaratively, new algorithms (DQN, etc.) get curricula for free.

## Design

### New TrainingConfig fields

```python
# spaceace/training/trainer.py
@dataclass
class LevelStage:
    levels: list[int]
    max_episode_steps: int | None = None  # None -> calibrate from Rust MCTS
    advance_win_rate: float = 0.7
    min_steps: int = 200_000  # don't advance before this many env steps

@dataclass
class TrainingConfig:
    ...
    curriculum: list[LevelStage] | None = None
    calibration_cache_path: Path | None = None
```

When `curriculum is None`, `Sb3Trainer` behaves exactly as today. When set, `Sb3Trainer.fit()` dispatches through a curriculum path.

### Stage transitions via callback

A `CurriculumCallback(BaseCallback)` lives in `spaceace/training/callbacks.py`. It:
1. Reads `info["episode_metrics"]` (already populated by `DenseShapedReward.episode_metrics()` — field `completed`).
2. Maintains a rolling deque of completion rates per evaluation window.
3. When smoothed rate ≥ `stage.advance_win_rate` and `self.num_timesteps ≥ stage.min_steps`, triggers a stage swap.

Stage swap procedure (runs inside the callback):
1. Save `VecNormalize` stats from `self.training_env` to a tempfile.
2. Build the next stage's `VecEnv` via `make_vec_env(...)` with the new level pool.
3. Wrap in a fresh `VecNormalize`, then `VecNormalize.load(stats_file, new_vec_env)` to carry statistics forward.
4. Call `self.model.set_env(new_env)`.

Levels inside a stage are sampled by wrapping the env in a tiny `RandomLevelEnv(gym.Wrapper)` that picks `level = random.choice(stage.levels)` on reset. This lives in `spaceace/training/envs.py` alongside `make_vec_env`.

### Calibration

Calibration is orthogonal to SB3 and should move out of the trainer entirely. New module `spaceace/training/calibration.py`:

```python
def calibrate_max_steps(level: int, cache: Path | None = None) -> int:
    """Run Rust MCTS to estimate how many steps an optimal player needs.

    Rounds up to a multiple of 100 and applies a 2x safety factor. Cached to
    `cache` keyed by (level, max_steps, num_sims)."""
```

`Sb3Trainer.fit()` resolves each stage's `max_episode_steps` via this helper at stage start, not all upfront.

### Migration

1. Add the dataclasses and `CurriculumCallback` + `RandomLevelEnv`.
2. Add `Sb3Trainer._fit_curriculum(config)` — branch at top of `fit()`.
3. Move calibration logic out of `curriculum_train.py` into `spaceace/training/calibration.py`, stripping the stage-management cruft.
4. Rewrite `spaceace/agents/ppo/curriculum_train.py` as a 40-line shim: CLI → `TrainingConfig(curriculum=[...])` → `Sb3Trainer().fit(config)`. Keep the old entry point alive.
5. Delete the inline helpers (`_level_max_steps_cache`, `_level_calibration`, `_calibrate_one_level`, etc.) once calibration.py is wired up.

## Files

**Create**
- `spaceace/training/calibration.py`

**Edit**
- `spaceace/training/trainer.py` — add `LevelStage` dataclass, `curriculum` field
- `spaceace/training/envs.py` — add `RandomLevelEnv` wrapper
- `spaceace/training/callbacks.py` — add `CurriculumCallback`
- `spaceace/training/sb3_trainer.py` — dispatch curriculum path from `fit()`
- `spaceace/agents/ppo/curriculum_train.py` — shrink to ~40-line CLI shim

**Delete**
- Nothing. `curriculum_train.py` keeps its CLI signature for muscle memory.

## Risks

- **VecNormalize carryover**: the current script's stat-preservation between stages is subtle. Getting this wrong shows up as training instability after the first stage swap. Test with a 2-stage micro-curriculum (levels 0→6, 50k steps each) and compare reward curves before vs. after.
- **Win rate signal noise**: smoothing window 3 evaluations is tight. If stage advances feel premature/stalled, expose the window size on `LevelStage`.
- **Calibration behavior change**: moving calibration out-of-line might change *when* it runs. Verify the cache file format stays compatible so existing entries aren't re-computed.

## Verification

```bash
# 1. Non-curriculum path still works (already covered by smoke).
tests/smoke.sh

# 2. Short curriculum runs without crashing.
uv run python -m spaceace.agents.ppo.curriculum_train \
    --timesteps 40000 --stages 0,6 --advance-win-rate 0.5

# 3. Calibration cache hits on second run.
rm -f data/calibration_cache.json
uv run python -m spaceace.agents.ppo.curriculum_train --timesteps 1000 --stages 0
uv run python -m spaceace.agents.ppo.curriculum_train --timesteps 1000 --stages 0
# Second invocation should skip MCTS calibration.
```

## Effort

**Medium.** ~1–1.5 days. Most of the code already exists; the work is reorganization, not new logic.
