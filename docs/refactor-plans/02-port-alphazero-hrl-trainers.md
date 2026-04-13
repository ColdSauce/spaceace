# Plan 2: Port AlphaZero and HRL training to `Trainer` subclasses

## Context

`Sb3Trainer` handles PPO (and any SB3 algo). AlphaZero and HRL currently ship as standalone training scripts with zero shared infrastructure — different file layouts, different model-saving conventions, different CLI flags. The goal is parity: every training entry point is `SomeTrainer(Trainer).fit(TrainingConfig)`.

The two are very different internally:

- **AlphaZero** (`spaceace/agents/alphazero/train.py`, 365 lines; `self_play.py`, `network.py`) uses **custom PyTorch, not SB3**. Loop: self-play via Rust `PyAlphaZeroEngine.play_games()` → replay buffer of `.npz` shards → supervised training of an `AlphaZeroNet` via `train_network()`. No gym Env, no VecEnv.
- **HRL** (`spaceace/agents/hrl/train_pilot.py`, 197 lines; `waypoint_env.py`, 321 lines; `generate_corridors.py`, 243 lines) trains a PPO pilot over a **waypoint-conditioned gym env**. Uses SB3, so it's structurally close to `Sb3Trainer`. Also depends on offline corridor level generation.

These need different handling.

## AlphaZero: `AlphaZeroTrainer`

### Design

AlphaZero doesn't fit the VecEnv shape `Sb3Trainer` assumes. Instead of forcing it into SB3 scaffolding, give it its own Trainer subclass with AlphaZero-specific sub-methods:

```python
# spaceace/agents/alphazero/trainer.py
class AlphaZeroTrainer(Trainer):
    def fit(self, config: TrainingConfig) -> Path:
        net = self._build_or_load_network(config)
        buffer = ReplayBuffer(path=config.save_dir / "replay", max_shards=50)
        for iteration in range(config.alphazero.iterations):
            stage = self._pick_stage(iteration, config.curriculum)
            self._self_play(net, stage, buffer, config)
            self._train_network(net, buffer, config)
            self._evaluate(net, stage, config)
            self._save_checkpoint(net, iteration, config.save_dir)
        return config.save_dir / "best_model.pt"
```

Each private method is a direct lift from `train.py` with the CLI/argparse/main plumbing removed. No behavior change.

### New `TrainingConfig` fields

AlphaZero needs a handful of hyperparams that PPO doesn't. Rather than bloating the base dataclass, nest them:

```python
@dataclass
class AlphaZeroHparams:
    iterations: int = 100
    games_per_iteration: int = 200
    simulations_per_move: int = 200
    replay_buffer_shards: int = 50
    network_train_epochs: int = 10
    network_batch_size: int = 256

@dataclass
class TrainingConfig:
    ...
    alphazero: AlphaZeroHparams = field(default_factory=AlphaZeroHparams)
```

PPO runs ignore `config.alphazero` entirely.

### Curriculum

AlphaZero's `CURRICULUM_STAGES` becomes the same `list[LevelStage]` field added in Plan 1. AlphaZero's custom "advance when win rate ≥ X" logic is the same rule — it can share `CurriculumCallback`'s decision function (refactor that function to a pure `should_advance(metrics_window, stage)` helper callable from both trainers).

### Registration

`spaceace/training/__init__.py` gains `TRAINER_REGISTRY["alphazero"] = AlphaZeroTrainer`. Update `train.py` (the top-level shim) or add `spaceace/cli/train.py` to dispatch via `TRAINER_REGISTRY[args.trainer]`.

## HRL: `HrlTrainer`

### Design

Because HRL's pilot is PPO-based, `HrlTrainer` composes with `Sb3Trainer` rather than reimplementing it:

```python
# spaceace/agents/hrl/trainer.py
class HrlTrainer(Trainer):
    def fit(self, config: TrainingConfig) -> Path:
        self._ensure_corridors(config)   # calls generate_corridors once
        sb3 = Sb3Trainer(env_factory=self._waypoint_env_factory)
        return sb3.fit(config)
```

The key extension point is `env_factory` on `Sb3Trainer`. Today `Sb3Trainer.fit` calls `make_vec_env(...)` hardcoded. Add an optional `env_factory: Callable[[TrainingConfig, int], VecEnv] | None = None` constructor arg; if set, it's used instead. `HrlTrainer` passes a factory that wraps `SpaceAceGymWrapper` in `WaypointEnv` before `StrategyWrapper`.

### WaypointEnv as a strategy (stretch)

`WaypointEnv` is really an `ObservationBuilder` + `RewardShaper` combo: it adds waypoint-conditioned features and PBRS rewards. Cleanest long-term move: split it into `WaypointObs` and `WaypointReward` strategies in `spaceace/strategies/`, register them, and HRL becomes `TrainingConfig(obs="waypoint", reward="waypoint_pbrs")`. That is a bigger rewrite than this plan needs; start with the env_factory escape hatch, promote to strategies later if the pattern catches on.

### Corridor generation

Move `spaceace/agents/hrl/generate_corridors.py` → `spaceace/tools/generate_corridors.py`. `HrlTrainer._ensure_corridors()` imports it, checks whether the corridor level files exist, generates them if not. This turns a manual pre-step into an automatic one.

## Files

**Create**
- `spaceace/agents/alphazero/trainer.py` — `AlphaZeroTrainer`
- `spaceace/agents/hrl/trainer.py` — `HrlTrainer`

**Edit**
- `spaceace/training/trainer.py` — add `AlphaZeroHparams`, extend `TrainingConfig`
- `spaceace/training/sb3_trainer.py` — add `env_factory` constructor arg
- `spaceace/training/__init__.py` — register both new trainers
- `spaceace/agents/alphazero/train.py` — shrink to CLI shim calling `AlphaZeroTrainer`
- `spaceace/agents/hrl/train_pilot.py` — shrink to CLI shim calling `HrlTrainer`

**Move**
- `spaceace/agents/hrl/generate_corridors.py` → `spaceace/tools/generate_corridors.py`

## Risks

- **AlphaZero replay buffer format**: `.npz` shard layout is load-bearing. Don't rename fields or change shard size during the port — that would invalidate existing buffers.
- **`set_env` on SB3 PPO**: setting a new env factory midway through training requires `model.set_env()`. For HRL, this is only called once at start, so the risk is minimal. Flag if Plan 1's curriculum callback runs inside HRL — it would call `set_env` too, and the waypoint factory must be applied consistently.
- **Corridor auto-generation**: generating 100 levels on first run is slow. Print a clear progress banner; respect `--skip-corridor-gen` for CI.
- **Curriculum sharing**: refactoring `should_advance()` into a shared helper is easy; verify AlphaZero and PPO both consume `info["episode_metrics"]["completed"]` or adjust accordingly.

## Verification

```bash
# AlphaZero micro-run
uv run python -m spaceace.agents.alphazero.train \
    --iterations 2 --games-per-iteration 4 --simulations 50 --level 0

# HRL micro-run (auto-generates corridors on first call)
uv run python -m spaceace.agents.hrl.train_pilot --timesteps 20000 --level 0

# Both agents still play via run.py (unchanged)
uv run python run.py --agent alphazero --level 0 --headless --episodes 1
uv run python run.py --agent hrl --level 0 --headless --episodes 1
```

Add these three commands to `tests/smoke.sh` once the trainers exist.

## Effort

**Medium (AlphaZero) + Small (HRL) = ~3 days combined.** AlphaZero is bigger because its training loop is custom PyTorch and needs careful line-by-line relocation. HRL is mostly adapter work.

**No prerequisites.** Plan 4 (backend wiring) is *not* required — HRL depends on grid-specific pathfinder methods, which already work.
