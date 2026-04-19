"""Unit tests for strategy classes and registry."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from spaceace.strategies import STRATEGY_REGISTRY, resolve
from spaceace.strategies.actions import ALL_ACTIONS, ACTION_NAMES, DiscreteAction6
from spaceace.strategies.observation import RawObs19, PathAugmentedObs23
from spaceace.strategies.rewards import SparseReward, DenseShapedReward


# ---------------------------------------------------------------------------
# STRATEGY_REGISTRY
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_has_observation_keys(self):
        assert "raw" in STRATEGY_REGISTRY["observation"]
        assert "path_augmented" in STRATEGY_REGISTRY["observation"]

    def test_has_reward_keys(self):
        assert "sparse" in STRATEGY_REGISTRY["reward"]
        assert "dense_shaped" in STRATEGY_REGISTRY["reward"]

    def test_has_actions_keys(self):
        assert "discrete6" in STRATEGY_REGISTRY["actions"]

    def test_has_pathfinder_keys(self):
        assert "rust" in STRATEGY_REGISTRY["pathfinder"]

    def test_resolve_returns_class(self):
        assert resolve("observation", "raw") is RawObs19
        assert resolve("observation", "path_augmented") is PathAugmentedObs23
        assert resolve("reward", "sparse") is SparseReward
        assert resolve("reward", "dense_shaped") is DenseShapedReward

    def test_resolve_unknown_raises(self):
        with pytest.raises(KeyError):
            resolve("observation", "nonexistent")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class TestActions:
    def test_all_actions_length(self):
        assert len(ALL_ACTIONS) == 6

    def test_all_actions_shape(self):
        for a in ALL_ACTIONS:
            assert a.shape == (3,)
            assert a.dtype == np.int32

    def test_action_names_length(self):
        assert len(ACTION_NAMES) == 6

    def test_action_names_match_actions(self):
        assert len(ACTION_NAMES) == len(ALL_ACTIONS)

    def test_coast_is_all_zeros(self):
        np.testing.assert_array_equal(ALL_ACTIONS[0], [0, 0, 0])

    def test_thrust_only(self):
        np.testing.assert_array_equal(ALL_ACTIONS[1], [0, 0, 1])


class TestDiscreteAction6:
    def test_space_shape(self):
        d = DiscreteAction6()
        assert d.space.shape == (3,)

    def test_decode_array(self):
        d = DiscreteAction6()
        result = d.decode(np.array([1, 0, 1], dtype=np.int32))
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_decode_list(self):
        d = DiscreteAction6()
        result = d.decode([0, 1, 0])
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_decode_clips(self):
        d = DiscreteAction6()
        result = d.decode([2, -1, 1])
        np.testing.assert_array_equal(result, [1, 0, 1])


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------


class TestRawObs19:
    def test_space_shape(self):
        obs = RawObs19()
        assert obs.space.shape == (36,)

    def test_passthrough(self):
        obs = RawObs19()
        raw = np.random.randn(36).astype(np.float32)
        result = obs.build(raw, {}, None)
        np.testing.assert_array_equal(result, raw)

    def test_reset_passthrough(self):
        obs = RawObs19()
        raw = np.random.randn(36).astype(np.float32)
        result = obs.reset(raw, {}, None)
        np.testing.assert_array_equal(result, raw)


def _mock_pathfinder():
    pf = MagicMock()
    pf.nearest_pickup_info.return_value = (100.0, 0.5, -0.5)
    return pf


def _mock_env():
    env = MagicMock()
    env.get_pickup_states.return_value = [False, False, False]
    return env


class TestPathAugmentedObs23:
    def test_space_shape(self):
        obs = PathAugmentedObs23(_mock_pathfinder(), max_steps=3000)
        assert obs.space.shape == (40,)

    def test_build_output_shape(self):
        obs = PathAugmentedObs23(_mock_pathfinder(), max_steps=3000)
        raw = np.zeros(36, dtype=np.float32)
        raw[8:16] = 100.0   # coarse wall distances
        raw[20:36] = 100.0  # fine wall distances
        result = obs.reset(raw, {}, _mock_env())
        assert result.shape == (40,)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Reward shapers
# ---------------------------------------------------------------------------


class TestSparseReward:
    def test_passthrough(self):
        r = SparseReward()
        r.reset(np.zeros(20), {}, None)
        result = r.shape(np.zeros(20), np.zeros(3), {"_base_reward": 42.0}, None)
        assert result == 42.0

    def test_default_zero(self):
        r = SparseReward()
        r.reset(np.zeros(20), {}, None)
        result = r.shape(np.zeros(20), np.zeros(3), {}, None)
        assert result == 0.0


class TestDenseShapedReward:
    def test_constants_frozen(self):
        """Verify reward constants haven't drifted from training values."""
        assert DenseShapedReward.STEP_COST == -0.01
        assert DenseShapedReward.CRASH_PENALTY == -200.0
        assert DenseShapedReward.LEVEL_COMPLETE_BONUS == 1000.0
        assert DenseShapedReward.PICKUP_BONUS == 100.0

    def test_crash_returns_penalty(self):
        pf = _mock_pathfinder()
        env = _mock_env()
        r = DenseShapedReward(pf, max_steps=3000)
        raw = np.zeros(19, dtype=np.float32)
        raw[8:16] = 100.0
        r.reset(raw, {}, env)
        result = r.shape(raw, np.array([0, 0, 0]), {"ship_exploded": True}, env)
        assert result == DenseShapedReward.CRASH_PENALTY

    def test_episode_metrics_keys(self):
        pf = _mock_pathfinder()
        env = _mock_env()
        r = DenseShapedReward(pf, max_steps=3000)
        raw = np.zeros(19, dtype=np.float32)
        r.reset(raw, {}, env)
        metrics = r.episode_metrics()
        assert "thrust_ratio" in metrics
        assert "pickups_collected" in metrics
        assert "crashed" in metrics
        assert "completed" in metrics
        assert "length" in metrics
