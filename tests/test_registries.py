"""Unit tests for agent and trainer registries."""

import pytest


class TestAgentRegistry:
    def test_all_agents_registered(self):
        from spaceace.agents import AGENT_REGISTRY

        expected = {"random", "human", "mcts", "ppo", "alphazero", "hrl"}
        assert expected.issubset(set(AGENT_REGISTRY.keys())), (
            f"Missing agents: {expected - set(AGENT_REGISTRY.keys())}"
        )

    def test_agents_are_callable(self):
        from spaceace.agents import AGENT_REGISTRY

        for name, cls in AGENT_REGISTRY.items():
            assert callable(cls), f"Agent {name} is not callable"


class TestTrainerRegistry:
    def test_all_trainers_registered(self):
        from spaceace.training import TRAINER_REGISTRY

        expected = {"sb3", "ppo", "alphazero", "hrl"}
        assert expected.issubset(set(TRAINER_REGISTRY.keys())), (
            f"Missing trainers: {expected - set(TRAINER_REGISTRY.keys())}"
        )

    def test_trainers_are_callable(self):
        from spaceace.training import TRAINER_REGISTRY

        for name, cls in TRAINER_REGISTRY.items():
            assert callable(cls), f"Trainer {name} is not callable"
