"""Unit tests for the agent registry."""


class TestAgentRegistry:
    def test_all_agents_registered(self):
        from spaceace.agents import AGENT_REGISTRY

        expected = {"random", "human", "tas", "ace"}
        assert expected.issubset(set(AGENT_REGISTRY.keys())), (
            f"Missing agents: {expected - set(AGENT_REGISTRY.keys())}"
        )

    def test_agents_are_callable(self):
        from spaceace.agents import AGENT_REGISTRY

        for name, cls in AGENT_REGISTRY.items():
            assert callable(cls), f"Agent {name} is not callable"
