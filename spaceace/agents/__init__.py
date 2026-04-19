"""Agent package — re-exports the registry defined in `base.py` and eager-imports
each built-in agent module so `@register_agent` decorators fire at import time.
"""

from __future__ import annotations

import importlib

from spaceace.agents.base import AGENT_REGISTRY, BaseAgent, register_agent


def load_agent_module(dotted_path: str) -> None:
    """Import an out-of-tree agent module so its `@register_agent` fires."""
    importlib.import_module(dotted_path)


# Eager imports — each decorates its class with @register_agent.
from spaceace.agents import random_agent  # noqa: E402,F401
from spaceace.agents import human  # noqa: E402,F401
from spaceace.agents.mcts import agent as _mcts_agent  # noqa: E402,F401
from spaceace.agents.mcts import goofy_agent as _goofy_mcts_agent  # noqa: E402,F401
from spaceace.agents.mcts import eager_agent as _eager_mcts_agent  # noqa: E402,F401
from spaceace.agents.mcts import rewind_agent as _rewind_mcts_agent  # noqa: E402,F401
from spaceace.agents.ppo import agent as _ppo_agent  # noqa: E402,F401
from spaceace.agents.alphazero import agent as _az_agent  # noqa: E402,F401
from spaceace.agents.hrl import agent as _hrl_agent  # noqa: E402,F401
from spaceace.agents.beam import agent as _beam_agent  # noqa: E402,F401
from spaceace.agents.astar import agent as _astar_agent  # noqa: E402,F401

__all__ = ["AGENT_REGISTRY", "BaseAgent", "register_agent", "load_agent_module"]
