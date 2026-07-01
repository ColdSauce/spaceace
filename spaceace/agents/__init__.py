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
from spaceace.agents import ace  # noqa: E402,F401
from spaceace.agents.tas import agent as _tas_agent  # noqa: E402,F401

__all__ = ["AGENT_REGISTRY", "BaseAgent", "register_agent", "load_agent_module"]
