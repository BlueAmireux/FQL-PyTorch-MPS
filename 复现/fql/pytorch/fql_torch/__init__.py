"""State-only FQL port: FQLAgent exposes loss, update, action and Flax mapping APIs."""

from .agent import FQLAgent, get_config

__all__ = ["FQLAgent", "get_config"]
