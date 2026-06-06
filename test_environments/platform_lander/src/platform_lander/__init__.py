"""Standalone SpaceX-style platform landing environment.

This package is intentionally independent from Gymnasium. It keeps the familiar
``reset``/``step``/``render`` API and lightweight ``Box``/``Discrete`` spaces so
it can be used by RL code without importing ``gymnasium``.
"""

from platform_lander.platform_lander import (
    DEFAULT_FAILURE_REWARD,
    DEFAULT_SHAPING_FACTOR,
    DEFAULT_SUCCESS_REWARD,
    DEFAULT_WIND_POWER,
    PlatformLander,
    heuristic,
)
from platform_lander.spaces import Box, Discrete

__all__ = [
    "DEFAULT_WIND_POWER",
    "DEFAULT_SUCCESS_REWARD",
    "DEFAULT_FAILURE_REWARD",
    "DEFAULT_SHAPING_FACTOR",
    "PlatformLander",
    "heuristic",
    "Box",
    "Discrete",
]
