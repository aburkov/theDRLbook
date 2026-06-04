"""Small Gymnasium-compatible core helpers used by :mod:`platform_lander`.

The environment code is adapted from Gymnasium's LunarLander v3, but this file
contains the minimal runtime support needed to use it without importing
Gymnasium.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DependencyNotInstalled(ImportError):
    """Raised when an optional rendering or physics dependency is missing."""


class Error(Exception):
    """Base package exception."""


def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Return a NumPy random generator and the seed used to create it."""

    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise Error(f"Seed must be a non-negative python integer, got {seed!r}")

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, int(seed_seq.entropy)


class Env:
    """Minimal environment base class with Gymnasium-style seeding."""

    metadata: dict[str, Any] = {"render_modes": []}
    render_mode: str | None = None

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._np_random, self._np_random_seed = np_random(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator) -> None:
        self._np_random = value
        self._np_random_seed = -1

    @property
    def np_random_seed(self) -> int:
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random_seed

    @property
    def unwrapped(self):
        return self

    def close(self) -> None:
        pass


class EzPickle:
    """Pickle objects by replaying their constructor arguments."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_ezpickle_args": self._ezpickle_args,
            "_ezpickle_kwargs": self._ezpickle_kwargs,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        obj = type(self)(*state["_ezpickle_args"], **state["_ezpickle_kwargs"])
        self.__dict__.update(obj.__dict__)

