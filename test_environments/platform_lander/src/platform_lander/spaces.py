"""Minimal action and observation spaces for the standalone environment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from platform_lander.core import np_random


class Space:
    """Small subset of Gymnasium's Space API."""

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: type | np.dtype | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self._shape = None if shape is None else tuple(int(x) for x in shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._np_random: np.random.Generator | None = None
        if isinstance(seed, np.random.Generator):
            self._np_random = seed
        elif seed is not None:
            self.seed(seed)

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self._shape

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self.seed()
        assert self._np_random is not None
        return self._np_random

    def seed(self, seed: int | None = None) -> int:
        self._np_random, rng_seed = np_random(seed)
        return rng_seed

    def sample(self):
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        raise NotImplementedError

    def __contains__(self, x: Any) -> bool:
        return self.contains(x)


class Box(Space):
    """A closed box in Euclidean space."""

    def __init__(
        self,
        low: int | float | np.ndarray,
        high: int | float | np.ndarray,
        shape: Sequence[int] | None = None,
        dtype: type | np.dtype = np.float32,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self.dtype = np.dtype(dtype)
        if shape is not None:
            if not isinstance(shape, Iterable):
                raise TypeError("Box shape must be iterable")
            shape = tuple(int(dim) for dim in shape)
        elif isinstance(low, np.ndarray):
            shape = low.shape
        elif isinstance(high, np.ndarray):
            shape = high.shape
        else:
            shape = (1,)

        self.low = np.full(shape, low, dtype=self.dtype) if np.isscalar(low) else np.asarray(low, dtype=self.dtype)
        self.high = np.full(shape, high, dtype=self.dtype) if np.isscalar(high) else np.asarray(high, dtype=self.dtype)
        if self.low.shape != tuple(shape) or self.high.shape != tuple(shape):
            raise ValueError("Box low/high shapes must match the provided shape")
        if np.any(self.low > self.high):
            raise ValueError("Box low values must be less than or equal to high values")
        super().__init__(shape=shape, dtype=self.dtype, seed=seed)

    def sample(self) -> np.ndarray:
        sample = self.np_random.uniform(self.low, self.high)
        return sample.astype(self.dtype)

    def contains(self, x: Any) -> bool:
        try:
            arr = np.asarray(x, dtype=self.dtype)
        except (TypeError, ValueError):
            return False
        return arr.shape == self.shape and bool(np.all(arr >= self.low) and np.all(arr <= self.high))

    def __repr__(self) -> str:
        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"


class Discrete(Space):
    """A finite set of integers ``{start, ..., start + n - 1}``."""

    def __init__(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        start: int = 0,
        dtype: type | np.dtype = np.int64,
    ) -> None:
        if int(n) <= 0:
            raise ValueError("Discrete n must be positive")
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError("Discrete dtype must be an integer dtype")
        self.n = self.dtype.type(n)
        self.start = self.dtype.type(start)
        super().__init__(shape=(), dtype=self.dtype, seed=seed)

    def sample(self):
        return self.start + self.np_random.integers(self.n, dtype=self.dtype.type)

    def contains(self, x: Any) -> bool:
        if isinstance(x, int):
            value = x
        elif isinstance(x, np.generic) and np.issubdtype(x.dtype, np.integer):
            value = int(x)
        elif isinstance(x, np.ndarray) and x.shape == () and np.issubdtype(x.dtype, np.integer):
            value = int(x)
        else:
            return False
        return int(self.start) <= value < int(self.start + self.n)

    def __repr__(self) -> str:
        return f"Discrete({int(self.n)})"

