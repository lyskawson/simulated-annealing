from __future__ import annotations

from typing import Callable

_MIN_TEMP = 1e-10

CoolingFn = Callable[[float], float]


def geometric(alpha: float) -> CoolingFn:
    """t <- alpha * t, alpha [0.95, 0.9999]."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    def _update(t: float) -> float:
        return max(alpha * t, _MIN_TEMP)

    _update.__name__ = f"geometric(alpha={alpha})"
    return _update


def linear(beta: float) -> CoolingFn:
    """t <- t - beta """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    def _update(t: float) -> float:
        return max(t - beta, _MIN_TEMP)

    _update.__name__ = f"linear(beta={beta})"
    return _update
