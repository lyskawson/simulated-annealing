from __future__ import annotations

import time
from abc import ABC, abstractmethod


class StoppingCondition(ABC):
    def reset(self) -> None:
        """Override when the condition has state."""

    @abstractmethod
    def should_stop(self, iteration: int, best_value: float, start_time: float) -> bool: ...


class MaxIterStopping(StoppingCondition):
    def __init__(self, max_iter: int) -> None:
        self.max_iter = max_iter

    def should_stop(self, iteration: int, best_value: float, start_time: float) -> bool:
        return iteration >= self.max_iter


class TimeStopping(StoppingCondition):
    def __init__(self, max_seconds: float) -> None:
        self.max_seconds = max_seconds

    def should_stop(self, iteration: int, best_value: float, start_time: float) -> bool:
        return (time.perf_counter() - start_time) >= self.max_seconds


class NoImprovementStopping(StoppingCondition):

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._no_improve_count = 0
        self._prev_best = float("-inf")

    def reset(self) -> None:
        self._no_improve_count = 0
        self._prev_best = float("-inf")

    def should_stop(self, iteration: int, best_value: float, start_time: float) -> bool:
        if best_value > self._prev_best:
            self._prev_best = best_value
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        return self._no_improve_count >= self.patience
