from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..solution import Solution


@dataclass
class AlgorithmResult:
    algorithm_name: str
    best_solution: Solution
    best_value: float
    best_history: list[float]     # best objective value at each recorded iteration
    current_history: list[float]  # current solution value at each recorded iteration
    runtime_seconds: float
    iterations: int
    params: dict[str, Any] = field(default_factory=dict)
