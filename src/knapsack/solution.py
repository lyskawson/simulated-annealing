from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .instance import Instance

ConstraintStrategy = Literal["reject", "penalty"]

_NEG_INF = float("-inf")
_DEFAULT_PENALTY_COEFF = 11  # strictly > max(c_i)=10, so no excess weight is ever beneficial


@dataclass
class Solution:
    bits: list[int]  # binary vector: bits[i] in {0, 1}

    def copy(self) -> Solution:
        return Solution(bits=self.bits[:])

    def total_value(self, instance: Instance) -> int:
        return sum(instance.costs[i] * self.bits[i] for i in range(instance.n))

    def total_weight(self, instance: Instance) -> int:
        return sum(instance.weights[i] * self.bits[i] for i in range(instance.n))

    def is_feasible(self, instance: Instance) -> bool:
        return self.total_weight(instance) <= instance.capacity

    def objective(
        self,
        instance: Instance,
        strategy: ConstraintStrategy = "reject",
        penalty_coeff: float = _DEFAULT_PENALTY_COEFF,
    ) -> float:
        value = self.total_value(instance)
        weight = self.total_weight(instance)
        excess = max(0, weight - instance.capacity)

        if strategy == "reject":
            return float(value) if excess == 0 else _NEG_INF

        # penalty strategy: penalise excess weight proportionally
        return float(value - penalty_coeff * excess)

    def __len__(self) -> int:
        return len(self.bits)
