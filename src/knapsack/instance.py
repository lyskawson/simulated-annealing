from __future__ import annotations

import sys
import os
from dataclasses import dataclass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from RandomNumberGenerator import RandomNumberGenerator


@dataclass(frozen=True)
class Instance:
    n: int
    costs: tuple[int, ...]
    weights: tuple[int, ...]
    capacity: int

    @property
    def total_weight(self) -> int:
        return sum(self.weights)

    def __repr__(self) -> str:
        items = [(self.costs[i], self.weights[i]) for i in range(self.n)]
        return (
            f"Instance(n={self.n}, capacity={self.capacity}, "
            f"total_weight={self.total_weight}, items={items})"
        )


def generate_instance(n: int, seed: int) -> Instance:

    rng = RandomNumberGenerator(seed)

    costs: list[int] = []
    weights: list[int] = []
    total_weight = 0

    for _ in range(n):
        c = rng.nextInt(1, 10)
        w = rng.nextInt(1, 10)
        costs.append(c)
        weights.append(w)
        total_weight += w

    capacity = rng.nextInt(total_weight // 4, total_weight // 2)

    return Instance(
        n=n,
        costs=tuple(costs),
        weights=tuple(weights),
        capacity=capacity,
    )
