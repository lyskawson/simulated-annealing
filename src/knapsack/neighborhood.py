from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .solution import Solution


class Neighborhood(ABC):
    @abstractmethod
    def random_neighbor(self, solution: Solution, rng: random.Random) -> Solution:
        """Return one random neighbor in O(n) time, without enumerating the full neighborhood."""


class BitFlipNeighborhood(Neighborhood):
    """Flip one randomly chosen bit. Neighborhood size = n."""

    def random_neighbor(self, solution: Solution, rng: random.Random) -> Solution:
        idx = rng.randrange(len(solution))
        neighbor = solution.copy()
        neighbor.bits[idx] = 1 - neighbor.bits[idx]
        return neighbor


class SwapNeighborhood(Neighborhood):
    """Swap one position that is 1 with one position that is 0.

    Neighborhood size = |ones| * |zeros|.
    Falls back to BitFlip when all bits are 0 or all are 1.
    """

    def random_neighbor(self, solution: Solution, rng: random.Random) -> Solution:
        ones = [i for i, b in enumerate(solution.bits) if b == 1]
        zeros = [i for i, b in enumerate(solution.bits) if b == 0]

        neighbor = solution.copy()
        if not ones or not zeros:
            idx = rng.randrange(len(solution))
            neighbor.bits[idx] = 1 - neighbor.bits[idx]
            return neighbor

        neighbor.bits[rng.choice(ones)] = 0
        neighbor.bits[rng.choice(zeros)] = 1
        return neighbor
