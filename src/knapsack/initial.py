from __future__ import annotations

import random

from .instance import Instance
from .solution import Solution


def random_solution(instance: Instance, rng: random.Random) -> Solution:
    return Solution(bits=[rng.randint(0, 1) for _ in range(instance.n)])


def random_feasible_solution(instance: Instance, rng: random.Random) -> Solution:
    sol = random_solution(instance, rng)
    indices = list(range(instance.n))
    rng.shuffle(indices)
    for i in indices:
        if sol.total_weight(instance) <= instance.capacity:
            break
        if sol.bits[i] == 1:
            sol.bits[i] = 0
    return sol


def empty_solution(instance: Instance) -> Solution:
    """All-zeros solution"""
    return Solution(bits=[0] * instance.n)


def greedy_solution(instance: Instance) -> Solution:
    """Greedy by value/weight"""
    order = sorted(
        range(instance.n),
        key=lambda i: instance.costs[i] / instance.weights[i],
        reverse=True,
    )
    bits = [0] * instance.n
    remaining = instance.capacity
    for i in order:
        if instance.weights[i] <= remaining:
            bits[i] = 1
            remaining -= instance.weights[i]
    return Solution(bits=bits)
