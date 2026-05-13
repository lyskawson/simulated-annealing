from __future__ import annotations

import random
import time

from ..instance import Instance
from ..neighborhood import Neighborhood
from ..solution import Solution, ConstraintStrategy
from ..stopping import StoppingCondition
from .base import AlgorithmResult


class RandomSearch:
    """Random Search (RS) for maximisation, as per instrukcja.pdf section 3.3.

    In each iteration a random neighbor x' is drawn.
    If f(x') > f(x) the current solution is updated.
    Because x only improves, the current solution equals the best at all times.
    """

    def __init__(self, neighborhood: Neighborhood, stopping: StoppingCondition) -> None:
        self._neighborhood = neighborhood
        self._stopping = stopping

    def run(
        self,
        instance: Instance,
        initial_solution: Solution,
        rng: random.Random,
        strategy: ConstraintStrategy = "reject",
        penalty_coeff: float = 11.0,
        record_every: int = 1,
    ) -> AlgorithmResult:
        self._stopping.reset()

        x = initial_solution.copy()
        f_x = x.objective(instance, strategy, penalty_coeff)

        best_history: list[float] = []
        current_history: list[float] = []

        start = time.perf_counter()
        iteration = 0

        while not self._stopping.should_stop(iteration, f_x, start):
            x_prime = self._neighborhood.random_neighbor(x, rng)
            f_prime = x_prime.objective(instance, strategy, penalty_coeff)

            if f_prime > f_x:
                x = x_prime
                f_x = f_prime

            if iteration % record_every == 0:
                best_history.append(f_x)
                current_history.append(f_x)

            iteration += 1

        return AlgorithmResult(
            algorithm_name="RandomSearch",
            best_solution=x,
            best_value=f_x,
            best_history=best_history,
            current_history=current_history,
            runtime_seconds=time.perf_counter() - start,
            iterations=iteration,
            params={
                "neighborhood": type(self._neighborhood).__name__,
                "stopping": type(self._stopping).__name__,
                "strategy": strategy,
                "penalty_coeff": penalty_coeff,
            },
        )
