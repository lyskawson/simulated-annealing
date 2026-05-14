from __future__ import annotations

import math
import random
import time

from ..cooling import CoolingFn
from ..instance import Instance
from ..neighborhood import Neighborhood
from ..solution import Solution, ConstraintStrategy
from ..stopping import StoppingCondition
from .base import AlgorithmResult
from ..initial import random_feasible_solution


_MIN_TEMP = 1e-10


def estimate_initial_temperature(
    instance: Instance,
    rng: random.Random,
    strategy: ConstraintStrategy,
    penalty_coeff: float,
    n_samples: int = 1000,
    factor: float = 1.0,
) -> float:
    values = [
        random_feasible_solution(instance, rng).objective(instance, strategy, penalty_coeff)
        for _ in range(n_samples)
    ]
    finite = [v for v in values if v > float("-inf")]
    return factor * (max(finite) - min(finite)) if len(finite) >= 2 else 1.0


class SimulatedAnnealing:

    def __init__(
        self,
        neighborhood: Neighborhood,
        stopping: StoppingCondition,
        cooling: CoolingFn,
        initial_temperature: float | None = None,
        t0_factor: float = 1.0,
        t0_samples: int = 1000,
    ) -> None:
        self._neighborhood = neighborhood
        self._stopping = stopping
        self._cooling = cooling
        self._initial_temperature = initial_temperature
        self._t0_factor = t0_factor
        self._t0_samples = t0_samples

    def run(
        self,
        instance: Instance,
        initial_solution: Solution,
        rng: random.Random,
        strategy: ConstraintStrategy = "reject",
        penalty_coeff: float = 11.0,
        record_every: int = 100,
    ) -> AlgorithmResult:
        self._stopping.reset()

        t = max(
            self._initial_temperature
            if self._initial_temperature is not None
            else estimate_initial_temperature(
                instance, rng, strategy, penalty_coeff,
                n_samples=self._t0_samples, factor=self._t0_factor,
            ),
            _MIN_TEMP,
        )

        x = initial_solution.copy()
        f_x = x.objective(instance, strategy, penalty_coeff)

        x_best = x.copy()
        f_best = f_x

        best_history: list[float] = []
        current_history: list[float] = []

        start = time.perf_counter()
        iteration = 0

        while not self._stopping.should_stop(iteration, f_best, start):
            x_prime = self._neighborhood.random_neighbor(x, rng)
            f_prime = x_prime.objective(instance, strategy, penalty_coeff)

            if f_prime > f_x:
                x = x_prime
                f_x = f_prime
            else:
                delta = f_prime - f_x
                if rng.random() < math.exp(delta / t):
                    x = x_prime
                    f_x = f_prime

            if f_x > f_best:
                x_best = x.copy()
                f_best = f_x

            t = self._cooling(t)

            if iteration % record_every == 0:
                best_history.append(f_best)
                current_history.append(f_x)

            iteration += 1

        return AlgorithmResult(
            algorithm_name="SimulatedAnnealing",
            best_solution=x_best,
            best_value=f_best,
            best_history=best_history,
            current_history=current_history,
            runtime_seconds=time.perf_counter() - start,
            iterations=iteration,
            params={
                "neighborhood": type(self._neighborhood).__name__,
                "stopping": type(self._stopping).__name__,
                "cooling": getattr(self._cooling, "__name__", repr(self._cooling)),
                "initial_temperature": self._initial_temperature,
                "t0_factor": self._t0_factor,
                "strategy": strategy,
                "penalty_coeff": penalty_coeff,
            },
        )
