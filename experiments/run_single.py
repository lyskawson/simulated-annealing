from __future__ import annotations

import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.knapsack.algorithms.random_search import RandomSearch
from src.knapsack.algorithms.simulated_annealing import SimulatedAnnealing
from src.knapsack.cooling import geometric
from src.knapsack.initial import greedy_solution, random_feasible_solution
from src.knapsack.instance import generate_instance
from src.knapsack.neighborhood import BitFlipNeighborhood
from src.knapsack.stopping import MaxIterStopping

_N = 20
_SEED = 42
_ALGO_SEED = 1
_MAX_ITER = 50000

def main() -> None:
    instance = generate_instance(_N, _SEED)
    print(f"Instance: n={instance.n}, capacity={instance.capacity}, total_weight={instance.total_weight}")
    print(f"  costs:   {list(instance.costs)}")
    print(f"  weights: {list(instance.weights)}")
    print(f"  greedy solution value: {greedy_solution(instance).total_value(instance)}")

    neighborhood = BitFlipNeighborhood()
    stopping = MaxIterStopping(_MAX_ITER)

    rng = random.Random(_ALGO_SEED)
    init = random_feasible_solution(instance, rng)

    print("\n--- Random Search ---")
    rng_rs = random.Random(_ALGO_SEED)
    result = RandomSearch(neighborhood, stopping).run(instance, init.copy(), rng_rs)
    print(f"  best value : {result.best_value}")
    print(f"  iterations : {result.iterations}")
    print(f"  runtime    : {result.runtime_seconds:.3f}s")
    print(f"  solution   : {result.best_solution.bits}")

    print("\n--- Simulated Annealing ---")
    rng_sa = random.Random(_ALGO_SEED)
    result = SimulatedAnnealing(neighborhood, stopping, geometric(0.995)).run(instance, init.copy(), rng_sa)
    print(f"  best value : {result.best_value}")
    print(f"  iterations : {result.iterations}")
    print(f"  runtime    : {result.runtime_seconds:.3f}s")
    print(f"  solution   : {result.best_solution.bits}")

if __name__ == "__main__":
    main()