from __future__ import annotations
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.knapsack.algorithms.random_search import RandomSearch
from src.knapsack.algorithms.simulated_annealing import SimulatedAnnealing
from src.knapsack.cooling import geometric
from src.knapsack.initial import random_feasible_solution
from src.knapsack.instance import generate_instance
from src.knapsack.neighborhood import BitFlipNeighborhood
from src.knapsack.stopping import MaxIterStopping

_N = 50
_SEED = 42
_RUNS = 10
_MAX_ITER = 50000
_PLOT_DIR = "results/plots"

def main() -> None:
    instance = generate_instance(_N, _SEED)
    print(f"Instance: n={instance.n}, capacity={instance.capacity}")

    neighborhood = BitFlipNeighborhood()
    rs_values, sa_values = [], []

    for run_id in range(_RUNS):
        rng = random.Random(run_id * 1000 + 1)
        init = random_feasible_solution(instance, rng)

        rng_rs = random.Random(run_id * 1000 + 1)
        r_rs = RandomSearch(neighborhood, MaxIterStopping(_MAX_ITER)).run(instance, init.copy(), rng_rs)
        rs_values.append(r_rs.best_value)

        rng_sa = random.Random(run_id * 1000 + 1)
        r_sa = SimulatedAnnealing(
            neighborhood, 
            MaxIterStopping(_MAX_ITER), 
            geometric(0.995)
        ).run(instance, init.copy(), rng_sa)
        sa_values.append(r_sa.best_value)

    rs, sa = np.array(rs_values), np.array(sa_values)
    print(f"\n{'':>4}  {'mean':>7}  {'min':>7}  {'max':>7}  {'std':>6}")
    print(f"{'RS':>4}  {rs.mean():>7.1f}  {rs.min():>7.1f}  {rs.max():>7.1f}  {rs.std():>6.1f}")
    print(f"{'SA':>4}  {sa.mean():>7.1f}  {sa.min():>7.1f}  {sa.max():>7.1f}  {sa.std():>6.1f}")

    os.makedirs(_PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([rs_values, sa_values], tick_labels=["RS", "SA"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    ax.set_title(f"RS vs SA  (n={_N}, {_RUNS} runs)")
    ax.set_ylabel("Best value")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(_PLOT_DIR, "comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nBoxplot saved to {path}")

if __name__ == "__main__":
    main()