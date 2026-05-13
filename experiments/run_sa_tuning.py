from __future__ import annotations
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.knapsack.algorithms.simulated_annealing import SimulatedAnnealing
from src.knapsack.cooling import geometric
from src.knapsack.initial import empty_solution, greedy_solution, random_feasible_solution
from src.knapsack.instance import generate_instance
from src.knapsack.neighborhood import BitFlipNeighborhood, SwapNeighborhood
from src.knapsack.stopping import MaxIterStopping

_N = 50
_SEED = 42
_RUNS = 5
_MAX_ITER = 50000
_RECORD = 200
_PLOT_DIR = "results/plots"

_BASE = dict(alpha=0.995, t0=None, init="random", neighborhood="bitflip")


def run_variant(instance, neighborhood, alpha, t0, init_fn, runs) -> np.ndarray:
    histories = []
    for run_id in range(runs):
        rng = random.Random(run_id * 1000 + 7)
        algo = SimulatedAnnealing(neighborhood, MaxIterStopping(_MAX_ITER),
                                  geometric(alpha), initial_temperature=t0)
        r = algo.run(instance, init_fn(instance).copy(), rng, record_every=_RECORD)
        histories.append(r.best_history)
    return np.array(histories)


def save_plot(ax, title, path):
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best value (mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print(f"Saved: {path}")


def add_curve(ax, matrix, label, color):
    steps = np.arange(matrix.shape[1]) * _RECORD
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    ax.plot(steps, mean, label=label, color=color, linewidth=2)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    return mean[-1]


def main() -> None:
    instance = generate_instance(_N, _SEED)
    print(f"Instance: n={instance.n}, capacity={instance.capacity}\n")
    os.makedirs(_PLOT_DIR, exist_ok=True)

    bf = BitFlipNeighborhood()
    sw = SwapNeighborhood()

    inits = {"greedy": greedy_solution, "empty": empty_solution,
             "random": lambda inst: random_feasible_solution(inst, random.Random(99))}

    # ------------------------------------------------------------------ #
    # Plot 1: effect of alpha (cooling rate)                              #
    # ------------------------------------------------------------------ #
    alphas = [0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))

    fig, ax = plt.subplots(figsize=(9, 5))
    print("Alpha comparison:")
    for alpha, color in zip(alphas, colors):
        matrix = run_variant(instance, bf, alpha=alpha, t0=None,
                             init_fn=inits[_BASE["init"]], runs=_RUNS)
        final = add_curve(ax, matrix, f"α={alpha}", color)
        print(f"  α={alpha:<6}  mean_final={final:.1f}")
    save_plot(ax, f"Effect of cooling rate α  (n={_N}, init={_BASE['init']})",
              os.path.join(_PLOT_DIR, "sa_alpha.png"))

    # ------------------------------------------------------------------ #
    # Plot 2: effect of initial temperature                               #
    # ------------------------------------------------------------------ #
    t0_values = [0.1, 1.0, 5.0, 20.0, None]   # None = auto from range
    t0_labels = ["T0=0.1", "T0=1", "T0=5", "T0=20", "T0=auto"]
    colors2 = plt.cm.plasma(np.linspace(0.1, 0.9, len(t0_values)))

    fig, ax = plt.subplots(figsize=(9, 5))
    print("\nInitial temperature comparison:")
    for t0, label, color in zip(t0_values, t0_labels, colors2):
        matrix = run_variant(instance, bf, alpha=_BASE["alpha"], t0=t0,
                             init_fn=inits[_BASE["init"]], runs=_RUNS)
        final = add_curve(ax, matrix, label, color)
        print(f"  {label:<10}  mean_final={final:.1f}")
    save_plot(ax, f"Effect of initial temperature T0  (n={_N}, α={_BASE['alpha']})",
              os.path.join(_PLOT_DIR, "sa_t0.png"))

    # ------------------------------------------------------------------ #
    # Plot 3: effect of initial solution                                  #
    # ------------------------------------------------------------------ #
    colors3 = ["#2196F3", "#F44336", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(9, 5))
    print("\nInitial solution comparison:")
    for (init_name, init_fn), color in zip(inits.items(), colors3):
        matrix = run_variant(instance, bf, alpha=_BASE["alpha"], t0=None,
                             init_fn=init_fn, runs=_RUNS)
        final = add_curve(ax, matrix, init_name, color)
        print(f"  {init_name:<10}  mean_final={final:.1f}")
    save_plot(ax, f"Effect of initial solution  (n={_N}, α={_BASE['alpha']})",
              os.path.join(_PLOT_DIR, "sa_init.png"))

    # ------------------------------------------------------------------ #
    # Plot 4: effect of neighborhood                                      #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5))
    print("\nNeighborhood comparison:")
    for (label, neigh), color in zip([("BitFlip", bf), ("Swap", sw)], ["#2196F3", "#FF9800"]):
        matrix = run_variant(instance, neigh, alpha=_BASE["alpha"], t0=None,
                             init_fn=inits[_BASE["init"]], runs=_RUNS)
        final = add_curve(ax, matrix, label, color)
        print(f"  {label:<10}  mean_final={final:.1f}")
    save_plot(ax, f"Effect of neighborhood  (n={_N}, α={_BASE['alpha']})",
              os.path.join(_PLOT_DIR, "sa_neighborhood.png"))


if __name__ == "__main__":
    main()
