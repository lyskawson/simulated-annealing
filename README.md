# Discrete Knapsack — Random Search & Simulated Annealing

**Course:** Zaawansowane Techniki Optymalizacji, Lab 5
**Problem:** 2.5 — Discrete Knapsack Problem

## Quick start

```bash
# Install dependencies (requires uv)
uv pip install numpy matplotlib tqdm

# Run RS vs SA comparison (n ∈ {20,50,100,200}, 10 runs each, instance seed=42)
python experiments/run_basic_comparison.py

# Run SA calibration comparison (well vs poorly calibrated)
python experiments/run_calibration.py

# Generate plots from CSV results
python experiments/plot_results.py
```

Results land in `results/csv/` and `results/plots/`.

### CLI options

```bash
python experiments/run_basic_comparison.py \
    --n 20 50 100 200 \
    --seed 42 \
    --max-iter 50000 \
    --runs 10 \
    --strategy reject \
    --output-dir results/csv

python experiments/run_calibration.py \
    --n 20 50 100 \
    --seed 42 \
    --max-iter 50000 \
    --runs 10 \
    --record-every 100 \
    --output-dir results/csv
```

---

## Architecture

```
src/knapsack/
├── instance.py          # Instance dataclass + generate_instance()
├── solution.py          # Solution dataclass + objective() with two constraint strategies
├── neighborhood.py      # BitFlipNeighborhood, SwapNeighborhood
├── initial.py           # random_solution, random_feasible_solution, empty_solution, greedy_solution
├── stopping.py          # MaxIterStopping, TimeStopping, NoImprovementStopping
├── cooling.py           # GeometricCooling, LinearCooling
└── algorithms/
    ├── base.py          # Algorithm ABC + AlgorithmResult dataclass
    ├── random_search.py # RS (instrukcja 3.3, adapted for maximisation)
    └── simulated_annealing.py  # SA (instrukcja 3.5, adapted for maximisation)
```

Each layer is behind an abstract interface. No business logic leaks into composables.

---

## Problem definition (section 2.5)

- **n** items, each with cost `c_i ∈ [1,10]` and weight `w_i ∈ [1,10]`
- Capacity `B ← nextInt(⌊S/4⌋, ⌊S/2⌋)` where `S = Σw_i`
- **Maximise** `Σ x_i·c_i` subject to `Σ x_i·w_i ≤ B`
- Solution representation: binary vector `x ∈ {0,1}^n`

Instance generation follows `problemy.pdf` section 2.5 exactly — only `RandomNumberGenerator` is used, always with an explicit seed.

---

## Design decisions

### Constraint handling — two strategies

| Strategy | Description | Trade-off |
|---|---|---|
| **reject** | Infeasible neighbour returns `−∞`; never accepted | Clean feasibility guarantee; can trap RS/SA when feasible region is isolated |
| **penalty** | `f = Σx_i·c_i − λ·max(0, Σx_i·w_i − B)`, `λ=11` | Allows exploration through infeasible space; λ > max(c_i)=10 so infeasibility is never profitable |

`reject` is the default for experiments because it guarantees all output solutions are feasible and makes the objective value directly comparable across runs.

### Neighbourhoods

| Neighbourhood | Move | Size | Use-case |
|---|---|---|---|
| **BitFlip** | Flip bit `i ∈ [0,n)` | n | Fast, always connected |
| **Swap** | Remove one item (1→0), add another (0→1) | \|ones\|×\|zeros\| | Maintains weight near capacity; better for packed knapsacks |

Both sample one neighbour in O(n) — the full neighbourhood is never materialised.

### Initial solutions

| Name | Description |
|---|---|
| **greedy** | Sort by `c_i/w_i` descending, pack greedily → typically near-optimal |
| **random** | Uniform random binary vector (may be infeasible) |
| **random_feasible** | Random vector repaired by removing items until feasible |
| **empty** | All zeros (always feasible, value = 0) — used as intentionally bad init in calibration |

### RS (section 3.3)

Adapted for maximisation: accept `x′` when `f(x′) > f(x)`.
Since only improvements are accepted, the current solution equals the best at all times — no separate `xbest` needed.

### SA (section 3.5)

Adapted for maximisation. Acceptance probability for a worsening move (`f(x′) ≤ f(x)`):

```
Δ = f(x′) − f(x)   # ≤ 0 when worsening
p = exp(Δ / t)     # ∈ (0, 1]
```

Initial temperature `T₀` is estimated from 1000 random solutions:

```
T₀ = factor × (max(f) − min(f))   # factor=1.0 by default
```

This ensures worsening moves of typical magnitude are accepted ~37% of the time at `T₀`.
`xbest` is maintained separately because SA accepts worsening moves and the current solution can degrade.

### Cooling

**Geometric** (`t ← α·t`): default `α=0.995`. A minimum floor of `1e-10` prevents division-by-zero.

---

## Reproducibility

- `RandomNumberGenerator(seed)` is used **only** in `instance.py` for generating `c_i`, `w_i`, and `B`.
- All algorithm randomness uses a separate `random.Random(algo_seed)` instance passed in.
- Same `instance_seed + algo_seed` → identical results.

---

## Sample results (seed=42, max_iter=50,000, random-feasible init, bit-flip, reject)

| n | RS mean | RS max | SA mean | SA max | Single run max time |
|---|---|---|---|---|---|
| 20 | 44.1 | 68.0 | 67.8 | 74.0 | < 0.15 s |
| 50 | 111.3 | 145.0 | 175.0 | 197.0 | < 0.28 s |
| 100 | 279.6 | 342.0 | 373.7 | 396.0 | < 0.50 s |
| 200 | 327.7 | 354.0 | 477.2 | 525.0 | < 0.92 s |

All runs ≤ 1 s, well within the 10 s limit.
SA consistently outperforms RS thanks to its ability to accept worsening moves and escape local optima.
Starting from a random feasible solution makes the algorithmic difference visible; a greedy init locks
both into the same local optimum with BitFlip+reject.

### SA calibration — n=100

| Variant | T₀ | α | Init | mean best | max best |
|---|---|---|---|---|---|
| SA_good | auto (~range) | 0.995 | greedy | 440.0 | 440.0 |
| SA_bad | 0.1 (fixed) | 0.8 | empty | 282.1 | 325.0 |

The poorly-calibrated SA (too-low T₀, too-fast cooling) behaves like hill-climbing from an empty knapsack: it accepts almost no worsening moves and gets trapped in poor local optima.
