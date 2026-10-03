# Multi-Robot Intralogistics Scheduling

Core research code for heterogeneous robot scheduling with load-dependent
deterioration and asymmetric loaded/empty travel times. The objective is to
minimize makespan through task assignment and sequencing.

## Contents

| File / directory | Purpose |
| --- | --- |
| `VNS-HE.py` | VNS with Lemma 2 sequencing and four assignment neighborhoods |
| `VNS.py` | Non-HE VNS, retaining the neighborhood time checks |
| `Metaheuristics.py` | Original SA, TS, GA-SA, and PSO-SA search engines |
| `Gurobi.py` | Position-indexed mixed-integer formulation |
| `lower_bound.py` | Adjusted average-based and single-task bounds |
| `full_benchmark_experiments/` | Original-method runners, aggregation, plots |
| `full_benchmark_lemma2_experiments/` | Four baseline runners with a common HE decoder |
| `export_instances.py` | Export deterministic instances and their lower bounds |
| `check_project.py` | Syntax and local source-dependency checks, without executing algorithms |

The release retains the main benchmark workflows only. Historical initialization
variants, tuning/sensitivity projects, result-rewriting utilities, and redundant
convergence scripts are excluded. `VNS.py` is the former time-guarded version;
there is no separate `VNS_time_guard.py` or VNS-SE variant.

## Setup

Use Python 3.10 or later, from this directory:

```powershell
python -m pip install -r requirements.txt
python check_project.py
python export_instances.py
```

The last two commands check sources and generate inputs, not optimization runs.
For exact-model experiments, also install `requirements-gurobi.txt` and configure
a valid Gurobi license. Dependency versions are not frozen. Record the actual
environment and hardware alongside results. Figures request Times New Roman.

## Run Experiments

Small trial, written to a separate experiment directory:

```powershell
python full_benchmark_experiments/run_vns_he.py --cases n20_m2 --replications 1 --small-time 5 --experiment-id quick_check
```

Common-sequencing comparison:

```powershell
python full_benchmark_experiments/run_vns_he.py
python full_benchmark_lemma2_experiments/run_sa.py
python full_benchmark_lemma2_experiments/run_ts.py
python full_benchmark_lemma2_experiments/run_ga_sa.py
python full_benchmark_lemma2_experiments/run_pso_sa.py
python full_benchmark_lemma2_experiments/collect_results.py
```

VNS-HE writes to `full_benchmark_results/full_benchmark_36cases/VNS_HE/`; the
four HE baselines write to `full_benchmark_lemma2_results/full_benchmark_36cases_lemma2/`.
Their collector aggregates the four baselines, not VNS-HE automatically.

For the original comparison, use `run_vns.py`, `run_sa.py`, `run_ts.py`,
`run_ga_sa.py`, and `run_pso_sa.py` in `full_benchmark_experiments/`.
The same directory contains `run_gurobi.py`, `run_lower_bound.py`, and
`collect_results.py`. These original metaheuristic runners are non-HE.
Use `--help` before running the full benchmark. Reusing an experiment ID can
overwrite outputs; run timing comparisons serially on the same hardware.

## Protocol and Analysis

36 configurations: 12 small / 12 medium / 12 large. Default heuristic budgets
are 60 / 90 / 120 seconds, lambda is 0.20 / 0.25 / 0.30, and each heuristic uses
10 search repetitions on the same instance per configuration. Gurobi runs once
per configuration with a default 1800-second limit.

`benchmark_common.py` defines inputs, seeds, and the authoritative bound:
`LB = max(adjusted average bound, single-task bound)`.
`GAP (%) = 100 * (Cmax - LB) / LB` is not necessarily the true optimality gap.

The retained plotting scripts visualize original repeated-run results and
Friedman/Nemenyi tests. They do not automatically merge the new HE experiments.
Recompute statistics for the exact methods being compared; do not reuse old
rankings or p-values. See [reproducibility notes](docs/REPRODUCIBILITY.md).

No historical numerical results are bundled. Core search parameters and operators
are preserved; release edits add comments, simplify entry points, and consolidate
the VNS and lower-bound files. No optimization experiments were run during packaging.

## Publication

Add the final paper citation and select an open-source license before publication.
The sequencing rule follows existing deterioration-scheduling results and is not
claimed as a new result. Organization was informed by
[Songyiyu-hub/Project](https://github.com/Songyiyu-hub/Project); no code was copied
from that repository.
