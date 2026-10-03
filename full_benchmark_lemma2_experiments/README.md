# Common Lemma 2 benchmark

The four runners reuse the original SA, TS, GA-SA and PSO-SA search engines,
parameters and four neighborhoods. An isolated engine instance replaces both
the full decoder and the partial-sequence completion evaluator. Consequently,
initial allocation, removal-gain evaluation, ruin/recreate, population creation,
mutation evaluation, PSO updates, restarts and nested SA/TS all use ascending
`rho = alpha / (q - 1)` before the Lemma 1 completion-time recurrence.
Priorities only break exact rho ties; they cannot override the sequencing rule.
Positive-duration jobs with `q = 1` are placed last, without division by zero.

Original source files and original results are not modified. The experiment
loads `../Metaheuristics.py` and the original benchmark utilities at runtime;
each algorithm config records the engine hash and parameters for provenance.

## Run independently

From the project root:

```powershell
python full_benchmark_lemma2_experiments/run_sa.py
python full_benchmark_lemma2_experiments/run_ts.py
python full_benchmark_lemma2_experiments/run_ga_sa.py
python full_benchmark_lemma2_experiments/run_pso_sa.py
python full_benchmark_lemma2_experiments/collect_results.py
```

Default: all 36 original instances, 10 repetitions, small/medium/large time
limits of 60/90/120 seconds and lambda values of 0.2/0.25/0.3. Instance and
search seed schedules and the adjusted lower bound match the original benchmark.
Example subset: `--cases n20_m2,n60_m3 --replications 10`.

Each process writes only its own algorithm directory under
`full_benchmark_lemma2_results/full_benchmark_36cases_lemma2/`:

- `run_details.csv`: per-run Cmax, GAP, actual elapsed time and initial Cmax.
- `average_gap_summary.csv`: mean Cmax/GAP, standard deviation and best/worst.
- `history/`: actual initial-completion, improvement and final time records.
- `config.json`: solver parameters, decoder, ranges and code provenance.

The initial event is timestamped after construction, not relabelled as time zero.
Event histories are not interpolated into fabricated convergence observations.
The original solver stopping checks are retained; a long neighborhood or
population initialization can still exceed its budget. Elapsed times are stored
as measured, not clipped. For timing comparisons, run algorithms serially on the
same hardware; simultaneous runs compete for CPU resources.

These are four HE-enabled baselines. VNS-HE results are not automatically merged
by their collector. Join the five methods only after checking configurations,
seeds, time limits, and the actual implementation. Original non-HE VNS results
belong to a separate comparison. Parameters here are inherited, not retuned.
