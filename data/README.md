# Benchmark Inputs

Run `python export_instances.py` from the repository root to create
`data/instances/n<n>_m<m>.json` for the 36 benchmark configurations and
`data/benchmark_manifest.csv` with seeds, time limits, and lower bounds.

This command generates deterministic inputs and evaluates the theoretical lower
bound; it does not run VNS, metaheuristics, or Gurobi. Existing files are protected
unless `--overwrite` is specified.

Each JSON contains `n`, `m`, `seed`, `t0`, `lambda`, task arrays `L`, `U`, `w`,
ADR arrays `Q`, `beta`, and task-by-ADR travel matrices `TL`, `TR`.
Task and ADR indices in arrays are zero-based. Travel times are generated input
parameters, not routes calculated from warehouse geometry.
