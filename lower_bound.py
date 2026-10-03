"""Calculate the benchmark's adjusted average and single-task lower bounds.

This entry point uses the same implementation as every experiment runner, rather
than maintaining a separate historical formula that could produce different GAPs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_benchmark_experiments import benchmark_common as benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="n20_m2", help="One canonical case, e.g. n150_m5.")
    parser.add_argument("--instance", type=Path, help="Alternatively read exported instance JSON.")
    args = parser.parse_args()
    if args.instance is not None:
        raw = json.loads(args.instance.read_text(encoding="utf-8-sig"))
    else:
        matches = [(index, case) for index, case in enumerate(benchmark.BENCHMARK_CASES)
                   if case.name == args.case]
        if not matches:
            parser.error(f"Unknown benchmark case: {args.case}")
        index, case = matches[0]
        raw = benchmark.build_raw_instance(case, benchmark.case_instance_seed(index))

    # Preparing coefficients and evaluating the bound does not invoke a solver.
    bounds = benchmark.compute_lower_bound(benchmark.prepare_common_data(raw))
    print(json.dumps(bounds, indent=2))


if __name__ == "__main__":
    main()
