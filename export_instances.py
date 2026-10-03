"""Export deterministic benchmark inputs and lower bounds without running solvers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from full_benchmark_experiments import benchmark_common as benchmark


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="all", help="Case names, e.g. n20_m2,n60_m3.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cases = benchmark.selected_cases(args.cases)
    if args.cases.lower() != "all":
        wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
        unknown = wanted - {case.name for _, case in cases}
        if unknown:
            parser.error(f"Unknown benchmark cases: {sorted(unknown)}")

    output = args.output_dir.resolve()
    paths = [output / "instances" / f"{case.name}.json" for _, case in cases]
    manifest = output / "benchmark_manifest.csv"
    existing = [path for path in [*paths, manifest] if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"Output already exists: {existing[0]}. Use a new directory or --overwrite.")

    rows = []
    instances = []
    for index, case in cases:
        seed = benchmark.case_instance_seed(index)
        raw = benchmark.build_raw_instance(case, seed)
        bounds = benchmark.compute_lower_bound(benchmark.prepare_common_data(raw))
        rows.append({
            "case": case.name, "scale": case.scale, "n": case.n, "m": case.m,
            "lambda": case.lambd, "instance_seed": seed,
            "heuristic_seconds": benchmark.heuristic_time_limit(
                case, benchmark.SMALL_TIME_LIMIT, benchmark.MEDIUM_TIME_LIMIT,
                benchmark.LARGE_TIME_LIMIT,
            ),
            "gurobi_seconds": benchmark.GUROBI_TIME_LIMIT,
            "replications": benchmark.DEFAULT_REPLICATIONS, **bounds,
        })
        instances.append(raw)

    (output / "instances").mkdir(parents=True, exist_ok=True)
    for path, raw in zip(paths, instances):
        path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(instances)} instances and their lower bounds to {output}")


if __name__ == "__main__":
    main()
