"""Collect original-method per-run results and case-wise summaries without rerunning solvers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark_common import (
    ALL_ALGORITHMS,
    PROJECT_ROOT,
    SUMMARY_FIELDS,
    RUN_DETAIL_FIELDS,
    safe_name,
    write_csv,
)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect separated benchmark run results.")
    parser.add_argument("--experiment-id", default="full_benchmark_36cases")
    args = parser.parse_args()

    root = PROJECT_ROOT / "full_benchmark_results" / args.experiment_id
    summary_dir = root / "summary"
    all_details = []
    all_summaries = []

    for algorithm in ALL_ALGORITHMS:
        alg_dir = root / safe_name(algorithm)
        all_details.extend(read_csv(alg_dir / "run_details.csv"))
        all_summaries.extend(read_csv(alg_dir / "average_gap_summary.csv"))

    if all_details:
        write_csv(summary_dir / "all_run_details.csv", all_details, RUN_DETAIL_FIELDS)
    if all_summaries:
        write_csv(summary_dir / "all_average_gap_summary.csv", all_summaries, SUMMARY_FIELDS)

    print(f"Collected details: {len(all_details)} rows")
    print(f"Collected summaries: {len(all_summaries)} rows")
    print(summary_dir)


if __name__ == "__main__":
    main()
