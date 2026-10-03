"""
Extract the best run among 10 replications for each algorithm and case.

The script reads existing files:

    full_benchmark_results/<experiment_id>/<algorithm>/run_details.csv

For each case and each of the six heuristic algorithms, it keeps the row with
the minimum Cmax and saves:

    full_benchmark_results/<experiment_id>/summary/best_runs_by_case_algorithm.csv

No algorithm is rerun by this script.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from benchmark_common import DEFAULT_EXPERIMENT_ID, HEURISTIC_ALGORITHMS, result_root, safe_name, write_csv


BEST_RUN_FIELDS = [
    "case",
    "scale",
    "n",
    "m",
    "lambda",
    "algorithm",
    "best_replication",
    "instance_seed",
    "search_seed",
    "time_limit",
    "lb",
    "best_cmax",
    "best_gap_percent",
    "status",
    "elapsed",
    "iterations",
    "task_counts",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def valid_cmax(row: dict[str, Any]) -> bool:
    text = str(row.get("cmax", "")).strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def best_row_for_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if valid_cmax(row)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["cmax"]))


def convert_best_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": row.get("case", ""),
        "scale": row.get("scale", ""),
        "n": row.get("n", ""),
        "m": row.get("m", ""),
        "lambda": row.get("lambda", ""),
        "algorithm": row.get("algorithm", ""),
        "best_replication": row.get("replication", ""),
        "instance_seed": row.get("instance_seed", ""),
        "search_seed": row.get("search_seed", ""),
        "time_limit": row.get("time_limit", ""),
        "lb": row.get("lb", ""),
        "best_cmax": row.get("cmax", ""),
        "best_gap_percent": row.get("gap_percent", ""),
        "status": row.get("status", ""),
        "elapsed": row.get("elapsed", ""),
        "iterations": row.get("iterations", ""),
        "task_counts": row.get("task_counts", ""),
    }


def extract_best_runs(experiment_id: str) -> list[dict[str, Any]]:
    root = result_root(experiment_id)
    best_rows: list[dict[str, Any]] = []

    for algorithm in HEURISTIC_ALGORITHMS:
        details_path = root / safe_name(algorithm) / "run_details.csv"
        rows = read_csv(details_path)
        if not rows:
            print(f"[warn] Missing or empty file: {details_path}")
            continue

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row.get("case", ""), []).append(row)

        for case_name in sorted(groups):
            best = best_row_for_group(groups[case_name])
            if best is not None:
                best_rows.append(convert_best_row(best))

    best_rows.sort(
        key=lambda row: (
            str(row["scale"]),
            int(row["n"]) if str(row["n"]).isdigit() else 0,
            int(row["m"]) if str(row["m"]).isdigit() else 0,
            HEURISTIC_ALGORITHMS.index(row["algorithm"])
            if row["algorithm"] in HEURISTIC_ALGORITHMS
            else 999,
        )
    )
    return best_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the best replication for each algorithm and benchmark case."
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = result_root(args.experiment_id)
    if not root.exists():
        raise FileNotFoundError(f"Experiment result folder not found: {root}")

    best_rows = extract_best_runs(args.experiment_id)
    output_path = root / "summary" / "best_runs_by_case_algorithm.csv"
    write_csv(output_path, best_rows, BEST_RUN_FIELDS)

    print("========== Best Runs Extracted ==========")
    print(f"Experiment: {args.experiment_id}")
    print(f"Rows: {len(best_rows)}")
    print(output_path)


if __name__ == "__main__":
    main()
