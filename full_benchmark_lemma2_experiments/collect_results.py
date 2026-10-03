"""Combine only the new Lemma 2 results, never the original benchmark rows."""

from __future__ import annotations

import csv

from experiment_common import ALGORITHMS, RUN_FIELDS, argument_parser, benchmark, result_root


def main() -> None:
    args = argument_parser("Collect the four Lemma 2-enabled algorithm results.").parse_args()
    root = result_root(args.experiment_id)
    rows = []
    for algorithm in ALGORITHMS:
        path = root / benchmark.safe_name(algorithm) / "run_details.csv"
        if not path.exists():
            print(f"Missing results: {algorithm}")
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            rows.extend(csv.DictReader(file))
    if not rows:
        raise FileNotFoundError(f"No Lemma 2 run results under {root}")
    output = root / "summary"
    benchmark.write_csv(output / "all_run_details.csv", rows, RUN_FIELDS)
    benchmark.write_csv(
        output / "all_average_gap_summary.csv", benchmark.summarize_rows(rows), benchmark.SUMMARY_FIELDS,
    )
    print(f"Collected {len(rows)} runs. Saved: {output}")


if __name__ == "__main__":
    main()
