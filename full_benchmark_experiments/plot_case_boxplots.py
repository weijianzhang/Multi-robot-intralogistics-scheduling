"""
Plot repeated-run line charts for selected benchmark cases.

The script reads:

    full_benchmark_results/<experiment_id>/<algorithm>/run_details.csv

and plots the six algorithms on the following six cases:

    40-2, 80-3, 150-5, 250-7, 400-9, 500-11

For each case, one figure is generated. The x-axis is the replication index
(1 to 10), and the y-axis is the selected metric, Cmax by default.
The output is saved to:

    full_benchmark_results/<experiment_id>/figures/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from benchmark_common import DEFAULT_EXPERIMENT_ID, HEURISTIC_ALGORITHMS, result_root, safe_name


CASES_TO_PLOT = ["n40_m2", "n80_m3", "n150_m5", "n250_m7", "n400_m9", "n500_m11"]
CASE_LABELS = {
    "n40_m2": "40-2",
    "n80_m3": "80-3",
    "n150_m5": "150-5",
    "n250_m7": "250-7",
    "n400_m9": "400-9",
    "n500_m11": "500-11",
}

ALGORITHM_COLORS = {
    "VNS": "#005AB5",
    "VNS-HE": "#DC3220",
    "SA": "#009E73",
    "TS": "#7F3C8D",
    "GA-SA": "#E69F00",
    "PSO-SA": "#000000",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def load_case_algorithm_values(
    experiment_id: str,
    cases: list[str],
    algorithms: list[str],
    metric: str,
) -> dict[str, dict[str, dict[int, float]]]:
    root = result_root(experiment_id)
    values = {case: {algorithm: {} for algorithm in algorithms} for case in cases}

    for algorithm in algorithms:
        details_path = root / safe_name(algorithm) / "run_details.csv"
        rows = read_csv(details_path)
        if not rows:
            print(f"[warn] Missing or empty file: {details_path}")
            continue
        for row in rows:
            case = row.get("case", "")
            if case not in values:
                continue
            parsed = parse_float(row.get(metric, ""))
            replication = parse_int(row.get("replication", ""))
            if parsed is not None and replication is not None:
                values[case][algorithm][replication] = parsed
    return values


def set_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 24,
            "axes.labelsize": 26,
            "axes.titlesize": 26,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "axes.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
        }
    )


def draw_line_plots(
    values: dict[str, dict[str, dict[int, float]]],
    cases: list[str],
    algorithms: list[str],
    metric: str,
    output_dir: Path,
) -> list[tuple[Path, Path]]:
    set_academic_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[tuple[Path, Path]] = []

    for case in cases:
        fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
        for index, algorithm in enumerate(algorithms):
            series = values[case][algorithm]
            if not series:
                print(f"[warn] No data for {case} - {algorithm}")
                continue
            x_values = sorted(series)
            y_values = [series[rep] for rep in x_values]
            ax.plot(
                x_values,
                y_values,
                label=algorithm,
                color=ALGORITHM_COLORS.get(algorithm, "#333333"),
                marker="s",
                linewidth=2.2,
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )

        ax.set_xlabel("Instance", fontweight="bold")
        ylabel = r"$C_{\max}$" if metric == "cmax" else "GAP (%)"
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xlim(0.7, 10.3)
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.45)
        ax.grid(axis="x", linestyle=":", linewidth=0.75, alpha=0.28)
        ax.tick_params(axis="both", direction="in", length=4.5, width=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.35)
        ax.legend(frameon=False, ncol=2, loc="best")

        run_counts = [len(values[case][algorithm]) for algorithm in algorithms]
        if any(count != 10 for count in run_counts):
            count_text = ", ".join(
                f"{algorithm}:{count}"
                for algorithm, count in zip(algorithms, run_counts)
                if count != 10
            )
            ax.text(
                0.02,
                0.97,
                f"runs {count_text}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                color="#555555",
            )

        case_file = CASE_LABELS.get(case, case).replace("-", "_")
        png_path = output_dir / f"{case_file}_{metric}_replication_lines.png"
        pdf_path = output_dir / f"{case_file}_{metric}_replication_lines.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append((png_path, pdf_path))

    return output_paths


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot repeated-run line charts for selected benchmark cases."
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--cases",
        default=",".join(CASES_TO_PLOT),
        help="Comma-separated case names, e.g. n40_m2,n60_m3.",
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(HEURISTIC_ALGORITHMS),
        help="Comma-separated algorithm names.",
    )
    parser.add_argument(
        "--metric",
        choices=["cmax", "gap_percent"],
        default="cmax",
        help="Metric used for line charts.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = parse_list(args.cases)
    algorithms = parse_list(args.algorithms)
    values = load_case_algorithm_values(args.experiment_id, cases, algorithms, args.metric)
    output_dir = result_root(args.experiment_id) / "figures"
    output_paths = draw_line_plots(values, cases, algorithms, args.metric, output_dir)

    print("========== Line Plots Saved ==========")
    for png_path, pdf_path in output_paths:
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
