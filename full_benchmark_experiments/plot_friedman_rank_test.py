"""
Draw Friedman rank-frequency plots for the six algorithms.

Input:
    full_benchmark_results/<experiment_id>/<algorithm>/run_details.csv

By default, each (case, replication) pair is treated as one complete block for
Friedman ranking. The objective is Cmax, and smaller values receive better ranks.

Output:
    full_benchmark_results/<experiment_id>/figures/friedman_rank_test.png
    full_benchmark_results/<experiment_id>/figures/friedman_rank_test.pdf
    full_benchmark_results/<experiment_id>/summary/friedman_mean_ranks.csv
"""

from __future__ import annotations

import argparse
import csv
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_common import DEFAULT_EXPERIMENT_ID, HEURISTIC_ALGORITHMS, result_root, safe_name


METHOD_COL = "Algorithm"
OBJECTIVE_COL = "Cmax"

COLOR_MAP = {
    "VNS": "#3B4992",
    "VNS-HE": "#EE0000",
    "SA": "#008B45",
    "TS": "#631879",
    "GA-SA": "#FFA500",
    "PSO-SA": "#00A1D5",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_algorithm_rows(experiment_id: str, algorithms: list[str]) -> pd.DataFrame:
    root = result_root(experiment_id)
    frames = []

    for algorithm in algorithms:
        path = root / safe_name(algorithm) / "run_details.csv"
        rows = read_csv(path)
        if not rows:
            print(f"Warning: missing or empty result file: {path}")
            continue

        data = pd.DataFrame(rows)
        required_cols = {"case", "replication", "cmax"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"{path} is missing columns: {sorted(missing_cols)}")

        data = data[["case", "replication", "cmax"]].copy()
        data[METHOD_COL] = algorithm
        data[OBJECTIVE_COL] = pd.to_numeric(data["cmax"], errors="coerce")
        data["replication"] = pd.to_numeric(data["replication"], errors="coerce")
        data = data.dropna(subset=[OBJECTIVE_COL, "replication"])
        data["replication"] = data["replication"].astype(int)
        frames.append(data[["case", "replication", METHOD_COL, OBJECTIVE_COL]])

    if not frames:
        raise ValueError("No valid algorithm result rows were found.")

    return pd.concat(frames, ignore_index=True)


def build_rank_table(
    data: pd.DataFrame,
    algorithms: list[str],
    block_level: str,
) -> pd.DataFrame:
    if block_level == "case_average":
        grouped = (
            data.groupby(["case", METHOD_COL], as_index=False)[OBJECTIVE_COL]
            .mean()
            .rename(columns={"case": "problem_id"})
        )
        table = grouped.pivot_table(
            index="problem_id",
            columns=METHOD_COL,
            values=OBJECTIVE_COL,
            aggfunc="mean",
        )
    else:
        data = data.copy()
        data["problem_id"] = data["case"].astype(str) + "_rep" + data["replication"].astype(str)
        table = data.pivot_table(
            index="problem_id",
            columns=METHOD_COL,
            values=OBJECTIVE_COL,
            aggfunc="mean",
        )

    table = table.reindex(columns=algorithms).dropna()
    if table.empty:
        method_counts = data.groupby(METHOD_COL)["case"].nunique()
        raise ValueError(
            "No complete Friedman blocks were found across the selected algorithms.\n"
            "Completed case counts by algorithm:\n"
            f"{method_counts.to_string()}"
        )
    return table


def print_friedman_summary(ranks: pd.DataFrame) -> pd.Series:
    mean_ranks = ranks.mean(axis=0).sort_values()

    print("=== Mean rank of each algorithm (lower is better) ===")
    for method, rank in mean_ranks.items():
        print(f"{method:10s}: {rank:.3f}")

    n_problems = ranks.shape[0]
    n_methods = ranks.shape[1]
    rank_sums = ranks.sum(axis=0)
    chi2_f = (
        12.0
        / (n_problems * n_methods * (n_methods + 1))
        * np.sum(rank_sums**2)
        - 3 * n_problems * (n_methods + 1)
    )

    print(f"\nNumber of complete blocks: {n_problems}")
    print(f"Friedman chi-square statistic: {chi2_f:.4f} (df = {n_methods - 1})")

    try:
        from scipy.stats import chi2

        p_value = 1.0 - chi2.cdf(chi2_f, df=n_methods - 1)
        print(f"Approximate p-value (chi-square): {p_value:.4e}")
    except ImportError:
        print("SciPy not installed: p-value not computed.")

    return mean_ranks


def build_rank_frequencies(ranks: pd.DataFrame, algorithms: list[str]) -> dict[str, np.ndarray]:
    n_methods = ranks.shape[1]
    rank_levels = np.arange(1, n_methods + 1)
    freq_dict = {}
    for algorithm in algorithms:
        rounded_ranks = ranks[algorithm].round().astype(int)
        freq_dict[algorithm] = np.array(
            [(rounded_ranks == rank).sum() for rank in rank_levels]
        )
    return freq_dict


def write_mean_ranks(
    mean_ranks: pd.Series,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"algorithm": algorithm, "mean_rank": float(rank)}
        for algorithm, rank in mean_ranks.items()
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["algorithm", "mean_rank"])
        writer.writeheader()
        writer.writerows(rows)


def set_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.titlesize": 19,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def plot_rank_frequencies(
    freq_dict: dict[str, np.ndarray],
    mean_ranks: pd.Series,
    algorithms: list[str],
    output_png: Path,
    output_pdf: Path,
) -> None:
    set_academic_style()
    n_methods = len(algorithms)
    rank_levels = np.arange(1, n_methods + 1)
    n_cols = 3 if n_methods > 3 else n_methods
    n_rows = ceil(n_methods / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.8 * n_rows),
        sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx]
        ax.barh(
            rank_levels,
            freq_dict[algorithm],
            color=COLOR_MAP.get(algorithm, "#4D4D4D"),
            edgecolor="#2F2F2F",
            linewidth=0.85,
        )
        ax.set_ylim(0.5, n_methods + 0.5)
        ax.set_yticks(rank_levels)
        ax.invert_yaxis()

        if idx % n_cols == 0:
            ax.set_ylabel("Rank")
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel("Frequency")

        ax.set_title(
            f"{algorithm}\nMean rank = {mean_ranks[algorithm]:.2f}",
            pad=6,
        )
        ax.tick_params(axis="both", width=1.3, length=5, direction="in")
        ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.grid(axis="y", visible=False)

        for spine in ax.spines.values():
            spine.set_linewidth(1.15)

    for ax in axes[n_methods:]:
        ax.axis("off")

    fig.suptitle(
        "Friedman Two-Way Analysis of Variance by Ranks",
        fontsize=24,
        y=0.985,
    )
    plt.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.86,
        bottom=0.13,
        wspace=0.20,
        hspace=0.52,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_algorithms(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Friedman rank-frequency comparison for the six algorithms."
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--algorithms",
        default=",".join(HEURISTIC_ALGORITHMS),
        help="Comma-separated algorithm names.",
    )
    parser.add_argument(
        "--block-level",
        choices=["case_replication", "case_average"],
        default="case_replication",
        help="Use each case-replication pair or each case average as a Friedman block.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    algorithms = parse_algorithms(args.algorithms)
    data = load_algorithm_rows(args.experiment_id, algorithms)
    table = build_rank_table(data, algorithms, args.block_level)

    ranks = table.rank(axis=1, method="average", ascending=True)
    mean_ranks = print_friedman_summary(ranks)
    freq_dict = build_rank_frequencies(ranks, algorithms)

    root = result_root(args.experiment_id)
    figures_dir = root / "figures"
    summary_dir = root / "summary"
    suffix = args.block_level
    output_png = figures_dir / f"friedman_rank_test_{suffix}.png"
    output_pdf = figures_dir / f"friedman_rank_test_{suffix}.pdf"
    mean_rank_path = summary_dir / f"friedman_mean_ranks_{suffix}.csv"

    write_mean_ranks(mean_ranks, mean_rank_path)
    plot_rank_frequencies(freq_dict, mean_ranks, algorithms, output_png, output_pdf)

    print("\nSaved files:")
    print(output_png)
    print(output_pdf)
    print(mean_rank_path)


if __name__ == "__main__":
    main()
