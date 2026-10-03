"""
Plot Nemenyi post-hoc significance heatmaps for four benchmark-size groups.

Input:
    full_benchmark_results/<experiment_id>/<algorithm>/run_details.csv

The script builds complete Friedman/Nemenyi blocks from each (case, replication)
pair, where columns are algorithms and values are Cmax. Four heatmaps are
generated for task-size ranges:

    20-80, 100-200, 225-300, 350-500

Output:
    full_benchmark_results/<experiment_id>/figures/nemenyi_heatmap_<range>.png
    full_benchmark_results/<experiment_id>/figures/nemenyi_heatmap_<range>.pdf
    full_benchmark_results/<experiment_id>/summary/nemenyi_p_values_<range>.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark_common import DEFAULT_EXPERIMENT_ID, HEURISTIC_ALGORITHMS, result_root, safe_name


SIZE_GROUPS = [
    ("20_80", 20, 80, "20-80"),
    ("100_200", 100, 200, "100-200"),
    ("225_300", 225, 300, "225-300"),
    ("350_500", 350, 500, "350-500"),
]

METHODS = ["VNS-HE", "VNS", "SA", "TS", "GA-SA", "PSO-SA"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def require_posthoc_nemenyi():
    try:
        from scikit_posthocs import posthoc_nemenyi_friedman
    except ImportError as exc:
        raise ImportError(
            "Package scikit-posthocs is required for Nemenyi p-values. "
            "Install it with: pip install scikit-posthocs"
        ) from exc
    return posthoc_nemenyi_friedman


def parse_case_n(case_name: str) -> int:
    # Expected form: n150_m5
    first = case_name.split("_", 1)[0]
    if not first.startswith("n"):
        raise ValueError(f"Invalid case name: {case_name}")
    return int(first[1:])


def load_all_runs(experiment_id: str, methods: list[str]) -> pd.DataFrame:
    root = result_root(experiment_id)
    frames = []

    for method in methods:
        path = root / safe_name(method) / "run_details.csv"
        rows = read_csv(path)
        if not rows:
            print(f"Warning: missing or empty result file: {path}")
            continue
        data = pd.DataFrame(rows)
        required = {"case", "replication", "cmax"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        data = data[["case", "replication", "cmax"]].copy()
        data["method"] = method
        data["cmax"] = pd.to_numeric(data["cmax"], errors="coerce")
        data["replication"] = pd.to_numeric(data["replication"], errors="coerce")
        data = data.dropna(subset=["case", "replication", "cmax"])
        data["replication"] = data["replication"].astype(int)
        data["n"] = data["case"].astype(str).map(parse_case_n)
        data["block_id"] = data["case"].astype(str) + "_rep" + data["replication"].astype(str)
        frames.append(data)

    if not frames:
        raise ValueError("No valid run rows were found.")
    return pd.concat(frames, ignore_index=True)


def build_group_table(
    data: pd.DataFrame,
    n_min: int,
    n_max: int,
    methods: list[str],
) -> pd.DataFrame:
    filtered = data[(data["n"] >= n_min) & (data["n"] <= n_max)].copy()
    table = filtered.pivot_table(
        index="block_id",
        columns="method",
        values="cmax",
        aggfunc="mean",
    )
    table = table.reindex(columns=methods).dropna()
    if table.empty:
        raise ValueError(f"No complete blocks found for n in [{n_min}, {n_max}].")
    return table


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "axes.linewidth": 1.15,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_nemenyi_heatmap(
    p_values: pd.DataFrame,
    group_label: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(145, 300, s=85, l=55, as_cmap=True)

    sns.heatmap(
        p_values,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "p-value"},
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18, width=1.4, length=4.8, colors="#1A1A1A")
    cbar.set_label("p-value", fontsize=22, labelpad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("#1A1A1A")

    methods = list(p_values.columns)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            val = p_values.iloc[i, j]
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=19,
                color="black",
            )

    for i in range(len(methods)):
        for j in range(i):
            p = p_values.iloc[i, j]
            if p <= 0.01:
                radius = 0.30
                stars = "***"
                color = "darkorange"
            elif p <= 0.05:
                radius = 0.22
                stars = "**"
                color = "orange"
            else:
                radius = 0.15
                stars = ""
                color = "lightgrey"

            circle = plt.Circle((j + 0.5, i + 0.5), radius, color=color, alpha=0.65)
            ax.add_patch(circle)
            if stars:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    stars,
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="black",
                )

    for i in range(len(methods)):
        ax.add_patch(
            plt.Rectangle((i, i), 1, 1, facecolor="lightgray", edgecolor="white")
        )

    ax.set_xticks(np.arange(len(methods)) + 0.5)
    ax.set_yticks(np.arange(len(methods)) + 0.5)
    ax.set_xticklabels(
        methods,
        rotation=45,
        ha="right",
        fontsize=22,
        fontweight="bold",
        color="#1A1A1A",
    )
    ax.set_yticklabels(methods, fontsize=22, fontweight="bold", color="#1A1A1A")
    ax.tick_params(axis="both", width=1.5, length=5.0, colors="#1A1A1A")
    ax.invert_yaxis()

    ax.set_xlabel("Algorithm", fontsize=24, fontweight="bold")
    ax.set_ylabel("Algorithm", fontsize=24, fontweight="bold")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_methods(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Nemenyi post-hoc heatmaps for benchmark-size groups."
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--methods", default=",".join(METHODS))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    methods = parse_methods(args.methods)
    posthoc_nemenyi_friedman = require_posthoc_nemenyi()

    data = load_all_runs(args.experiment_id, methods)
    root = result_root(args.experiment_id)
    figures_dir = root / "figures"
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for group_key, n_min, n_max, group_label in SIZE_GROUPS:
        table = build_group_table(data, n_min, n_max, methods)
        p_values = posthoc_nemenyi_friedman(table)
        p_values = p_values.reindex(index=methods, columns=methods)

        p_path = summary_dir / f"nemenyi_p_values_{group_key}.csv"
        p_values.to_csv(p_path, encoding="utf-8-sig")

        output_png = figures_dir / f"nemenyi_heatmap_{group_key}.png"
        output_pdf = figures_dir / f"nemenyi_heatmap_{group_key}.pdf"
        draw_nemenyi_heatmap(p_values, group_label, output_png, output_pdf)

        print(
            f"[{group_label}] complete blocks={table.shape[0]}, "
            f"saved: {output_png}"
        )


if __name__ == "__main__":
    main()
