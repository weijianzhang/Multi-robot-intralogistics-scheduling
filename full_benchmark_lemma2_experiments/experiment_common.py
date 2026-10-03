"""Reuse instance generation, lower bounds and CSV schemas without old outputs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_lemma2_benchmark_common", PROJECT_ROOT / "full_benchmark_experiments" / "benchmark_common.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load the existing benchmark utilities.")
benchmark = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = benchmark
_spec.loader.exec_module(benchmark)

ALGORITHMS = ["SA", "TS", "GA-SA", "PSO-SA"]
DEFAULT_EXPERIMENT_ID = "full_benchmark_36cases_lemma2"
RESULTS_ROOT = PROJECT_ROOT / "full_benchmark_lemma2_results"
RUN_FIELDS = benchmark.RUN_DETAIL_FIELDS + ["decoder", "initial_cmax", "initial_elapsed"]
HISTORY_FIELDS = [
    "case", "algorithm", "replication", "instance_seed", "search_seed", "lb",
    "iteration", "elapsed", "candidate_cmax", "current_cmax", "best_cmax", "event",
]


def result_root(experiment_id: str) -> Path:
    if not experiment_id or experiment_id in (".", "..") or "/" in experiment_id or "\\" in experiment_id:
        raise ValueError("Experiment ID must be a single directory name.")
    return RESULTS_ROOT / experiment_id


def argument_parser(description: str):
    parser = benchmark.base_arg_parser(description)
    parser.set_defaults(experiment_id=DEFAULT_EXPERIMENT_ID)
    return parser
