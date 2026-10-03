"""
Shared utilities for the 36-case full benchmark experiment.

Each run script imports this module, generates the same deterministic instances,
computes the same lower bound, and writes its own result files under:

    full_benchmark_results/<experiment_id>/<algorithm>/
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ID = "full_benchmark_36cases"
BASE_SEED = 20260722
DEFAULT_REPLICATIONS = 10

SMALL_TIME_LIMIT = 60.0
MEDIUM_TIME_LIMIT = 90.0
LARGE_TIME_LIMIT = 120.0
GUROBI_TIME_LIMIT = 1800.0

T0_RANGE = (0.0, 2.0)
L_RANGE = (0.2, 2.0)
U_RANGE = (0.2, 2.0)
W_RANGE = (0.5, 3.0)
Q_RANGE = (6.0, 8.0)
TL_RANGE = (2.0, 5.0)
TR_RANGE = (1.5, 3.5)
BETA_RANGE = (0.01, 0.03)

HEURISTIC_ALGORITHMS = ["VNS", "VNS-HE", "SA", "TS", "GA-SA", "PSO-SA"]
ALL_ALGORITHMS = HEURISTIC_ALGORITHMS + ["Gurobi"]


@dataclass(frozen=True)
class BenchmarkCase:
    scale: str
    n: int
    m: int
    lambd: float

    @property
    def name(self) -> str:
        return f"n{self.n}_m{self.m}"


# Selected configurations, not a Cartesian product: 12 cases per scale.
BENCHMARK_CASES = [
    BenchmarkCase("small", 20, 2, 0.20),
    BenchmarkCase("small", 20, 3, 0.20),
    BenchmarkCase("small", 40, 2, 0.20),
    BenchmarkCase("small", 40, 3, 0.20),
    BenchmarkCase("small", 40, 5, 0.20),
    BenchmarkCase("small", 60, 2, 0.20),
    BenchmarkCase("small", 60, 3, 0.20),
    BenchmarkCase("small", 60, 5, 0.20),
    BenchmarkCase("small", 80, 3, 0.20),
    BenchmarkCase("small", 80, 5, 0.20),
    BenchmarkCase("small", 100, 3, 0.20),
    BenchmarkCase("small", 100, 5, 0.20),
    BenchmarkCase("medium", 150, 5, 0.25),
    BenchmarkCase("medium", 150, 7, 0.25),
    BenchmarkCase("medium", 175, 5, 0.25),
    BenchmarkCase("medium", 175, 7, 0.25),
    BenchmarkCase("medium", 200, 5, 0.25),
    BenchmarkCase("medium", 200, 7, 0.25),
    BenchmarkCase("medium", 200, 9, 0.25),
    BenchmarkCase("medium", 225, 5, 0.25),
    BenchmarkCase("medium", 225, 7, 0.25),
    BenchmarkCase("medium", 225, 9, 0.25),
    BenchmarkCase("medium", 250, 7, 0.25),
    BenchmarkCase("medium", 250, 9, 0.25),
    BenchmarkCase("large", 300, 9, 0.30),
    BenchmarkCase("large", 300, 11, 0.30),
    BenchmarkCase("large", 350, 9, 0.30),
    BenchmarkCase("large", 350, 11, 0.30),
    BenchmarkCase("large", 400, 9, 0.30),
    BenchmarkCase("large", 400, 11, 0.30),
    BenchmarkCase("large", 400, 13, 0.30),
    BenchmarkCase("large", 450, 9, 0.30),
    BenchmarkCase("large", 450, 11, 0.30),
    BenchmarkCase("large", 450, 13, 0.30),
    BenchmarkCase("large", 500, 11, 0.30),
    BenchmarkCase("large", 500, 13, 0.30),
]


RUN_DETAIL_FIELDS = [
    "case",
    "scale",
    "n",
    "m",
    "lambda",
    "algorithm",
    "replication",
    "instance_seed",
    "search_seed",
    "time_limit",
    "lb",
    "cmax",
    "gap_percent",
    "status",
    "elapsed",
    "iterations",
    "task_counts",
    "gurobi_status",
    "gurobi_bound",
    "gurobi_mip_gap",
]

SUMMARY_FIELDS = [
    "case",
    "scale",
    "n",
    "m",
    "lambda",
    "algorithm",
    "replications",
    "lb",
    "avg_cmax",
    "std_cmax",
    "best_cmax",
    "worst_cmax",
    "avg_gap_percent",
    "best_gap_percent",
    "worst_gap_percent",
]

LOWER_BOUND_FIELDS = [
    "case",
    "scale",
    "n",
    "m",
    "lambda",
    "instance_seed",
    "q_relaxed",
    "phi",
    "lb_avg",
    "lb_task",
    "lb",
]


def safe_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def uniform_list(rng: random.Random, size: int, lo: float, hi: float) -> list[float]:
    return [round(rng.uniform(lo, hi), 4) for _ in range(size)]


def uniform_matrix(
    rng: random.Random, rows: int, cols: int, lo: float, hi: float
) -> list[list[float]]:
    return [[round(rng.uniform(lo, hi), 4) for _ in range(cols)] for _ in range(rows)]


def case_instance_seed(case_index: int) -> int:
    return BASE_SEED + (case_index + 1) * 100000


def algorithm_seed_offset(algorithm: str) -> int:
    if algorithm in HEURISTIC_ALGORITHMS:
        return HEURISTIC_ALGORITHMS.index(algorithm) * 100
    return 0


def search_seed(case_index: int, replication: int, algorithm: str) -> int:
    return case_instance_seed(case_index) + replication * 1000 + algorithm_seed_offset(algorithm)


def build_raw_instance(case: BenchmarkCase, seed: int) -> dict[str, Any]:
    """Generate the same task/ADR data for all methods and search repetitions."""
    rng = random.Random(seed)
    return {
        "n": case.n,
        "m": case.m,
        "seed": seed,
        "t0": round(rng.uniform(*T0_RANGE), 4),
        "lambda": case.lambd,
        "L": uniform_list(rng, case.n, *L_RANGE),
        "U": uniform_list(rng, case.n, *U_RANGE),
        "w": uniform_list(rng, case.n, *W_RANGE),
        "Q": uniform_list(rng, case.m, *Q_RANGE),
        "TL": uniform_matrix(rng, case.n, case.m, *TL_RANGE),
        "TR": uniform_matrix(rng, case.n, case.m, *TR_RANGE),
        "beta": uniform_list(rng, case.m, *BETA_RANGE),
    }


def prepare_common_data(raw_data: dict[str, Any]) -> dict[str, Any]:
    """Compute model coefficients on a copy so solvers cannot alter shared inputs."""
    data = copy.deepcopy(raw_data)
    n = data["n"]
    m = data["m"]
    gamma = [[0.0 for _ in range(m)] for _ in range(n)]
    q = [[0.0 for _ in range(m)] for _ in range(n)]
    alpha = [[0.0 for _ in range(m)] for _ in range(n)]
    rho = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for k in range(m):
            gamma[i][k] = data["beta"][k] * (
                1.0 + data["lambda"] * data["w"][i] / data["Q"][k]
            )
            q[i][k] = (1.0 + gamma[i][k]) ** 2
            alpha[i][k] = (
                (1.0 + gamma[i][k]) ** 2 * data["L"][i]
                + (1.0 + gamma[i][k]) * (data["TL"][i][k] + data["U"][i])
                + data["TR"][i][k]
            )
            rho[i][k] = alpha[i][k] / max(q[i][k] - 1.0, 1e-12)

    data["gamma"] = gamma
    data["q"] = q
    data["alpha"] = alpha
    data["rho"] = rho
    return data


def compute_lower_bound(data: dict[str, Any]) -> dict[str, float]:
    """Combine the adjusted common-factor relaxation and the single-task bound."""
    n = data["n"]
    m = data["m"]
    feasible = [
        [k for k in range(m) if data["w"][i] <= data["Q"][k] + 1e-9]
        for i in range(n)
    ]
    infeasible_jobs = [i + 1 for i, candidates in enumerate(feasible) if not candidates]
    if infeasible_jobs:
        raise ValueError(f"Jobs without feasible AGV: {infeasible_jobs}")

    q_relaxed = min(data["q"][i][k] for i in range(n) for k in feasible[i])
    # S >= t0 lets the intercept absorb (q_ik - q_relaxed) * t0 safely.
    alpha_bar = [
        min(
            data["alpha"][i][k] + (data["q"][i][k] - q_relaxed) * data["t0"]
            for k in feasible[i]
        )
        for i in range(n)
    ]
    alpha_desc = sorted(alpha_bar, reverse=True)

    ell = n // m
    r0 = n - ell * m
    if r0 == 0:
        phi = m * (q_relaxed**ell) * data["t0"]
    else:
        phi = (q_relaxed**ell) * data["t0"] * (m + r0 * (q_relaxed - 1.0))

    # Larger relaxed contributions receive smaller positional weights globally.
    contribution = sum(
        alpha_desc[i - 1] * (q_relaxed ** (math.ceil(i / m) - 1))
        for i in range(1, n + 1)
    )
    lb_avg = (phi + contribution) / m
    lb_task = max(
        min(data["q"][i][k] * data["t0"] + data["alpha"][i][k] for k in feasible[i])
        for i in range(n)
    )
    lb = max(lb_avg, lb_task)

    return {
        "q_relaxed": q_relaxed,
        "phi": phi,
        "lb_avg": lb_avg,
        "lb_task": lb_task,
        "lb": lb,
    }


def selected_cases(case_filter: str) -> list[tuple[int, BenchmarkCase]]:
    if not case_filter or case_filter.lower() == "all":
        wanted = None
    else:
        wanted = {item.strip() for item in case_filter.split(",") if item.strip()}
    cases = [
        (idx, case)
        for idx, case in enumerate(BENCHMARK_CASES)
        if wanted is None or case.name in wanted
    ]
    if not cases:
        raise ValueError("No benchmark cases selected.")
    return cases


def heuristic_time_limit(
    case: BenchmarkCase, small_time: float, medium_time: float, large_time: float
) -> float:
    if case.scale == "small":
        return small_time
    if case.scale == "medium":
        return medium_time
    return large_time


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def configure_algorithm_module(module: ModuleType, case: BenchmarkCase, time_limit: float) -> None:
    for name, value in {
        "N": case.n,
        "M": case.m,
        "LAMBDA": case.lambd,
        "TIME_LIMIT": time_limit,
    }.items():
        if hasattr(module, name):
            setattr(module, name, value)


def prepare_for_module(module: ModuleType, raw_data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(raw_data)
    if hasattr(module, "prepare_data"):
        return module.prepare_data(data)
    return prepare_common_data(data)


def quiet_context(enabled: bool):
    if enabled:
        return contextlib.redirect_stdout(io.StringIO())
    return contextlib.nullcontext()


def result_root(experiment_id: str) -> Path:
    return PROJECT_ROOT / "full_benchmark_results" / experiment_id


def algorithm_output_dir(experiment_id: str, algorithm: str) -> Path:
    return result_root(experiment_id) / safe_name(algorithm)


def save_instance(experiment_id: str, case: BenchmarkCase, raw_data: dict[str, Any]) -> Path:
    instance_dir = result_root(experiment_id) / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    path = instance_dir / f"{case.name}.json"
    if not path.exists():
        path.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gap_percent(cmax: Any, lb: float) -> Any:
    if cmax == "" or lb <= 0:
        return ""
    return (float(cmax) - lb) / lb * 100.0


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate successful runs per case and method; retain the actual run count."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok" or row["cmax"] != "":
            groups.setdefault((row["case"], row["algorithm"]), []).append(row)

    summary = []
    for (case_name, algorithm), group in sorted(groups.items()):
        cmax_values = [float(row["cmax"]) for row in group if row["cmax"] != ""]
        gap_values = [float(row["gap_percent"]) for row in group if row["gap_percent"] != ""]
        if not cmax_values:
            continue
        first = group[0]
        summary.append(
            {
                "case": case_name,
                "scale": first["scale"],
                "n": first["n"],
                "m": first["m"],
                "lambda": first["lambda"],
                "algorithm": algorithm,
                "replications": len(cmax_values),
                "lb": first["lb"],
                "avg_cmax": statistics.fmean(cmax_values),
                "std_cmax": statistics.stdev(cmax_values) if len(cmax_values) > 1 else 0.0,
                "best_cmax": min(cmax_values),
                "worst_cmax": max(cmax_values),
                "avg_gap_percent": statistics.fmean(gap_values) if gap_values else "",
                "best_gap_percent": min(gap_values) if gap_values else "",
                "worst_gap_percent": max(gap_values) if gap_values else "",
            }
        )
    return summary


def write_run_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "run_details.csv", rows, RUN_DETAIL_FIELDS)
    write_csv(output_dir / "average_gap_summary.csv", summarize_rows(rows), SUMMARY_FIELDS)


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--cases", default="all", help="Comma-separated case names, e.g. n20_m2,n60_m3.")
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--small-time", type=float, default=SMALL_TIME_LIMIT)
    parser.add_argument("--medium-time", type=float, default=MEDIUM_TIME_LIMIT)
    parser.add_argument("--large-time", type=float, default=LARGE_TIME_LIMIT)
    parser.add_argument("--gurobi-time", type=float, default=GUROBI_TIME_LIMIT)
    parser.add_argument("--verbose", action="store_true")
    return parser


def write_experiment_config(experiment_id: str, extra: dict[str, Any]) -> None:
    root = result_root(experiment_id)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        return
    config = {
        "experiment_id": experiment_id,
        "base_seed": BASE_SEED,
        "benchmark_cases": [
            {"case": case.name, "scale": case.scale, "n": case.n, "m": case.m, "lambda": case.lambd}
            for case in BENCHMARK_CASES
        ],
        "parameter_ranges": {
            "T0_RANGE": T0_RANGE,
            "L_RANGE": L_RANGE,
            "U_RANGE": U_RANGE,
            "W_RANGE": W_RANGE,
            "Q_RANGE": Q_RANGE,
            "TL_RANGE": TL_RANGE,
            "TR_RANGE": TR_RANGE,
            "BETA_RANGE": BETA_RANGE,
        },
        **extra,
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def run_vns_family(
    algorithm: str,
    module_file: str,
    experiment_id: str,
    case_filter: str,
    replications: int,
    small_time: float,
    medium_time: float,
    large_time: float,
    verbose: bool,
) -> None:
    module = load_module(PROJECT_ROOT / module_file, f"full_benchmark_{safe_name(algorithm)}")
    output_dir = algorithm_output_dir(experiment_id, algorithm)
    output_dir.mkdir(parents=True, exist_ok=True)
    quiet = not verbose
    rows: list[dict[str, Any]] = []
    write_experiment_config(experiment_id, {"runner": algorithm})

    for case_index, case in selected_cases(case_filter):
        seed = case_instance_seed(case_index)
        raw_data = build_raw_instance(case, seed)
        save_instance(experiment_id, case, raw_data)
        lb_info = compute_lower_bound(prepare_common_data(raw_data))
        time_limit = heuristic_time_limit(case, small_time, medium_time, large_time)
        print(f"[{algorithm}] {case.name} x{replications}, time={time_limit:.0f}s", flush=True)

        for rep in range(1, replications + 1):
            s_seed = search_seed(case_index, rep, algorithm)
            try:
                configure_algorithm_module(module, case, time_limit)
                data = prepare_for_module(module, raw_data)
                with quiet_context(quiet):
                    solution, history = module.solve_vns(data, seed=s_seed, time_limit=time_limit)
                last = history[-1] if history else {"iteration": 0, "elapsed": 0.0}
                result = {
                    "status": "ok",
                    "cmax": float(solution.cmax),
                    "elapsed": float(last.get("elapsed", time_limit)),
                    "iterations": int(last.get("iteration", 0)),
                    "task_counts": json.dumps([len(seq) for seq in solution.sequences]),
                }
            except Exception as exc:
                result = {
                    "status": f"error: {exc}",
                    "cmax": "",
                    "elapsed": "",
                    "iterations": "",
                    "task_counts": "",
                }
            rows.append(make_run_row(case, algorithm, rep, seed, s_seed, time_limit, lb_info["lb"], result))
            write_run_outputs(output_dir, rows)

    print(f"[{algorithm}] saved: {output_dir}")


def run_meta_algorithm(
    algorithm: str,
    experiment_id: str,
    case_filter: str,
    replications: int,
    small_time: float,
    medium_time: float,
    large_time: float,
    verbose: bool,
) -> None:
    module = load_module(PROJECT_ROOT / "Metaheuristics.py", f"full_benchmark_{safe_name(algorithm)}")
    output_dir = algorithm_output_dir(experiment_id, algorithm)
    output_dir.mkdir(parents=True, exist_ok=True)
    quiet = not verbose
    rows: list[dict[str, Any]] = []
    write_experiment_config(experiment_id, {"runner": algorithm})

    for case_index, case in selected_cases(case_filter):
        seed = case_instance_seed(case_index)
        raw_data = build_raw_instance(case, seed)
        save_instance(experiment_id, case, raw_data)
        lb_info = compute_lower_bound(prepare_common_data(raw_data))
        time_limit = heuristic_time_limit(case, small_time, medium_time, large_time)
        print(f"[{algorithm}] {case.name} x{replications}, time={time_limit:.0f}s", flush=True)

        for rep in range(1, replications + 1):
            s_seed = search_seed(case_index, rep, algorithm)
            try:
                configure_algorithm_module(module, case, time_limit)
                data = prepare_for_module(module, raw_data)
                with quiet_context(quiet):
                    solution, history = module.solve_algorithm(
                        algorithm, data, seed=s_seed, time_limit=time_limit
                    )
                last = history[-1] if history else {"iteration": 0, "elapsed": 0.0}
                result = {
                    "status": "ok",
                    "cmax": float(solution.cmax),
                    "elapsed": float(last.get("elapsed", time_limit)),
                    "iterations": int(last.get("iteration", 0)),
                    "task_counts": json.dumps([len(seq) for seq in solution.sequences]),
                }
            except Exception as exc:
                result = {
                    "status": f"error: {exc}",
                    "cmax": "",
                    "elapsed": "",
                    "iterations": "",
                    "task_counts": "",
                }
            rows.append(make_run_row(case, algorithm, rep, seed, s_seed, time_limit, lb_info["lb"], result))
            write_run_outputs(output_dir, rows)

    print(f"[{algorithm}] saved: {output_dir}")


def make_run_row(
    case: BenchmarkCase,
    algorithm: str,
    replication: int,
    instance_seed: int,
    s_seed: Any,
    time_limit: float,
    lb: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case": case.name,
        "scale": case.scale,
        "n": case.n,
        "m": case.m,
        "lambda": case.lambd,
        "algorithm": algorithm,
        "replication": replication,
        "instance_seed": instance_seed,
        "search_seed": s_seed,
        "time_limit": time_limit,
        "lb": lb,
        "cmax": result.get("cmax", ""),
        "gap_percent": gap_percent(result.get("cmax", ""), lb),
        "status": result.get("status", ""),
        "elapsed": result.get("elapsed", ""),
        "iterations": result.get("iterations", ""),
        "task_counts": result.get("task_counts", ""),
        "gurobi_status": result.get("gurobi_status", ""),
        "gurobi_bound": result.get("gurobi_bound", ""),
        "gurobi_mip_gap": result.get("gurobi_mip_gap", ""),
    }


def run_gurobi(
    experiment_id: str,
    case_filter: str,
    gurobi_time: float,
    verbose: bool,
) -> None:
    output_dir = algorithm_output_dir(experiment_id, "Gurobi")
    output_dir.mkdir(parents=True, exist_ok=True)
    quiet = not verbose
    rows: list[dict[str, Any]] = []
    write_experiment_config(experiment_id, {"runner": "Gurobi"})

    try:
        gurobi_module = load_module(PROJECT_ROOT / "Gurobi.py", "full_benchmark_gurobi")
    except Exception as exc:
        print(f"[Gurobi] module load failed: {exc}", flush=True)
        gurobi_module = None

    for case_index, case in selected_cases(case_filter):
        seed = case_instance_seed(case_index)
        raw_data = build_raw_instance(case, seed)
        save_instance(experiment_id, case, raw_data)
        data = prepare_common_data(raw_data)
        lb_info = compute_lower_bound(data)
        print(f"[Gurobi] {case.name}, time={gurobi_time:.0f}s", flush=True)

        if gurobi_module is None:
            result = {
                "status": "module_unavailable",
                "cmax": "",
                "elapsed": "",
                "iterations": "",
                "task_counts": "",
                "gurobi_status": "",
                "gurobi_bound": "",
                "gurobi_mip_gap": "",
            }
        else:
            try:
                start = time.perf_counter()
                with quiet_context(quiet):
                    model, _y, _z, _cpos, _ck, cmax_var, _horizon, _big_m = gurobi_module.solve(
                        data, time_limit=gurobi_time, mip_gap=None
                    )
                elapsed = time.perf_counter() - start
                result = {
                    "status": "ok" if model.SolCount > 0 else "no_incumbent",
                    "cmax": float(cmax_var.X) if model.SolCount > 0 else "",
                    "elapsed": elapsed,
                    "iterations": "",
                    "task_counts": "",
                    "gurobi_status": getattr(model, "Status", ""),
                    "gurobi_bound": getattr(model, "ObjBound", ""),
                    "gurobi_mip_gap": getattr(model, "MIPGap", "") if model.SolCount > 0 else "",
                }
            except Exception as exc:
                result = {
                    "status": f"error: {exc}",
                    "cmax": "",
                    "elapsed": "",
                    "iterations": "",
                    "task_counts": "",
                    "gurobi_status": "",
                    "gurobi_bound": "",
                    "gurobi_mip_gap": "",
                }
        rows.append(make_run_row(case, "Gurobi", 1, seed, "", gurobi_time, lb_info["lb"], result))
        write_run_outputs(output_dir, rows)

    print(f"[Gurobi] saved: {output_dir}")
