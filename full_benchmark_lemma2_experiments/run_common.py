"""Separate benchmark runner for the four Lemma 2-enabled metaheuristics."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time

from experiment_common import (
    ALGORITHMS, HISTORY_FIELDS, RUN_FIELDS, argument_parser, benchmark, result_root,
)
import lemma2_metaheuristics as meta


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_algorithm(algorithm: str) -> None:
    """Persist each repetition, configuration provenance, and measured event history."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    args = argument_parser(f"Run {algorithm} with Lemma 2 on the 36 benchmark cases.").parse_args()
    if args.replications < 1 or min(args.small_time, args.medium_time, args.large_time) <= 0:
        raise ValueError("Replications and time limits must be positive.")
    cases = benchmark.selected_cases(args.cases)
    root = result_root(args.experiment_id)
    output = root / benchmark.safe_name(algorithm)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    parameters = {
        key: value for key, value in vars(meta.engine).items()
        if key.isupper() and isinstance(value, (int, float, str, list, tuple))
    }
    write_json(output / "config.json", {
        "algorithm": algorithm,
        "decoder": "Lemma 2: ascending rho; original priority and job ID break ties",
        "partial_evaluation": "Lemma 2 sorting and Lemma 1 recurrence",
        "experiment_id": args.experiment_id,
        "replications": args.replications,
        "time_limits": {"small": args.small_time, "medium": args.medium_time, "large": args.large_time},
        "cases": [vars(case) for _, case in cases],
        "parameter_ranges": {
            name: getattr(benchmark, name) for name in
            ("T0_RANGE", "L_RANGE", "U_RANGE", "W_RANGE", "Q_RANGE", "TL_RANGE", "TR_RANGE", "BETA_RANGE")
        },
        "base_seed": benchmark.BASE_SEED,
        "search_parameters": parameters,
        "engine_source": str(meta.ENGINE_PATH),
        "engine_sha256": hashlib.sha256(meta.ENGINE_PATH.read_bytes()).hexdigest(),
        "timing": "Original solver clock includes initial solution, sorting and search; elapsed is not clipped.",
    })

    for case_index, case in cases:
        seed = benchmark.case_instance_seed(case_index)
        raw = benchmark.build_raw_instance(case, seed)
        write_json(root / "instances" / f"{case.name}.json", raw)
        lb = benchmark.compute_lower_bound(benchmark.prepare_common_data(raw))["lb"]
        limit = benchmark.heuristic_time_limit(case, args.small_time, args.medium_time, args.large_time)
        print(f"[{algorithm} / Lemma 2] {case.name}, {args.replications} runs, {limit:g}s", flush=True)

        for rep in range(1, args.replications + 1):
            search_seed = benchmark.search_seed(case_index, rep, algorithm)
            history = []
            initial = {}
            result = {"status": "error", "cmax": "", "elapsed": "", "iterations": "", "task_counts": ""}
            try:
                benchmark.configure_algorithm_module(meta.engine, case, limit)
                data = meta.prepare_data(copy.deepcopy(raw))
                start = time.perf_counter()
                with benchmark.quiet_context(not args.verbose):
                    solution, history = meta.solve_algorithm(algorithm, data, search_seed, limit)
                elapsed = time.perf_counter() - start
                initial = history[0] if history and history[0]["event"] == "initial_lemma2_solution" else {}
                result = {
                    "status": "ok", "cmax": float(solution.cmax), "elapsed": elapsed,
                    "iterations": int(history[-1]["iteration"]) if history else 0,
                    "task_counts": json.dumps([len(seq) for seq in solution.sequences]),
                }
            except Exception as exc:
                result["status"] = f"error: {type(exc).__name__}: {exc}"

            row = benchmark.make_run_row(case, algorithm, rep, seed, search_seed, limit, lb, result)
            row.update({
                "decoder": "lemma2", "initial_cmax": initial.get("best_cmax", ""),
                "initial_elapsed": initial.get("elapsed", ""),
            })
            rows.append(row)
            benchmark.write_csv(output / "run_details.csv", rows, RUN_FIELDS)
            benchmark.write_csv(
                output / "average_gap_summary.csv", benchmark.summarize_rows(rows), benchmark.SUMMARY_FIELDS,
            )
            history_rows = [
                {"case": case.name, "algorithm": algorithm, "replication": rep,
                 "instance_seed": seed, "search_seed": search_seed, "lb": lb, **event}
                for event in history
            ]
            benchmark.write_csv(
                output / "history" / f"{case.name}_rep{rep:02d}.csv", history_rows, HISTORY_FIELDS,
            )
            value = f"Cmax={result['cmax']:.6f}, elapsed={result['elapsed']:.2f}s" if result["status"] == "ok" else result["status"]
            print(f"  run {rep:02d}/{args.replications}: {value}", flush=True)
    print(f"[{algorithm}] saved: {output}", flush=True)
