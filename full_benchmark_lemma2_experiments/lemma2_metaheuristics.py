"""Isolated SA/TS/GA-SA/PSO-SA engines with a common Lemma 2 decoder.

The search operators and parameters are loaded from the existing
Metaheuristics.py. Only evaluation hooks are replaced on this private module
instance; importing this file does not change the original experiment modules.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

from experiment_common import benchmark


ENGINE_PATH = Path(__file__).resolve().parent.parent / "Metaheuristics.py"
engine = benchmark.load_module(ENGINE_PATH, "_lemma2_metaheuristics_engine")
Solution = engine.Solution
_original_prepare_data = engine.prepare_data
_original_decode = engine.decode
_original_initial_solution = engine.build_initial_solution
_initial_log = None


def prepare_data(data: dict) -> dict:
    data = _original_prepare_data(data)
    rho = []
    for job in range(data["n"]):
        indices = []
        for agv in range(data["m"]):
            q = data["q"][job][agv]
            alpha = data["alpha"][job][agv]
            if not math.isfinite(q) or not math.isfinite(alpha) or q < 1.0 or alpha < 0.0:
                raise ValueError("Lemma 2 evaluation requires finite q >= 1 and alpha >= 0.")
            # A positive-duration, non-deteriorating task follows q > 1 tasks.
            indices.append(alpha / (q - 1.0) if q > 1.0 else (math.inf if alpha > 0 else 0.0))
        if not engine.feasible_agvs(data, job):
            raise ValueError(f"Job {job + 1} has no capacity-feasible AGV.")
        rho.append(indices)
    if not math.isfinite(data["t0"]) or data["t0"] < 0.0:
        raise ValueError("The initial availability time must be finite and nonnegative.")
    data["rho"] = rho
    return data


def sort_sequence(seq: list[int], agv: int, data: dict, vector=None) -> list[int]:
    """Use rho as the primary key; priorities cannot override Lemma 2."""
    if "rho" not in data:
        prepare_data(data)
    n = data["n"]
    return sorted(
        seq,
        key=lambda job: (
            data["rho"][job][agv],
            vector[n + job] if vector is not None else 0.0,
            job,
        ),
    )


def decode(vector: list[float], data: dict) -> Solution:
    n = data["n"]
    if len(vector) != 2 * n or not all(math.isfinite(value) for value in vector):
        raise ValueError("A candidate must contain 2n finite real-valued components.")
    repaired = engine.repair_capacity(vector, data)
    assignment = engine.assignment_from_vector(repaired, data)
    sequences = [[] for _ in range(data["m"])]
    for job, agv in enumerate(assignment):
        sequences[agv].append(job)

    # Reuse the original row construction and Lemma 1 recurrence, supplying
    # temporary priorities that enforce Lemma 2. Keep the actual search vector.
    ordered_vector = repaired[:]
    for agv, seq in enumerate(sequences):
        for rank, job in enumerate(sort_sequence(seq, agv, data, repaired), start=1):
            ordered_vector[n + job] = float(rank)
    solution = _original_decode(ordered_vector, data)
    solution.vector = repaired
    for row in solution.rows:
        row["rho_ik"] = data["rho"][row["job"] - 1][row["agv"] - 1]
    return solution


def agv_sequence_completion(seq: list[int], agv: int, data: dict) -> float:
    """Apply the same HE ordering during partial insertion/removal evaluation."""
    current = data["t0"]
    for job in sort_sequence(seq, agv, data):
        if data["w"][job] > data["Q"][agv] + engine.EPS:
            return math.inf
        current = data["q"][job][agv] * current + data["alpha"][job][agv]
    return current


def build_initial_solution(data: dict, rng: random.Random) -> Solution:
    solution = _original_initial_solution(data, rng)
    if _initial_log is not None and "row" not in _initial_log:
        _initial_log["row"] = engine.log_row(
            0, time.perf_counter() - _initial_log["start"],
            solution, solution, solution, "initial_lemma2_solution",
        )
    return solution


# All candidates, including nested SA, population construction/restarts,
# PSO updates, partial reinsertion and removal-gain calculations use these hooks.
engine.prepare_data = prepare_data
engine.decode = decode
engine.agv_sequence_completion = agv_sequence_completion
engine.build_initial_solution = build_initial_solution


def solve_algorithm(algorithm: str, data: dict, seed: int, time_limit: float):
    global _initial_log
    _initial_log = {"start": time.perf_counter()}
    try:
        solution, history = engine.solve_algorithm(algorithm, data, seed, time_limit)
        initial = _initial_log.get("row")
        return solution, ([initial] if initial is not None else []) + history
    finally:
        _initial_log = None
