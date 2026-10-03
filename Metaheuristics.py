"""
Comparative metaheuristics for the multi-AGV scheduling problem.

Algorithms:
    - SA
    - TS
    - GA-SA
    - PSO-SA

All algorithms use the same representation as VNS.py:
    - A real-coded vector of length 2n.
    - V[0:n] encodes job-to-AGV assignment through the ceiling map.
    - V[n:2n] encodes the execution priority inside the assigned AGV.

This file intentionally does not use rho sorting. AGV internal sequences
are decoded only by the priority segment V[n:2n]. Unlike VNS.py, these engines
do not quantize priorities. The HE adapter replaces full and partial evaluation
without changing this file's search strategies.
Every algorithm uses a time-based stopping condition.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


# -----------------------------
# General experiment parameters
# -----------------------------
N = 200  # Number of transportation jobs.
M = 5 # Number of AGVs.
SEED = 20260722  # Random seed used for reproducible instance generation and search.
TIME_LIMIT = 60.0  # Runtime limit in seconds for each algorithm.
ALGORITHMS_TO_RUN = ["SA", "TS", "GA-SA", "PSO-SA"]  # Algorithms solved sequentially.
EPS = 1e-9  # Numerical tolerance used in capacity and comparison checks.

# -----------------------------
# VNS neighborhood parameters
# -----------------------------
SEGMENT_REASSIGN_TRIALS = 30  # Number of short segment reassignment attempts used by VNS.
CRITICAL_JOB_LIMIT = 5  # Number of high removal-gain jobs considered on the bottleneck AGV.
EXCHANGE_TARGET_LIMIT = 12  # Number of candidate exchange jobs considered from a target AGV.
RUIN_RECREATE_TRIALS = 10  # Number of ruin-and-recreate attempts used by VNS.
RUIN_MIN_SIZE = 3  # Minimum number of removed jobs in ruin-and-recreate.
RUIN_MAX_SIZE = 12  # Maximum number of removed jobs in ruin-and-recreate.
SEGMENT_MAX_SIZE = 20  # Maximum random segment length in reassignment.

# -----------------------------
# SA parameters
# -----------------------------
SA_INITIAL_TEMP = 50.0  # Fixed initial temperature for the standalone SA algorithm.
SA_FINAL_TEMP = 1e-4  # Minimum temperature, preventing division by near-zero values.
SA_COOLING = 0.997  # Geometric cooling rate; closer to 1 means slower cooling.

# -----------------------------
# TS parameters
# -----------------------------
TS_CANDIDATES = 45  # Number of random neighbor moves sampled per tabu-search iteration.
TS_TENURE = 20  # Maximum number of recent moves kept in the tabu list.

# -----------------------------
# GA-SA parameters
# -----------------------------
GA_POP_SIZE = 100  # Population size.
GA_TOURNAMENT_SIZE = 3  # Number of individuals sampled in tournament selection.
GA_MUTATION_RATE = 0.10  # Probability of mutating assignment/priority genes.
GA_SA_INITIAL_TEMP = 100.0  # Fixed initial temperature for SA polishing inside GA-SA.
GA_SA_STEPS = 24  # Number of short SA polishing steps applied to each offspring.
GA_OFFSPRING_TRIALS = 2  # Number of offspring sampled each generation; the best one survives.
GA_ELITE_COUNT = 4  # Number of best individuals protected from replacement.
GA_STAGNATION_RESTART = 25  # Restart worst individuals after this many generations without improvement.
GA_RESTART_FRACTION = 0.25  # Fraction of the population rebuilt during a stagnation restart.

# -----------------------------
# PSO-SA parameters
# -----------------------------
PSO_SWARM_SIZE = 100  # Number of particles in the swarm.
PSO_INERTIA = 0.72  # Inertia weight controlling persistence of particle movement.
PSO_COGNITIVE = 1.45  # Cognitive learning factor toward a particle's personal best.
PSO_SOCIAL = 1.45  # Social learning factor toward the global best.
PSO_ASSIGNMENT_VMAX = 1.25  # Velocity bound for assignment genes.
PSO_PRIORITY_VMAX = 6.0  # Velocity bound for priority genes.
PSO_SA_INTERVAL = 5  # Apply SA polishing every this many PSO iterations.
PSO_SA_STEPS = 24  # Number of SA steps applied to selected particles.
PSO_SA_INITIAL_TEMP = 500.0  # Fixed initial temperature for SA polishing inside PSO-SA.
PSO_ELITE_COUNT = 4  # Number of best particles protected and periodically polished.
PSO_STAGNATION_RESTART = 20  # Restart weak particles after this many iterations without improvement.
PSO_RESTART_FRACTION = 0.25  # Fraction of worst particles rebuilt during stagnation restart.
PSO_MUTATION_RATE = 0.25  # Mutation rate used when creating initial/restarted particles.

# -----------------------------
# Instance generation parameters
# -----------------------------
T0_RANGE = (0.0, 2.0)  # Earliest available time range for all AGVs.
L_RANGE = (0.2, 2.0)  # Job loading time range.
U_RANGE = (0.2, 2.0)  # Job unloading time range.
W_RANGE = (0.5, 3.0)  # Job cargo weight range.
Q_RANGE = (6.0, 8.0)  # AGV rated capacity range.
TL_RANGE = (2.0, 5.0)  # Normal loaded travel time range.
TR_RANGE = (1.5, 3.5)  # Normal empty return time range.
BETA_RANGE = (0.01, 0.03)  # AGV base deterioration coefficient range.
LAMBDA = 0.2  # Load-dependent deterioration sensitivity coefficient.


@dataclass
class Solution:
    vector: list[float]
    assignment: list[int]
    sequences: list[list[int]]
    completion: list[float]
    cmax: float
    rows: list[dict]


def uniform_list(rng: random.Random, size: int, lo: float, hi: float) -> list[float]:
    return [round(rng.uniform(lo, hi), 4) for _ in range(size)]


def uniform_matrix(
    rng: random.Random, rows: int, cols: int, lo: float, hi: float
) -> list[list[float]]:
    return [[round(rng.uniform(lo, hi), 4) for _ in range(cols)] for _ in range(rows)]


def build_instance(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    data = {
        "n": N,
        "m": M,
        "seed": seed,
        "t0": round(rng.uniform(*T0_RANGE), 4),
        "lambda": LAMBDA,
        "L": uniform_list(rng, N, *L_RANGE),
        "U": uniform_list(rng, N, *U_RANGE),
        "w": uniform_list(rng, N, *W_RANGE),
        "Q": uniform_list(rng, M, *Q_RANGE),
        "TL": uniform_matrix(rng, N, M, *TL_RANGE),
        "TR": uniform_matrix(rng, N, M, *TR_RANGE),
        "beta": uniform_list(rng, M, *BETA_RANGE),
    }
    return prepare_data(data)


def prepare_data(data: dict) -> dict:
    n = data["n"]
    m = data["m"]
    gamma = [[0.0 for _ in range(m)] for _ in range(n)]
    q = [[0.0 for _ in range(m)] for _ in range(n)]
    alpha = [[0.0 for _ in range(m)] for _ in range(n)]

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

    data["gamma"] = gamma
    data["q"] = q
    data["alpha"] = alpha
    return data


def load_or_build_instance(base_dir: Path) -> dict:
    instance_path = base_dir / f"agv_instance_n{N}_m{M}.json"
    if instance_path.exists():
        with instance_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("n") == N and data.get("m") == M:
            return prepare_data(data)
    return build_instance()


def feasible_agvs(data: dict, job: int) -> list[int]:
    return [agv for agv in range(data["m"]) if data["w"][job] <= data["Q"][agv] + EPS]


def assignment_value(agv: int) -> float:
    return agv + 0.5


def assignment_from_vector(vector: list[float], data: dict) -> list[int]:
    assignment = []
    for job in range(data["n"]):
        value = min(max(vector[job], EPS), data["m"])
        agv = math.ceil(value) - 1
        assignment.append(min(max(agv, 0), data["m"] - 1))
    return assignment


def repair_capacity(vector: list[float], data: dict) -> list[float]:
    repaired = vector[:]
    assignment = assignment_from_vector(repaired, data)
    for job, agv in enumerate(assignment):
        if data["w"][job] <= data["Q"][agv] + EPS:
            continue
        candidates = feasible_agvs(data, job)
        if not candidates:
            raise ValueError(f"Job {job + 1} cannot be processed by any AGV.")
        repaired[job] = assignment_value(candidates[0])
    return repaired


def vector_from_sequences(sequences: list[list[int]], data: dict) -> list[float]:
    n = data["n"]
    vector = [0.0 for _ in range(2 * n)]
    seen = set()
    for agv, seq in enumerate(sequences):
        for rank, job in enumerate(seq, start=1):
            vector[job] = assignment_value(agv)
            vector[n + job] = float(rank)
            seen.add(job)

    missing = set(range(n)) - seen
    if missing:
        raise ValueError(f"Missing jobs in sequence encoding: {sorted(missing)}")
    return repair_capacity(vector, data)


def decode(vector: list[float], data: dict) -> Solution:
    """Original decoder: ceiling-map assignment, then ascending real priorities."""
    vector = repair_capacity(vector, data)
    n = data["n"]
    m = data["m"]
    assignment = assignment_from_vector(vector, data)
    sequences = [[] for _ in range(m)]

    for job, agv in enumerate(assignment):
        sequences[agv].append(job)

    for agv in range(m):
        sequences[agv].sort(key=lambda job: (vector[n + job], job))

    completion = []
    rows = []
    for agv, seq in enumerate(sequences):
        current_time = data["t0"]
        for pos, job in enumerate(seq, start=1):
            start_time = current_time
            finish_time = data["q"][job][agv] * start_time + data["alpha"][job][agv]
            rows.append(
                {
                    "agv": agv + 1,
                    "position": pos,
                    "job": job + 1,
                    "start_time": start_time,
                    "finish_time": finish_time,
                    "L_i": data["L"][job],
                    "U_i": data["U"][job],
                    "w_i": data["w"][job],
                    "Q_k": data["Q"][agv],
                    "T_L_ik": data["TL"][job][agv],
                    "T_R_ik": data["TR"][job][agv],
                    "beta_k": data["beta"][agv],
                    "gamma_ik": data["gamma"][job][agv],
                    "q_ik": data["q"][job][agv],
                    "alpha_ik": data["alpha"][job][agv],
                }
            )
            current_time = finish_time
        completion.append(current_time)

    return Solution(
        vector=vector,
        assignment=assignment,
        sequences=sequences,
        completion=completion,
        cmax=max(completion),
        rows=rows,
    )


def evaluate_sequences(sequences: list[list[int]], data: dict) -> Solution:
    return decode(vector_from_sequences(sequences, data), data)


def agv_sequence_completion(seq: list[int], agv: int, data: dict) -> float:
    current_time = data["t0"]
    for job in seq:
        current_time = data["q"][job][agv] * current_time + data["alpha"][job][agv]
    return current_time


def build_initial_solution(data: dict, rng: random.Random) -> Solution:
    sequences = [[] for _ in range(data["m"])]
    completions = [data["t0"] for _ in range(data["m"])]

    for job in range(data["n"]):
        if job < data["m"] and data["w"][job] <= data["Q"][job] + EPS:
            best_agv = job
        else:
            best_agv = None
            best_cmax = float("inf")
            for agv in feasible_agvs(data, job):
                candidate_seq = sequences[agv] + [job]
                candidate_completion = agv_sequence_completion(candidate_seq, agv, data)
                candidate_cmax = max(
                    candidate_completion,
                    max(completions[g] for g in range(data["m"]) if g != agv),
                )
                if candidate_cmax < best_cmax:
                    best_cmax = candidate_cmax
                    best_agv = agv

        sequences[best_agv].append(job)
        completions[best_agv] = agv_sequence_completion(sequences[best_agv], best_agv, data)

    return evaluate_sequences(sequences, data)


def better(candidate: Solution | None, incumbent: Solution) -> bool:
    return candidate is not None and candidate.cmax < incumbent.cmax - 1e-8


def clone_sequences(sequences: list[list[int]]) -> list[list[int]]:
    return [seq[:] for seq in sequences]


def partial_cmax(sequences: list[list[int]], data: dict) -> float:
    completions = []
    for agv, seq in enumerate(sequences):
        completions.append(agv_sequence_completion(seq, agv, data))
    return max(completions)


def bottleneck_agv(solution: Solution) -> int:
    return max(range(len(solution.completion)), key=lambda agv: solution.completion[agv])


def removal_gain(solution: Solution, agv: int, job: int, data: dict) -> float:
    reduced_seq = [item for item in solution.sequences[agv] if item != job]
    return solution.completion[agv] - agv_sequence_completion(reduced_seq, agv, data)


def top_removal_jobs(solution: Solution, agv: int, data: dict, limit: int) -> list[int]:
    ranked = sorted(
        solution.sequences[agv],
        key=lambda job: removal_gain(solution, agv, job, data),
        reverse=True,
    )
    return ranked[: min(limit, len(ranked))]


def add_unique_jobs(pool: list[int], jobs: list[int], limit: int) -> None:
    seen = set(pool)
    for job in jobs:
        if job not in seen:
            pool.append(job)
            seen.add(job)
        if len(pool) >= limit:
            break


def exchange_candidate_jobs(
    solution: Solution,
    source: int,
    target: int,
    source_job: int,
    data: dict,
    rng: random.Random,
) -> list[int]:
    feasible = [
        job
        for job in solution.sequences[target]
        if data["w"][job] <= data["Q"][source] + EPS
        and data["w"][source_job] <= data["Q"][target] + EPS
    ]
    if not feasible:
        return []

    pool: list[int] = []
    add_unique_jobs(
        pool,
        sorted(
            feasible,
            key=lambda job: removal_gain(solution, target, job, data),
            reverse=True,
        ),
        EXCHANGE_TARGET_LIMIT // 2,
    )
    add_unique_jobs(
        pool,
        sorted(feasible, key=lambda job: data["alpha"][job][source]),
        EXCHANGE_TARGET_LIMIT,
    )

    remaining = [job for job in feasible if job not in set(pool)]
    rng.shuffle(remaining)
    add_unique_jobs(pool, remaining, EXCHANGE_TARGET_LIMIT)
    return pool[:EXCHANGE_TARGET_LIMIT]


def sample_critical_task_transfer(
    solution: Solution, data: dict, rng: random.Random
) -> tuple[Solution, tuple]:
    source = bottleneck_agv(solution)
    best_candidate = None
    best_move = None

    for job in top_removal_jobs(solution, source, data, CRITICAL_JOB_LIMIT):
        targets = [agv for agv in range(data["m"]) if agv != source]
        rng.shuffle(targets)
        for target in targets:
            if data["w"][job] > data["Q"][target] + EPS:
                continue
            trial = clone_sequences(solution.sequences)
            trial[source].remove(job)
            trial[target].append(job)
            candidate = evaluate_sequences(trial, data)
            if best_candidate is None or candidate.cmax < best_candidate.cmax:
                best_candidate = candidate
                best_move = (job, source, target)

    if best_candidate is None:
        return solution, ("N1_transfer_none",)
    return best_candidate, ("N1_transfer", *best_move)


def sample_limited_candidate_exchange(
    solution: Solution, data: dict, rng: random.Random
) -> tuple[Solution, tuple]:
    source = bottleneck_agv(solution)
    best_candidate = None
    best_move = None

    for source_job in top_removal_jobs(solution, source, data, CRITICAL_JOB_LIMIT):
        targets = [agv for agv in range(data["m"]) if agv != source]
        rng.shuffle(targets)
        for target in targets:
            for target_job in exchange_candidate_jobs(
                solution, source, target, source_job, data, rng
            ):
                trial = clone_sequences(solution.sequences)
                trial[source].remove(source_job)
                trial[target].remove(target_job)
                trial[source].append(target_job)
                trial[target].append(source_job)
                candidate = evaluate_sequences(trial, data)
                if best_candidate is None or candidate.cmax < best_candidate.cmax:
                    best_candidate = candidate
                    best_move = (source_job, target_job, source, target)

    if best_candidate is None:
        return solution, ("N2_exchange_none",)
    return best_candidate, ("N2_exchange", *best_move)


def sample_short_segment_reassignment(
    solution: Solution, data: dict, rng: random.Random
) -> tuple[Solution, tuple]:
    n = data["n"]
    best_candidate = None
    best_move = None

    for _ in range(SEGMENT_REASSIGN_TRIALS):
        left = rng.randrange(n)
        max_segment_len = max(1, min(n - left, SEGMENT_MAX_SIZE, max(2, n // 50)))
        right = left + rng.randrange(1, max_segment_len + 1)
        vector = solution.vector[:]

        for job in range(left, right):
            current_agv = solution.assignment[job]
            candidates = [agv for agv in feasible_agvs(data, job) if agv != current_agv]
            if not candidates:
                candidates = feasible_agvs(data, job)
            target_agv = rng.choice(candidates)
            vector[job] = target_agv + rng.uniform(0.05, 0.95)

        candidate = decode(vector, data)
        if best_candidate is None or candidate.cmax < best_candidate.cmax:
            best_candidate = candidate
            best_move = (left, right)

    if best_candidate is None:
        return solution, ("N3_segment_reassign_none",)
    return best_candidate, ("N3_segment_reassign", *best_move)


def greedy_reinsert_removed_jobs(
    base_sequences: list[list[int]], removed_jobs: list[int], data: dict
) -> Solution:
    sequences = clone_sequences(base_sequences)
    for job in sorted(
        removed_jobs,
        key=lambda item: min(data["alpha"][item][agv] for agv in feasible_agvs(data, item)),
        reverse=True,
    ):
        best_agv = None
        best_cmax = float("inf")
        for agv in feasible_agvs(data, job):
            trial = clone_sequences(sequences)
            trial[agv].append(job)
            trial_cmax = partial_cmax(trial, data)
            if trial_cmax < best_cmax:
                best_cmax = trial_cmax
                best_agv = agv
        sequences[best_agv].append(job)
    return evaluate_sequences(sequences, data)


def sample_small_ruin_recreate(
    solution: Solution, data: dict, rng: random.Random
) -> tuple[Solution, tuple]:
    source = bottleneck_agv(solution)
    best_candidate = None
    best_removed = None
    n = data["n"]
    max_remove = min(RUIN_MAX_SIZE, max(RUIN_MIN_SIZE, n // 40))

    for _ in range(RUIN_RECREATE_TRIALS):
        remove_size = rng.randint(RUIN_MIN_SIZE, max_remove)
        critical_count = max(1, remove_size // 2)
        removed = top_removal_jobs(solution, source, data, critical_count)

        remaining_jobs = [
            job
            for seq in solution.sequences
            for job in seq
            if job not in set(removed)
        ]
        rng.shuffle(remaining_jobs)
        add_unique_jobs(removed, remaining_jobs, remove_size)

        removed_set = set(removed)
        base_sequences = [
            [job for job in seq if job not in removed_set]
            for seq in solution.sequences
        ]
        candidate = greedy_reinsert_removed_jobs(base_sequences, removed, data)
        if best_candidate is None or candidate.cmax < best_candidate.cmax:
            best_candidate = candidate
            best_removed = tuple(removed)

    if best_candidate is None:
        return solution, ("N4_ruin_recreate_none",)
    return best_candidate, ("N4_ruin_recreate", best_removed)


def choose_other_agv(current_agv: int, data: dict, job: int, rng: random.Random) -> int:
    choices = [agv for agv in feasible_agvs(data, job) if agv != current_agv]
    return rng.choice(choices or feasible_agvs(data, job))


def random_neighbor(
    solution: Solution, data: dict, rng: random.Random, strength: int = 1
) -> tuple[Solution, tuple]:
    """Sample the four assignment neighborhoods and return a TS move attribute."""
    operators = [
        sample_critical_task_transfer,
        sample_limited_candidate_exchange,
        sample_short_segment_reassignment,
        sample_small_ruin_recreate,
    ]
    current = solution
    last_move: tuple = ("VNS_none",)

    for _ in range(max(1, strength)):
        ordered = operators[:]
        rng.shuffle(ordered)
        for operator in ordered:
            candidate, last_move = operator(current, data, rng)
            if candidate is not current:
                current = candidate
                break
    return current, last_move


def mutate_vector(
    vector: list[float], data: dict, rng: random.Random, mutation_rate: float
) -> list[float]:
    n = data["n"]
    child = vector[:]
    assignment = assignment_from_vector(child, data)
    for job in range(n):
        if rng.random() < mutation_rate:
            old_agv = assignment[job]
            new_agv = choose_other_agv(old_agv, data, job, rng)
            child[job] = new_agv + rng.uniform(0.05, 0.95)
        if rng.random() < mutation_rate:
            child[n + job] += rng.gauss(0.0, 2.0)
    if rng.random() < mutation_rate and n >= 2:
        a, b = rng.sample(range(n), 2)
        child[n + a], child[n + b] = child[n + b], child[n + a]

    child = repair_capacity(child, data)
    if rng.random() < 0.65:
        candidate, _ = random_neighbor(decode(child, data), data, rng, strength=2)
        return candidate.vector
    return child


def crossover(
    parent_a: Solution, parent_b: Solution, data: dict, rng: random.Random
) -> list[float]:
    n = data["n"]
    child = parent_a.vector[:]
    cut1, cut2 = sorted(rng.sample(range(n), 2))
    child[cut1:cut2] = parent_b.vector[cut1:cut2]
    if rng.random() < 0.5:
        child[n + cut1 : n + cut2] = parent_b.vector[n + cut1 : n + cut2]
    else:
        for job in range(n):
            child[n + job] = (
                0.5 * parent_a.vector[n + job]
                + 0.5 * parent_b.vector[n + job]
                + rng.gauss(0.0, 0.2)
            )
    return repair_capacity(child, data)


def simulated_annealing_polish(
    solution: Solution,
    data: dict,
    rng: random.Random,
    steps: int,
    initial_temperature: float,
) -> Solution:
    """Short SA refinement used inside population-based search."""
    current = solution
    best = solution
    temperature = max(SA_FINAL_TEMP, initial_temperature)
    for _ in range(steps):
        candidate, _ = random_neighbor(current, data, rng)
        delta = candidate.cmax - current.cmax
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, SA_FINAL_TEMP)):
            current = candidate
            if current.cmax < best.cmax:
                best = current
        temperature = max(SA_FINAL_TEMP, temperature * SA_COOLING)
    return best


def tabu_polish(
    solution: Solution, data: dict, rng: random.Random, steps: int
) -> Solution:
    current = solution
    best = solution
    tabu_queue: deque[tuple] = deque()
    tabu_set: set[tuple] = set()

    for _ in range(steps):
        best_candidate = None
        best_move = None
        for _ in range(max(5, TS_CANDIDATES // 3)):
            candidate, move = random_neighbor(current, data, rng)
            if move in tabu_set and candidate.cmax >= best.cmax - 1e-8:
                continue
            if best_candidate is None or candidate.cmax < best_candidate.cmax:
                best_candidate = candidate
                best_move = move
        if best_candidate is None:
            continue

        current = best_candidate
        if current.cmax < best.cmax:
            best = current
        tabu_queue.append(best_move)
        tabu_set.add(best_move)
        if len(tabu_queue) > TS_TENURE:
            expired = tabu_queue.popleft()
            tabu_set.discard(expired)
    return best


def final_intensification(
    solution: Solution,
    data: dict,
    rng: random.Random,
    start: float,
    time_limit: float,
) -> Solution:
    if time.perf_counter() - start >= 0.98 * time_limit:
        return solution
    refined = tabu_polish(solution, data, rng, steps=8)
    return refined if refined.cmax < solution.cmax - 1e-8 else solution


def solve_sa(
    data: dict, seed: int = SEED, time_limit: float = TIME_LIMIT
) -> tuple[Solution, list[dict]]:
    """Timed SA with geometric cooling and probabilistic worsening-move acceptance."""
    rng = random.Random(seed)
    start = time.perf_counter()
    current = build_initial_solution(data, rng)
    best = current
    temperature = max(SA_FINAL_TEMP, SA_INITIAL_TEMP)
    history = []
    iteration = 0

    while time.perf_counter() - start < time_limit:
        iteration += 1
        candidate, _ = random_neighbor(current, data, rng)
        delta = candidate.cmax - current.cmax
        accepted = delta < 0 or rng.random() < math.exp(-delta / max(temperature, SA_FINAL_TEMP))
        event = "reject"
        if accepted:
            current = candidate
            event = "accept"
        if current.cmax < best.cmax - 1e-8:
            best = current
            event = "new_best"
        temperature = max(SA_FINAL_TEMP, temperature * SA_COOLING)

        if event == "new_best":
            elapsed = time.perf_counter() - start
            history.append(log_row(iteration, elapsed, candidate, current, best, event))

    best = final_intensification(best, data, rng, start, time_limit)
    history.append(log_row(iteration, time.perf_counter() - start, best, current, best, "finished"))
    return best, history


def solve_ts(
    data: dict, seed: int = SEED, time_limit: float = TIME_LIMIT
) -> tuple[Solution, list[dict]]:
    """Sample neighbors, enforce tabu tenure, and allow best-improving aspiration."""
    rng = random.Random(seed)
    start = time.perf_counter()
    current = build_initial_solution(data, rng)
    best = current
    history = []
    tabu_queue: deque[tuple] = deque()
    tabu_set: set[tuple] = set()
    iteration = 0

    while time.perf_counter() - start < time_limit:
        iteration += 1
        best_candidate = None
        best_move = None
        for _ in range(TS_CANDIDATES):
            candidate, move = random_neighbor(current, data, rng)
            if move in tabu_set and candidate.cmax >= best.cmax - 1e-8:
                continue
            if best_candidate is None or candidate.cmax < best_candidate.cmax:
                best_candidate = candidate
                best_move = move

        if best_candidate is None:
            continue
        current = best_candidate
        event = "move"
        if current.cmax < best.cmax - 1e-8:
            best = current
            event = "new_best"

        tabu_queue.append(best_move)
        tabu_set.add(best_move)
        if len(tabu_queue) > TS_TENURE:
            expired = tabu_queue.popleft()
            tabu_set.discard(expired)

        if event == "new_best":
            elapsed = time.perf_counter() - start
            history.append(log_row(iteration, elapsed, current, current, best, event))

    best = final_intensification(best, data, rng, start, time_limit)
    history.append(log_row(iteration, time.perf_counter() - start, best, current, best, "finished"))
    return best, history


def tournament(population: list[Solution], rng: random.Random) -> Solution:
    contenders = rng.sample(population, min(GA_TOURNAMENT_SIZE, len(population)))
    return min(contenders, key=lambda solution: solution.cmax)


def solve_ga_sa(
    data: dict, seed: int = SEED, time_limit: float = TIME_LIMIT
) -> tuple[Solution, list[dict]]:
    """Population search with crossover/mutation, SA refinement, and elitism."""
    rng = random.Random(seed)
    start = time.perf_counter()
    initial = build_initial_solution(data, rng)
    population = [initial]
    while len(population) < GA_POP_SIZE:
        base = rng.choice(population)
        mutation_rate = rng.uniform(0.18, 0.45)
        vector = mutate_vector(base.vector, data, rng, mutation_rate=mutation_rate)
        population.append(decode(vector, data))
    population.sort(key=lambda solution: solution.cmax)
    best = population[0]
    history = []
    generation = 0
    stagnant_generations = 0

    while time.perf_counter() - start < time_limit:
        generation += 1
        best_child = None
        for _ in range(GA_OFFSPRING_TRIALS):
            if time.perf_counter() - start >= time_limit:
                break
            parent_a = tournament(population, rng)
            parent_b = tournament(population, rng)
            child_vector = crossover(parent_a, parent_b, data, rng)
            child_vector = mutate_vector(child_vector, data, rng, GA_MUTATION_RATE)
            child = decode(child_vector, data)
            child = simulated_annealing_polish(
                child, data, rng, GA_SA_STEPS, GA_SA_INITIAL_TEMP
            )
            if best_child is None or child.cmax < best_child.cmax:
                best_child = child

        if best_child is None:
            break

        replace_range = range(GA_ELITE_COUNT, len(population))
        worst_index = max(replace_range, key=lambda index: population[index].cmax)
        child = best_child
        if child.cmax < population[worst_index].cmax:
            population[worst_index] = child
        population.sort(key=lambda solution: solution.cmax)

        event = "evolve"
        if population[0].cmax < best.cmax - 1e-8:
            best = population[0]
            stagnant_generations = 0
            event = "new_best"
        else:
            stagnant_generations += 1

        if stagnant_generations >= GA_STAGNATION_RESTART:
            restart_count = max(1, int(GA_POP_SIZE * GA_RESTART_FRACTION))
            for idx in range(GA_POP_SIZE - restart_count, GA_POP_SIZE):
                if time.perf_counter() - start >= time_limit:
                    break
                base = population[rng.randrange(max(1, GA_POP_SIZE // 3))]
                vector = mutate_vector(base.vector, data, rng, mutation_rate=0.5)
                population[idx] = decode(vector, data)
            population.sort(key=lambda solution: solution.cmax)
            stagnant_generations = 0

        if event == "new_best":
            elapsed = time.perf_counter() - start
            history.append(log_row(generation, elapsed, child, population[0], best, event))

    best = final_intensification(best, data, rng, start, time_limit)
    history.append(log_row(generation, time.perf_counter() - start, best, population[0], best, "finished"))
    return best, history


def fitness(solution: Solution) -> float:
    return 1.0 / (1.0 + solution.cmax)


def clamp_particle_vector(vector: list[float], data: dict) -> list[float]:
    n = data["n"]
    m = data["m"]
    clamped = vector[:]
    for job in range(n):
        clamped[job] = min(max(clamped[job], EPS), m - EPS)
        clamped[n + job] = min(max(clamped[n + job], 0.0), float(n))
    return repair_capacity(clamped, data)


def random_velocity(data: dict, rng: random.Random) -> list[float]:
    n = data["n"]
    velocity = [0.0 for _ in range(2 * n)]
    for job in range(n):
        velocity[job] = rng.uniform(-PSO_ASSIGNMENT_VMAX, PSO_ASSIGNMENT_VMAX)
        velocity[n + job] = rng.uniform(-PSO_PRIORITY_VMAX, PSO_PRIORITY_VMAX)
    return velocity


def build_pso_particle(base: Solution, data: dict, rng: random.Random, mutation_rate: float) -> Solution:
    vector = mutate_vector(base.vector, data, rng, mutation_rate=mutation_rate)
    vector = clamp_particle_vector(vector, data)
    candidate = decode(vector, data)
    if rng.random() < 0.45:
        candidate = simulated_annealing_polish(
            candidate,
            data,
            rng,
            steps=max(4, PSO_SA_STEPS // 3),
            initial_temperature=PSO_SA_INITIAL_TEMP,
        )
    return candidate


def update_particle_position(
    position: list[float],
    velocity: list[float],
    personal_best: list[float],
    global_best: list[float],
    data: dict,
    rng: random.Random,
) -> tuple[list[float], list[float]]:
    n = data["n"]
    new_position = position[:]
    new_velocity = velocity[:]

    for dim in range(2 * n):
        vmax = PSO_ASSIGNMENT_VMAX if dim < n else PSO_PRIORITY_VMAX
        cognitive = PSO_COGNITIVE * rng.random() * (personal_best[dim] - position[dim])
        social = PSO_SOCIAL * rng.random() * (global_best[dim] - position[dim])
        value = PSO_INERTIA * velocity[dim] + cognitive + social
        value = min(max(value, -vmax), vmax)
        new_velocity[dim] = value
        new_position[dim] += value

    return clamp_particle_vector(new_position, data), new_velocity


def restart_weak_pso_particles(
    particles: list[Solution],
    velocities: list[list[float]],
    personal_best: list[Solution],
    global_best: Solution,
    data: dict,
    rng: random.Random,
) -> None:
    elite_count = min(PSO_ELITE_COUNT, len(particles))
    restart_count = max(1, int(len(particles) * PSO_RESTART_FRACTION))
    ranked = sorted(range(len(particles)), key=lambda idx: particles[idx].cmax)
    replace_indices = ranked[elite_count:][-restart_count:]

    for offset, index in enumerate(replace_indices):
        mutation_rate = PSO_MUTATION_RATE if offset % 2 == 0 else min(0.85, PSO_MUTATION_RATE + 0.35)
        particles[index] = build_pso_particle(global_best, data, rng, mutation_rate=mutation_rate)
        velocities[index] = random_velocity(data, rng)
        personal_best[index] = particles[index]


def solve_pso_sa(
    data: dict, seed: int = SEED, time_limit: float = TIME_LIMIT
) -> tuple[Solution, list[dict]]:
    """Particle updates with periodic elite SA refinement and stagnation restarts."""
    rng = random.Random(seed)
    start = time.perf_counter()
    initial = build_initial_solution(data, rng)
    particles = [initial]
    velocities = [random_velocity(data, rng)]

    while len(particles) < PSO_SWARM_SIZE:
        mutation_rate = PSO_MUTATION_RATE if len(particles) < PSO_SWARM_SIZE // 2 else 0.55
        particles.append(build_pso_particle(initial, data, rng, mutation_rate=mutation_rate))
        velocities.append(random_velocity(data, rng))

    personal_best = particles[:]
    global_best = min(personal_best, key=lambda solution: solution.cmax)
    current_best = global_best
    history = []
    iteration = 0
    no_improve = 0

    while time.perf_counter() - start < time_limit:
        iteration += 1
        event = "iteration"

        for index, particle in enumerate(particles):
            if time.perf_counter() - start >= time_limit:
                break
            new_vector, new_velocity = update_particle_position(
                particle.vector,
                velocities[index],
                personal_best[index].vector,
                global_best.vector,
                data,
                rng,
            )
            candidate = decode(new_vector, data)
            if rng.random() < 0.18:
                candidate, _ = random_neighbor(candidate, data, rng, strength=1)

            particles[index] = candidate
            velocities[index] = new_velocity

            if candidate.cmax < personal_best[index].cmax - 1e-8:
                personal_best[index] = candidate

        current_best = min(personal_best, key=lambda solution: solution.cmax)

        if iteration % PSO_SA_INTERVAL == 0:
            elite_indices = sorted(range(len(personal_best)), key=lambda idx: personal_best[idx].cmax)[
                : min(PSO_ELITE_COUNT, len(personal_best))
            ]
            for index in elite_indices:
                if time.perf_counter() - start >= time_limit:
                    break
                refined = simulated_annealing_polish(
                    personal_best[index],
                    data,
                    rng,
                    steps=PSO_SA_STEPS,
                    initial_temperature=PSO_SA_INITIAL_TEMP,
                )
                if refined.cmax < personal_best[index].cmax - 1e-8:
                    personal_best[index] = refined
                    particles[index] = refined
                    velocities[index] = random_velocity(data, rng)
            current_best = min(personal_best, key=lambda solution: solution.cmax)
            event = "sa_refine"

        if current_best.cmax < global_best.cmax - 1e-8:
            global_best = current_best
            no_improve = 0
            event = "new_best"
        else:
            no_improve += 1

        if no_improve >= PSO_STAGNATION_RESTART:
            restart_weak_pso_particles(
                particles,
                velocities,
                personal_best,
                global_best,
                data,
                rng,
            )
            current_best = min(personal_best, key=lambda solution: solution.cmax)
            if current_best.cmax < global_best.cmax - 1e-8:
                global_best = current_best
                event = "new_best"
            else:
                event = "restart"
            no_improve = 0

        if event == "new_best":
            elapsed = time.perf_counter() - start
            history.append(log_row(iteration, elapsed, current_best, current_best, global_best, event))

    global_best = final_intensification(global_best, data, rng, start, time_limit)
    history.append(
        log_row(
            iteration,
            time.perf_counter() - start,
            global_best,
            current_best,
            global_best,
            "finished",
        )
    )
    return global_best, history


def log_row(
    iteration: int,
    elapsed: float,
    candidate: Solution,
    current: Solution,
    best: Solution,
    event: str,
) -> dict:
    return {
        "iteration": iteration,
        "elapsed": elapsed,
        "candidate_cmax": candidate.cmax,
        "current_cmax": current.cmax,
        "best_cmax": best.cmax,
        "event": event,
    }


def print_status(
    iteration: int,
    elapsed: float,
    state_value: float | int,
    candidate: Solution,
    current: Solution,
    best: Solution,
    event: str,
) -> None:
    print(
        f"{iteration:4d} | {elapsed:10.2f} | {state_value:11.4g} | "
        f"{candidate.cmax:9.6f} | {current.cmax:7.6f} | "
        f"{best.cmax:7.6f} | {event}",
        flush=True,
    )


def plot_gantt(rows: list[dict], cmax: float, output_path: Path, title: str) -> None:
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))
    cmap = plt.get_cmap("tab20")
    agvs = sorted({row["agv"] for row in rows})
    y_pos = {agv: idx for idx, agv in enumerate(agvs)}

    for row in rows:
        start = row["start_time"]
        finish = row["finish_time"]
        duration = finish - start
        color = cmap((row["job"] - 1) % 20)
        ax.barh(
            y_pos[row["agv"]],
            duration,
            left=start,
            height=0.55,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(
            start + duration / 2,
            y_pos[row["agv"]],
            f"J{row['job']}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.axvline(cmax, color="red", linestyle="--", linewidth=1.2, label=f"Cmax={cmax:.2f}")
    ax.set_yticks([y_pos[agv] for agv in agvs])
    ax.set_yticklabels([f"AGV {agv}" for agv in agvs])
    ax.set_xlabel("Time")
    ax.set_ylabel("AGV")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_results(algorithm: str, data: dict, solution: Solution, base_dir: Path) -> Path:
    suffix = f"n{data['n']}_m{data['m']}"
    safe_name = algorithm.lower().replace("-", "_")
    gantt_path = base_dir / f"{safe_name}_gantt_{suffix}.png"
    plot_gantt(solution.rows, solution.cmax, gantt_path, f"{algorithm} Multi-AGV Scheduling Gantt Chart")
    return gantt_path


def print_final_summary(results: list[dict]) -> None:
    print("\n========== Metaheuristics Summary ==========")
    print("algorithm | elapsed(s) | iterations | best Cmax | AGV completions | task counts | gantt")
    print("-" * 118)
    for result in results:
        completion_text = "[" + ", ".join(f"{value:.3f}" for value in result["completion"]) + "]"
        task_count_text = "[" + ", ".join(str(value) for value in result["task_counts"]) + "]"
        print(
            f"{result['algorithm']:9s} | "
            f"{result['elapsed']:10.2f} | "
            f"{result['iterations']:10d} | "
            f"{result['cmax']:9.6f} | "
            f"{completion_text:15s} | "
            f"{task_count_text:11s} | "
            f"{result['gantt_path']}"
        )


def solve_algorithm(
    algorithm: str, data: dict, seed: int, time_limit: float
) -> tuple[Solution, list[dict]]:
    if algorithm == "SA":
        return solve_sa(data, seed=seed, time_limit=time_limit)
    if algorithm == "TS":
        return solve_ts(data, seed=seed, time_limit=time_limit)
    if algorithm == "GA-SA":
        return solve_ga_sa(data, seed=seed, time_limit=time_limit)
    if algorithm == "PSO-SA":
        return solve_pso_sa(data, seed=seed, time_limit=time_limit)
    raise ValueError(f"Unknown algorithm: {algorithm}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    instance = load_or_build_instance(root)
    all_results = []
    for offset, algorithm_name in enumerate(ALGORITHMS_TO_RUN):
        print(f"Running {algorithm_name} for {TIME_LIMIT:.0f}s ...", flush=True)
        best_solution, search_history = solve_algorithm(
            algorithm_name,
            instance,
            seed=SEED + offset,
            time_limit=TIME_LIMIT,
        )
        gantt_path = export_results(algorithm_name, instance, best_solution, root)
        final_history = search_history[-1] if search_history else {"iteration": 0, "elapsed": 0.0}
        all_results.append(
            {
                "algorithm": algorithm_name,
                "elapsed": final_history["elapsed"],
                "iterations": final_history["iteration"],
                "cmax": best_solution.cmax,
                "completion": best_solution.completion,
                "task_counts": [len(seq) for seq in best_solution.sequences],
                "gantt_path": gantt_path,
            }
        )

    print_final_summary(all_results)
