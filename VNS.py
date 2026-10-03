"""
VNS for the multi-AGV scheduling problem with a real-coded two-segment encoding.

The solution vector V has length 2n:
    - V[0:n] encodes job-to-AGV assignment by the ceiling map.
    - V[n:2n] encodes job priority inside the assigned AGV.

This version does not use any structural priority repair rule. Decoding groups
priorities by round(priority / 10) and breaks group ties by task ID.
It retains the time checks formerly provided by VNS_time_guard.py.
Every neighborhood is evaluated by decoding the full assignment and sequence.
The four neighborhoods are:
    N1: bottleneck critical-task transfer
    N2: limited-candidate bottleneck task exchange
    N3: short random encoding-segment AGV reassignment
    N4: small ruin-and-greedy-recreate assignment repair
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


# Standalone defaults; batch runners override the instance size and time budget.
N = 100
M = 3
SEED = 20260722
TIME_LIMIT = 60.0
MAX_ITER = 3000
NO_IMPROVE_LIMIT = 500
SHAKING_LEVELS = 1
ITER_PRINT_INTERVAL = 1
SEGMENT_REASSIGN_TRIALS = 30
SEGMENT_REVERSAL_TRIALS = 30
CRITICAL_JOB_LIMIT = 5
EXCHANGE_TARGET_LIMIT = 12
RUIN_RECREATE_TRIALS = 10
RUIN_MIN_SIZE = 3
RUIN_MAX_SIZE = 12
SEGMENT_MAX_SIZE = 20
EPS = 1e-9

T0_RANGE = (0.0, 2.0)
L_RANGE = (0.2, 2.0)
U_RANGE = (0.2, 2.0)
W_RANGE = (0.5, 3.0)
Q_RANGE = (6.0, 8.0)
TL_RANGE = (2.0, 5.0)
TR_RANGE = (1.5, 3.5)
BETA_RANGE = (0.01, 0.03)
LAMBDA = 0.2


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
    """Precompute load-dependent multipliers and task contribution intercepts."""
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
    return [k for k in range(data["m"]) if data["w"][job] <= data["Q"][k] + EPS]


def assignment_value(agv: int) -> float:
    return agv + 0.5


def assignment_from_vector(vector: list[float], data: dict) -> list[int]:
    assignment = []
    for i in range(data["n"]):
        value = min(max(vector[i], EPS), data["m"])
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
    """Repair assignments and evaluate quantized-priority task sequences."""
    vector = repair_capacity(vector, data)
    n = data["n"]
    m = data["m"]
    assignment = assignment_from_vector(vector, data)
    sequences = [[] for _ in range(m)]

    for job, agv in enumerate(assignment):
        sequences[agv].append(job)

    for agv in range(m):
        sequences[agv].sort(key=lambda job: (round(vector[n + job] / 10.0), job))

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


def clone_sequences(sequences: list[list[int]]) -> list[list[int]]:
    return [seq[:] for seq in sequences]


def partial_cmax(sequences: list[list[int]], data: dict) -> float:
    completions = []
    for agv, seq in enumerate(sequences):
        completions.append(agv_sequence_completion(seq, agv, data))
    return max(completions)


def agv_sequence_completion(seq: list[int], agv: int, data: dict) -> float:
    current_time = data["t0"]
    for job in seq:
        current_time = data["q"][job][agv] * current_time + data["alpha"][job][agv]
    return current_time


def build_initial_solution(data: dict, rng: random.Random) -> Solution:
    """Construct a greedy allocation without initial variable neighborhood descent."""
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


def bottleneck_agv(solution: Solution) -> int:
    return max(range(len(solution.completion)), key=lambda agv: solution.completion[agv])


def better(candidate: Solution | None, incumbent: Solution) -> bool:
    return candidate is not None and candidate.cmax < incumbent.cmax - 1e-8


def removal_gain(solution: Solution, agv: int, job: int, data: dict) -> float:
    reduced_seq = [item for item in solution.sequences[agv] if item != job]
    return solution.completion[agv] - agv_sequence_completion(reduced_seq, agv, data)


def top_removal_jobs(
    solution: Solution, agv: int, data: dict, limit: int
) -> list[int]:
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


def time_expired(deadline: float | None) -> bool:
    """Use an absolute perf_counter deadline shared by the search and neighborhoods."""
    return deadline is not None and time.perf_counter() >= deadline


def neighborhood_critical_task_transfer(
    solution: Solution, data: dict, rng: random.Random, deadline: float | None = None
) -> Solution | None:
    """N1: search critical-task transfers with deadline checks between candidates."""
    source = bottleneck_agv(solution)
    best_candidate = None

    for job in top_removal_jobs(solution, source, data, CRITICAL_JOB_LIMIT):
        if time_expired(deadline):
            return best_candidate
        for target in range(data["m"]):
            if time_expired(deadline):
                return best_candidate
            if target == source or data["w"][job] > data["Q"][target] + EPS:
                continue
            trial = clone_sequences(solution.sequences)
            trial[source].remove(job)
            trial[target].append(job)
            candidate = evaluate_sequences(trial, data)
            if better(candidate, solution) and (
                best_candidate is None or candidate.cmax < best_candidate.cmax
            ):
                best_candidate = candidate
    return best_candidate


def neighborhood_limited_candidate_exchange(
    solution: Solution, data: dict, rng: random.Random, deadline: float | None = None
) -> Solution | None:
    """N2: restrict exchange partners to a bounded pool around the bottleneck."""
    source = bottleneck_agv(solution)
    best_candidate = None

    for source_job in top_removal_jobs(solution, source, data, CRITICAL_JOB_LIMIT):
        if time_expired(deadline):
            return best_candidate
        for target in range(data["m"]):
            if time_expired(deadline):
                return best_candidate
            if target == source:
                continue
            for target_job in exchange_candidate_jobs(
                solution, source, target, source_job, data, rng
            ):
                if time_expired(deadline):
                    return best_candidate
                trial = clone_sequences(solution.sequences)
                trial[source].remove(source_job)
                trial[target].remove(target_job)
                trial[source].append(target_job)
                trial[target].append(source_job)
                candidate = evaluate_sequences(trial, data)
                if better(candidate, solution) and (
                    best_candidate is None or candidate.cmax < best_candidate.cmax
                ):
                    best_candidate = candidate
    return best_candidate


def neighborhood_short_segment_reassignment(
    solution: Solution, data: dict, rng: random.Random, deadline: float | None = None
) -> Solution | None:
    """N3: diversify allocations by reassigning a short segment of task IDs."""
    n = data["n"]
    best_candidate = None

    for _ in range(SEGMENT_REASSIGN_TRIALS):
        if time_expired(deadline):
            return best_candidate
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
        if better(candidate, solution) and (
            best_candidate is None or candidate.cmax < best_candidate.cmax
        ):
            best_candidate = candidate
    return best_candidate


def greedy_reinsert_removed_jobs(
    base_sequences: list[list[int]],
    removed_jobs: list[int],
    data: dict,
    deadline: float | None = None,
) -> Solution:
    sequences = clone_sequences(base_sequences)
    for job in sorted(
        removed_jobs,
        key=lambda item: min(data["alpha"][item][agv] for agv in feasible_agvs(data, item)),
        reverse=True,
    ):
        if time_expired(deadline):
            return evaluate_sequences(sequences, data)
        best_agv = None
        best_cmax = float("inf")
        for agv in feasible_agvs(data, job):
            if time_expired(deadline):
                return evaluate_sequences(sequences, data)
            trial = clone_sequences(sequences)
            trial[agv].append(job)
            trial_cmax = partial_cmax(trial, data)
            if trial_cmax < best_cmax:
                best_cmax = trial_cmax
                best_agv = agv
        sequences[best_agv].append(job)
    return evaluate_sequences(sequences, data)


def neighborhood_small_ruin_recreate(
    solution: Solution, data: dict, rng: random.Random, deadline: float | None = None
) -> Solution | None:
    """N4: remove critical/random tasks and evaluate greedy reconstruction."""
    source = bottleneck_agv(solution)
    best_candidate = None
    n = data["n"]
    max_remove = min(RUIN_MAX_SIZE, max(RUIN_MIN_SIZE, n // 40))

    for _ in range(RUIN_RECREATE_TRIALS):
        if time_expired(deadline):
            return best_candidate
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
        candidate = greedy_reinsert_removed_jobs(base_sequences, removed, data, deadline)
        if better(candidate, solution) and (
            best_candidate is None or candidate.cmax < best_candidate.cmax
        ):
            best_candidate = candidate
    return best_candidate


def clipped_assignment_value(value: float, data: dict) -> float:
    return min(max(value, EPS), data["m"] - EPS)


def continuous_shaking(
    solution: Solution,
    data: dict,
    rng: random.Random,
    level: int,
    deadline: float | None = None,
) -> Solution:
    """Perturb both encoding segments; abort shaking if the shared deadline expires."""
    n = data["n"]
    vector = solution.vector[:]
    level = max(1, min(level, SHAKING_LEVELS))
    assign_count = min(n, max(1, level + n // 10))
    priority_count = min(n, max(2, level * 2 + n // 5))
    assign_sigma = 0.18 + 0.12 * level
    priority_sigma = 0.75 + 0.65 * level

    assignment_jobs = rng.sample(range(n), assign_count)
    for job in assignment_jobs:
        if time_expired(deadline):
            return solution
        current_agv = assignment_from_vector(vector, data)[job]
        valid_agvs = feasible_agvs(data, job)
        if len(valid_agvs) > 1 and rng.random() < 0.45 + 0.1 * level:
            choices = [agv for agv in valid_agvs if agv != current_agv]
            target_agv = rng.choice(choices or valid_agvs)
            vector[job] = target_agv + rng.uniform(0.05, 0.95)
        else:
            vector[job] = clipped_assignment_value(
                vector[job] + rng.gauss(0.0, assign_sigma), data
            )

    priority_jobs = rng.sample(range(n), priority_count)
    for job in priority_jobs:
        if time_expired(deadline):
            return decode(vector, data)
        vector[n + job] += rng.gauss(0.0, priority_sigma)

    if level >= 3:
        left, right = sorted(rng.sample(range(n), 2))
        priorities = [vector[n + job] for job in range(n)]
        segment = priorities[left : right + 1]
        if rng.random() < 0.5:
            segment.reverse()
        else:
            rng.shuffle(segment)
        for offset, value in enumerate(segment):
            if time_expired(deadline):
                return decode(vector, data)
            vector[n + left + offset] = value

    if level == SHAKING_LEVELS:
        for _ in range(max(1, n // 8)):
            if time_expired(deadline):
                return decode(vector, data)
            a, b = rng.sample(range(n), 2)
            vector[n + a], vector[n + b] = vector[n + b], vector[n + a]

    return decode(vector, data)


def local_search(
    solution: Solution,
    data: dict,
    neighborhoods: list[Callable[[Solution, dict, random.Random, float | None], Solution | None]],
    rng: random.Random,
    deadline: float | None = None,
) -> Solution:
    current = solution
    index = 0
    while index < len(neighborhoods) and not time_expired(deadline):
        candidate = neighborhoods[index](current, data, rng, deadline)
        if better(candidate, current):
            current = candidate
            index = 0
        else:
            index += 1
    return current


def solve_vns(data: dict, seed: int = SEED, time_limit: float = TIME_LIMIT) -> tuple[Solution, list[dict]]:
    rng = random.Random(seed)
    start = time.perf_counter()
    deadline = start + time_limit

    neighborhoods = [
        neighborhood_critical_task_transfer,
        neighborhood_limited_candidate_exchange,
        neighborhood_short_segment_reassignment,
        neighborhood_small_ruin_recreate,
    ]
    current = build_initial_solution(data, rng)
    best = current
    history = [
        {
            "iteration": 0,
            "elapsed": 0.0,
            "shaking_level": 0,
            "candidate_cmax": best.cmax,
            "current_cmax": current.cmax,
            "best_cmax": best.cmax,
            "no_improve": 0,
            "event": "initial_local_optimum",
        }
    ]
    print("\n========== VNS Iteration Log ==========")
    print("iter | elapsed(s) | shake | candidate | current | best | no_improve | event")
    print(
        f"{0:4d} | {0.0:10.2f} | {0:5d} | {best.cmax:9.6f} | "
        f"{current.cmax:7.6f} | {best.cmax:7.6f} | {0:10d} | initial_local_optimum"
    )

    iteration = 0
    no_improve = 0
    shaking_level = 1

    while (
        iteration < MAX_ITER
        and no_improve < NO_IMPROVE_LIMIT
        and not time_expired(deadline)
    ):
        iteration += 1
        used_shaking_level = shaking_level
        shaken = continuous_shaking(current, data, rng, used_shaking_level, deadline)
        if time_expired(deadline):
            break
        candidate = local_search(shaken, data, neighborhoods, rng, deadline)

        if candidate.cmax < current.cmax - 1e-8:
            current = candidate
            shaking_level = 1
            event = "accept"
        else:
            shaking_level = 1 + (shaking_level % SHAKING_LEVELS)
            event = "reject"

        if current.cmax < best.cmax - 1e-8:
            best = current
            no_improve = 0
            event = "new_best"
        else:
            no_improve += 1

        elapsed = time.perf_counter() - start
        history.append(
            {
                "iteration": iteration,
                "elapsed": elapsed,
                "shaking_level": used_shaking_level,
                "candidate_cmax": candidate.cmax,
                "current_cmax": current.cmax,
                "best_cmax": best.cmax,
                "no_improve": no_improve,
                "event": event,
            }
        )
        if iteration % ITER_PRINT_INTERVAL == 0:
            print(
                f"{iteration:4d} | {elapsed:10.2f} | {used_shaking_level:5d} | "
                f"{candidate.cmax:9.6f} | {current.cmax:7.6f} | "
                f"{best.cmax:7.6f} | {no_improve:10d} | {event}",
                flush=True,
            )

    history.append(
        {
            "iteration": iteration,
            "elapsed": time.perf_counter() - start,
            "shaking_level": shaking_level,
            "candidate_cmax": best.cmax,
            "current_cmax": current.cmax,
            "best_cmax": best.cmax,
            "no_improve": no_improve,
            "event": "finished",
        }
    )
    return best, history


def plot_gantt(rows: list[dict], cmax: float, output_path: Path) -> None:
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
    ax.set_title("VNS Multi-AGV Scheduling Gantt Chart")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_results(data: dict, solution: Solution, history: list[dict], base_dir: Path) -> None:
    suffix = f"n{data['n']}_m{data['m']}"
    gantt_path = base_dir / f"vns_gantt_{suffix}.png"

    plot_gantt(solution.rows, solution.cmax, gantt_path)

    print("\n========== VNS Solve Summary ==========")
    print(f"Best Cmax: {solution.cmax:.6f}")
    for agv, seq in enumerate(solution.sequences):
        print(f"AGV {agv + 1}: C_k={solution.completion[agv]:.6f}, sequence={[job + 1 for job in seq]}")

    print("\nSaved files:")
    print(gantt_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    instance = load_or_build_instance(root)
    best_solution, search_history = solve_vns(instance)
    export_results(instance, best_solution, search_history, root)
