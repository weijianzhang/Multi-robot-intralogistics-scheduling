"""
Gurobi MILP for multi-AGV scheduling with load-dependent deterioration
and AGV-dependent loaded/empty travel times.

The image in the prompt gives parameter ranges rather than exact data.
This script generates a reproducible n=20, m=3 instance from those ranges,
then builds and solves the corrected position-indexed MILP.
"""

from __future__ import annotations

import random
from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB


N = 50
M = 3
SEED = 20260722

# Parameter ranges from the image.
T0_RANGE = (0.0, 2.0)
L_RANGE = (0.2, 2.0)
U_RANGE = (0.2, 2.0)
W_RANGE = (0.5, 3.0)
Q_RANGE = (6.0, 8.0)
TL_RANGE = (2.0, 5.0)
TR_RANGE = (1.5, 3.5)
BETA_RANGE = (0.01, 0.03)
LAMBDA = 0.2


def uniform_list(rng: random.Random, size: int, lo: float, hi: float) -> list[float]:
    return [round(rng.uniform(lo, hi), 4) for _ in range(size)]


def uniform_matrix(
    rng: random.Random, rows: int, cols: int, lo: float, hi: float
) -> list[list[float]]:
    return [[round(rng.uniform(lo, hi), 4) for _ in range(cols)] for _ in range(rows)]


def build_instance(seed: int = SEED) -> dict:
    rng = random.Random(seed)

    t0 = round(rng.uniform(*T0_RANGE), 4)
    L = uniform_list(rng, N, *L_RANGE)
    U = uniform_list(rng, N, *U_RANGE)
    w = uniform_list(rng, N, *W_RANGE)
    Q = uniform_list(rng, M, *Q_RANGE)
    TL = uniform_matrix(rng, N, M, *TL_RANGE)
    TR = uniform_matrix(rng, N, M, *TR_RANGE)
    beta = uniform_list(rng, M, *BETA_RANGE)

    gamma = [[0.0 for _ in range(M)] for _ in range(N)]
    q = [[0.0 for _ in range(M)] for _ in range(N)]
    alpha = [[0.0 for _ in range(M)] for _ in range(N)]

    for i in range(N):
        for k in range(M):
            gamma[i][k] = beta[k] * (1.0 + LAMBDA * w[i] / Q[k])
            q[i][k] = (1.0 + gamma[i][k]) ** 2
            alpha[i][k] = (
                (1.0 + gamma[i][k]) ** 2 * L[i]
                + (1.0 + gamma[i][k]) * (TL[i][k] + U[i])
                + TR[i][k]
            )

    return {
        "n": N,
        "m": M,
        "seed": seed,
        "t0": t0,
        "lambda": LAMBDA,
        "L": L,
        "U": U,
        "w": w,
        "Q": Q,
        "TL": TL,
        "TR": TR,
        "beta": beta,
        "gamma": gamma,
        "q": q,
        "alpha": alpha,
    }


def compute_big_m(data: dict) -> tuple[float, float]:
    """Return a safe completion-time upper bound H and Big-M value G."""
    q_max = max(max(row) for row in data["q"])
    alpha_max = max(max(row) for row in data["alpha"])

    # A conservative horizon: one AGV executes all N jobs, each step using
    # the largest deterioration multiplier and largest intercept.
    horizon = data["t0"]
    for _ in range(data["n"]):
        horizon = q_max * horizon + alpha_max

    # Enough to deactivate both lower and upper recursive constraints.
    big_m = q_max * horizon + alpha_max
    return horizon, big_m


def solve(data: dict, time_limit: float | None = None, mip_gap: float | None = None):
    """Solve the position-indexed MILP; no heuristic sequencing rule is imposed."""
    n = data["n"]
    m = data["m"]
    jobs = range(n)
    agvs = range(m)
    positions = range(1, n + 1)
    stages = range(0, n + 1)

    horizon, big_m = compute_big_m(data)

    model = gp.Model("load_dependent_multi_agv_scheduling")
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # y assigns a task to an ADR/position; z marks occupied sequence positions.
    y = model.addVars(jobs, agvs, positions, vtype=GRB.BINARY, name="y")
    z = model.addVars(agvs, positions, vtype=GRB.BINARY, name="z")
    Cpos = model.addVars(agvs, stages, lb=0.0, ub=horizon, name="Cpos")
    Ck = model.addVars(agvs, lb=0.0, ub=horizon, name="Ck")
    Cmax = model.addVar(lb=0.0, ub=horizon, name="Cmax")

    model.setObjective(Cmax, GRB.MINIMIZE)

    # Each job is assigned exactly once.
    model.addConstrs(
        (
            gp.quicksum(y[i, k, r] for k in agvs for r in positions) == 1
            for i in jobs
        ),
        name="assign_once",
    )

    # At most one job can occupy each AGV-position pair.
    model.addConstrs(
        (
            gp.quicksum(y[i, k, r] for i in jobs) == z[k, r]
            for k in agvs
            for r in positions
        ),
        name="position_occupied",
    )

    # No gaps in each AGV sequence.
    model.addConstrs(
        (z[k, r + 1] <= z[k, r] for k in agvs for r in range(1, n)),
        name="position_continuity",
    )

    # Capacity feasibility. In this generated instance w_i <= Q_k always holds,
    # but the constraint is kept for the mathematical model.
    model.addConstrs(
        (
            data["w"][i] * y[i, k, r] <= data["Q"][k]
            for i in jobs
            for k in agvs
            for r in positions
        ),
        name="capacity",
    )

    # Initial available time.
    model.addConstrs((Cpos[k, 0] == data["t0"] for k in agvs), name="initial_time")

    # Completion-time recursion for occupied positions.
    model.addConstrs(
        (
            Cpos[k, r]
            >= data["q"][i][k] * Cpos[k, r - 1]
            + data["alpha"][i][k]
            - big_m * (1 - y[i, k, r])
            for i in jobs
            for k in agvs
            for r in positions
        ),
        name="completion_lb",
    )

    model.addConstrs(
        (
            Cpos[k, r]
            <= data["q"][i][k] * Cpos[k, r - 1]
            + data["alpha"][i][k]
            + big_m * (1 - y[i, k, r])
            for i in jobs
            for k in agvs
            for r in positions
        ),
        name="completion_ub",
    )

    # Empty positions inherit the previous completion time.
    model.addConstrs(
        (Cpos[k, r] >= Cpos[k, r - 1] for k in agvs for r in positions),
        name="empty_inherit_lb",
    )

    model.addConstrs(
        (
            Cpos[k, r] <= Cpos[k, r - 1] + big_m * z[k, r]
            for k in agvs
            for r in positions
        ),
        name="empty_inherit_ub",
    )

    # Final completion time for each AGV and makespan definition.
    model.addConstrs((Ck[k] == Cpos[k, n] for k in agvs), name="agv_final")
    model.addConstrs((Cmax >= Ck[k] for k in agvs), name="makespan")

    model.optimize()
    return model, y, z, Cpos, Ck, Cmax, horizon, big_m


def plot_gantt(rows: list[dict], cmax: float, output_path: Path) -> None:
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))
    cmap = plt.get_cmap("tab20")
    agvs = sorted({row["agv"] for row in rows})
    y_pos = {agv: idx for idx, agv in enumerate(agvs)}

    for row in rows:
        agv = row["agv"]
        job = row["job"]
        start = row["start_time"]
        finish = row["finish_time"]
        duration = finish - start
        color = cmap((job - 1) % 20)

        ax.barh(
            y_pos[agv],
            duration,
            left=start,
            height=0.55,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(
            start + duration / 2,
            y_pos[agv],
            f"J{job}",
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
    ax.set_title("Multi-AGV Scheduling Gantt Chart")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_results(data: dict, model, y, z, Cpos, Ck, Cmax, horizon: float, big_m: float):
    out_dir = Path(__file__).resolve().parent

    rows = []
    if model.SolCount > 0:
        for k in range(data["m"]):
            for r in range(1, data["n"] + 1):
                if z[k, r].X > 0.5:
                    assigned_job = None
                    for i in range(data["n"]):
                        if y[i, k, r].X > 0.5:
                            assigned_job = i
                            break

                    if assigned_job is not None:
                        i = assigned_job
                        rows.append(
                            {
                                "agv": k + 1,
                                "position": r,
                                "job": i + 1,
                                "start_time": Cpos[k, r - 1].X,
                                "finish_time": Cpos[k, r].X,
                                "L_i": data["L"][i],
                                "U_i": data["U"][i],
                                "w_i": data["w"][i],
                                "Q_k": data["Q"][k],
                                "T_L_ik": data["TL"][i][k],
                                "T_R_ik": data["TR"][i][k],
                                "beta_k": data["beta"][k],
                                "gamma_ik": data["gamma"][i][k],
                                "q_ik": data["q"][i][k],
                                "alpha_ik": data["alpha"][i][k],
                            }
                        )

    gantt_path = out_dir / "agv_gantt_n20_m3.png"
    if rows:
        plot_gantt(rows, Cmax.X, gantt_path)

    print("\n========== Solve Summary ==========")
    print(f"Status code: {model.Status}")
    print(f"Horizon upper bound H: {horizon:.6f}")
    print(f"Big-M G: {big_m:.6f}")

    if model.SolCount == 0:
        print("No feasible solution found.")
        return

    print(f"Objective Cmax: {Cmax.X:.6f}")
    print(f"MIPGap: {model.MIPGap:.6g}" if model.IsMIP else "MIPGap: N/A")
    for k in range(data["m"]):
        task_sequence = [row["job"] for row in rows if row["agv"] == k + 1]
        print(
            f"AGV {k + 1}: C_k={Ck[k].X:.6f}, "
            f"sequence={task_sequence if task_sequence else 'empty'}"
        )

    print("\nSaved files:")
    if rows:
        print(gantt_path)


if __name__ == "__main__":
    instance = build_instance()
    solved = solve(instance, time_limit=600, mip_gap=None)
    export_results(instance, *solved)
