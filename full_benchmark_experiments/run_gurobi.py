"""Run the MILP once per case with the configured Gurobi time limit."""

from __future__ import annotations

from benchmark_common import base_arg_parser, run_gurobi


def main() -> None:
    args = base_arg_parser("Run Gurobi once per benchmark case.").parse_args()
    run_gurobi(
        experiment_id=args.experiment_id,
        case_filter=args.cases,
        gurobi_time=args.gurobi_time,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
