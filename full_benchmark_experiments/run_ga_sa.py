"""Run original priority-decoded GA-SA on the shared 36-case benchmark."""

from __future__ import annotations

from benchmark_common import base_arg_parser, run_meta_algorithm


def main() -> None:
    args = base_arg_parser("Run GA-SA on the 36 benchmark cases.").parse_args()
    run_meta_algorithm(
        algorithm="GA-SA",
        experiment_id=args.experiment_id,
        case_filter=args.cases,
        replications=args.replications,
        small_time=args.small_time,
        medium_time=args.medium_time,
        large_time=args.large_time,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
