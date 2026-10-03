"""Run the main VNS-HE implementation with Lemma 2 candidate sequencing."""

from __future__ import annotations

from benchmark_common import base_arg_parser, run_vns_family


def main() -> None:
    args = base_arg_parser("Run VNS-HE on the 36 benchmark cases.").parse_args()
    run_vns_family(
        algorithm="VNS-HE",
        module_file="VNS-HE.py",
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
