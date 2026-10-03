"""Calculate the shared adjusted lower bounds for all selected benchmark cases."""

from __future__ import annotations

from benchmark_common import (
    LOWER_BOUND_FIELDS,
    base_arg_parser,
    build_raw_instance,
    case_instance_seed,
    compute_lower_bound,
    result_root,
    save_instance,
    selected_cases,
    prepare_common_data,
    write_csv,
    write_experiment_config,
)


def main() -> None:
    args = base_arg_parser("Compute lower bounds for the 36 benchmark cases.").parse_args()
    output_dir = result_root(args.experiment_id) / "lower_bound"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_config(args.experiment_id, {"runner": "lower_bound"})

    rows = []
    for case_index, case in selected_cases(args.cases):
        seed = case_instance_seed(case_index)
        raw_data = build_raw_instance(case, seed)
        save_instance(args.experiment_id, case, raw_data)
        lb_info = compute_lower_bound(prepare_common_data(raw_data))
        rows.append(
            {
                "case": case.name,
                "scale": case.scale,
                "n": case.n,
                "m": case.m,
                "lambda": case.lambd,
                "instance_seed": seed,
                **lb_info,
            }
        )
        print(f"[LB] {case.name}: LB={lb_info['lb']:.6f}", flush=True)
        write_csv(output_dir / "lower_bounds.csv", rows, LOWER_BOUND_FIELDS)

    print(f"[LB] saved: {output_dir}")


if __name__ == "__main__":
    main()
