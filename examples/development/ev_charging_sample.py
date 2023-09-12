"""Example script for EV charging study."""

import numpy as np
import pathlib
import scipy.sparse as sp

import mesmo


def main():
    # Settings.
    scenario_basename = "singapore_geylang"
    results_path = mesmo.utils.get_results_path(__file__, scenario_basename)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.config.config["paths"]["additional_data"].append(
        (
            pathlib.Path(__file__).parent
            / ".."
            / ".."
            / ".."
            / "data_processing"
            / "singapore_synthetic_grid_to_mesmo"
            / "output"
        )
    )
    mesmo.data_interface.recreate_database()

    # Obtain problems.
    problem_dict = mesmo.problems.ProblemDict(
        {
            "Baseline scenario": mesmo.problems.NominalOperationProblem(f"{scenario_basename}"),
            "100% EVs (uncontrolled)": mesmo.problems.NominalOperationProblem(f"{scenario_basename}"),
        }
    )

    # Modify problems.
    for key in problem_dict:
        if key in ["25% EVs (smart charging)", "75% EVs (smart charging)", "100% EVs (smart charging)"]:
            problem_dict[key].optimization_problem.define_variable(
                "branch_peak_power", branch=problem_dict[key].electric_grid_model.branches
            )
            for branch_index, _ in enumerate(problem_dict[key].electric_grid_model.branches):
                problem_dict[key].optimization_problem.define_constraint(
                    (
                        "variable",
                        np.ones((len(problem_dict[key].timesteps), 1)),
                        dict(
                            name="branch_peak_power",
                            branch=problem_dict[key].electric_grid_model.branches[branch_index],
                        ),
                    ),
                    ">=",
                    (
                        "variable",
                        sp.eye(len(problem_dict[key].timesteps)),
                        dict(
                            name="branch_power_magnitude_vector_1",
                            timestep=problem_dict[key].timesteps,
                            branch=problem_dict[key].electric_grid_model.branches[branch_index],
                        ),
                    ),
                    (
                        "variable",
                        sp.eye(len(problem_dict[key].timesteps)),
                        dict(
                            name="branch_power_magnitude_vector_2",
                            timestep=problem_dict[key].timesteps,
                            branch=problem_dict[key].electric_grid_model.branches[branch_index],
                        ),
                    ),
                )
            problem_dict[key].optimization_problem.define_objective(("variable", 1e9, dict(name="branch_peak_power")))
        elif key in ["100% EVs (peak shaving)"]:
            # TODO: Label optimization variables with 'per_unit' for better understanding?
            problem_dict[key].optimization_problem.define_variable("der_peak_power")
            problem_dict[key].optimization_problem.define_constraint(
                ("variable", np.ones((len(problem_dict[key].timesteps), 1)), dict(name="der_peak_power")),
                "<=",
                (
                    "variable",
                    (
                        sp.block_diag(
                            [np.array([np.real(problem_dict[key].electric_grid_model.der_power_vector_reference)])]
                            * len(problem_dict[key].timesteps)
                        )
                    ),
                    dict(
                        name="der_active_power_vector",
                        timestep=problem_dict[key].timesteps,
                        der=problem_dict[key].electric_grid_model.ders,
                    ),
                ),
            )
            problem_dict[key].optimization_problem.define_objective(("variable", -1e9, dict(name="der_peak_power")))
        elif key in ["100% EVs (price based)"]:
            price_data = mesmo.data_interface.PriceData(
                f"{scenario_basename}_100percent", price_type="singapore_wholesale"
            )
            problem_dict[key].linear_electric_grid_model_set.define_optimization_parameters(
                problem_dict[key].optimization_problem, price_data
            )
            problem_dict[key].der_model_set.define_optimization_parameters(
                problem_dict[key].optimization_problem, price_data
            )

    # Solve problems.
    problem_dict.solve()

    # Obtain results.
    results_dict = problem_dict.get_results()
    results_sets = dict()
    results_sets["_nominal"] = {
        key: value
        for key, value in results_dict.items()
        if key in ["Baseline scenario", "25% EVs (uncontrolled)", "75% EVs (uncontrolled)", "100% EVs (uncontrolled)"]
    }
    results_sets["_nominal_ev_only"] = {
        key: value
        for key, value in results_dict.items()
        if key in ["25% EVs (uncontrolled)", "75% EVs (uncontrolled)", "100% EVs (uncontrolled)"]
    }

    # Make plots.
    for results_label, results_set in results_sets.items():
        if len(results_set) > 0:
            mesmo.plots.plot_histogram_cumulative_branch_utilization(
                results_set, results_path, filename_suffix=results_label, branch_type="transformer", vertical_line=0.9
            )
            mesmo.plots.plot_histogram_cumulative_branch_utilization(
                results_set, results_path, filename_suffix=results_label, branch_type="line", vertical_line=0.9
            )
            mesmo.plots.plot_histogram_node_utilization(
                results_set,
                results_path,
                filename_suffix=results_label,
            )
            mesmo.plots.plot_aggregate_timeseries_der_power(
                results_set,
                results_path,
                filename_suffix=results_label,
                value_unit_label="MW",
                der_type_labels={"fixed_load": "Base load", "flexible_ev_charger": "EV charging"},
            )

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
