"""Plotting entrypoint development script"""

import mesmo


def main():
    # Settings.
    scenario_name = "singapore_geylang"

    results_path = mesmo.utils.get_results_path("run_operation_problem", scenario_name)
    results = mesmo.api.run_nominal_operation_problem(scenario_name, results_path=results_path, store_results=False)

    # TODO: Debug re-loading of results from files
    # results = mesmo.problems.Results().load(results_path)
    run_results = results.get_run_results()

    mesmo.plots.der_active_power_time_series(run_results, results_path)
    mesmo.plots.der_reactive_power_time_series(run_results, results_path)
    mesmo.plots.der_apparent_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_active_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_reactive_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_apparent_power_time_series(run_results, results_path)

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
