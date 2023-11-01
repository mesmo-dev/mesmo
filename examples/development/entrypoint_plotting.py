"""Plotting entrypoint development script"""

import mesmo
from mesmo import plots


def main():
    # Settings.
    scenario_name = "singapore_geylang"

    results_path = mesmo.utils.get_results_path("run_operation_problem", scenario_name)

    results_raw = mesmo.api.run_nominal_operation_problem(
        scenario_name, results_path=results_path, store_results=False, recreate_database=False
    )
    results = results_raw.get_run_results(scenario_name=scenario_name)

    # Write results to compressed JSON
    print("Writing results to file")
    results.to_json_file(mesmo.config.base_path / "results" / "results.json.bz2", compress=True)

    # Load results from compressed JSON
    print("Loading results from file")
    results = mesmo.data_models.RunResults.from_json_file(
        mesmo.config.base_path / "results" / "results.json.bz2", decompress=True
    )

    # Sample plotting to file
    plots.plot_to_file(plots.der_active_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_reactive_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_apparent_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_aggregated_active_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_aggregated_reactive_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_aggregated_apparent_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.node_voltage_per_unit_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.node_aggregated_voltage_per_unit_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.electric_grid_assets, results=results, results_path=results_path)
    plots.plot_to_file(plots.electric_grid_node_voltage_nominal, results=results, results_path=results_path)
    plots.plot_to_file(plots.electric_grid_node_voltage_magnitude_min, results=results, results_path=results_path)

    # Sample JSON return
    print(plots.plot_to_json(plots.der_active_power_time_series, results=results))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
