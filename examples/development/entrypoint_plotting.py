"""Plotting entrypoint development script"""

import mesmo
from mesmo import plots


def main():
    # Settings.
    scenario_name = "singapore_geylang"
    # TODO: Split singapore_all scenario into separate folder

    mesmo.utils.cleanup()
    results_path = mesmo.utils.get_results_path("run_operation_problem", scenario_name)
    results_raw = mesmo.api.run_nominal_operation_problem(
        scenario_name, results_path=results_path, store_results=False, recreate_database=False
    )
    results = results_raw.get_run_results()

    # Roundtrip save/load to/from JSON, just for demonstration
    with open(results_path / "results.json", "w", encoding="utf-8") as file:
        print("Dumping results to file")
        file.write(results.model_dump_json())
    with open(results_path / "results.json", "r", encoding="utf-8") as file:
        print("Loading results from file")
        results = mesmo.data_models.RunResults.model_validate_json(file.read())

    # Sample plotting to file, just for demonstration
    plots.plot_to_file(plots.der_active_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_reactive_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_apparent_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_aggregated_active_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_aggregated_reactive_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.der_aggregated_apparent_power_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.node_voltage_per_unit_time_series, results=results, results_path=results_path)
    plots.plot_to_file(plots.node_aggregated_voltage_per_unit_time_series, results=results, results_path=results_path)

    # Sample JSON return
    print(plots.plot_to_json(plots.der_active_power_time_series, results=results))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
