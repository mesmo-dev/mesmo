"""Plotting entrypoint development script"""

import mesmo


def main():
    # Settings.
    scenario_name = "singapore_geylang"
    # TODO: Split singapore_all scenario into separate folder

    mesmo.utils.cleanup()
    results_path = mesmo.utils.get_results_path("run_operation_problem", scenario_name)
    results = mesmo.api.run_nominal_operation_problem(
        scenario_name, results_path=results_path, store_results=False, recreate_database=False
    )
    run_results = results.get_run_results()

    # Roundtrip save/load to/from JSON, just for demonstration
    with open(results_path / "run_results.json", "w", encoding="utf-8") as file:
        print("Dumping results to file")
        file.write(run_results.model_dump_json())
    with open(results_path / "run_results.json", "r", encoding="utf-8") as file:
        print("Loading results from file")
        run_results = mesmo.data_models.RunResults.model_validate_json(file.read())

    # TODO: Return JSON object, function should take run_id as input
    mesmo.plots.der_active_power_time_series(run_results, results_path)
    mesmo.plots.der_reactive_power_time_series(run_results, results_path)
    mesmo.plots.der_apparent_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_active_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_reactive_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_apparent_power_time_series(run_results, results_path)
    mesmo.plots.node_voltage_per_unit_time_series(run_results, results_path)
    mesmo.plots.node_aggregated_voltage_per_unit_time_series(run_results, results_path)

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
