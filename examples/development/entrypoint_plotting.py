"""Plotting entrypoint development script"""

import mesmo


def main():
    # Settings.
    scenario_name = "singapore_geylang"
    # TODO: Split singapore_all into folder

    results_path = mesmo.utils.get_results_path("run_operation_problem", scenario_name)
    results = mesmo.api.run_nominal_operation_problem(scenario_name, results_path=results_path, store_results=False)

    # TODO: Debug re-loading of results from files
    # results = mesmo.problems.Results().load(results_path)
    run_results = results.get_run_results()

    # TODO: Return JSON object, should probably take run_id as input
    mesmo.plots.der_active_power_time_series(run_results, results_path)
    mesmo.plots.der_reactive_power_time_series(run_results, results_path)
    mesmo.plots.der_apparent_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_active_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_reactive_power_time_series(run_results, results_path)
    mesmo.plots.der_aggregated_apparent_power_time_series(run_results, results_path)
    mesmo.plots.node_voltage_per_unit_time_series(run_results, results_path)
    mesmo.plots.node_aggregated_voltage_per_unit_time_series(run_results, results_path)

    # TODO: RuntimeError: no validator found for <class 'pandas.core.indexes.base.Index'>, see `arbitrary_types_allowed` in Config
    # TODO: File "D:\py_venvs\_PyCodes\SebastianT_VP_Test\mesmo_repo\mesmo\data_models\model_index.py", line 8, in <module> class ElectricGridModelIndex(base_model.BaseModel):
    # TODO: python 3.8.10, pydantic 1.10.12 -> move to poetry

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
