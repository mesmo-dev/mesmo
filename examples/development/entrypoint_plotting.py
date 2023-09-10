"""Plotting entrypoint development script"""

import mesmo


def main():
    # Settings.
    scenario_name = "singapore_geylang"

    results_path = mesmo.utils.get_results_path("run_operation_problem", scenario_name)
    mesmo.api.run_nominal_operation_problem(scenario_name, results_path=results_path)

    results = mesmo.problems.Results().load(results_path)

    print(results)


if __name__ == "__main__":
    main()
