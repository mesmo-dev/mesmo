"""Project SITEM baseline scenario evaluation script."""

import fledge.api


def main():

    # Settings.
    scenario_name = 'ema_sample_grid'

    # Obtain operation / power flow problem solution.
    results = fledge.api.run_nominal_operation_problem(scenario_name)

    # Print results.
    print(results)


if __name__ == '__main__':
    main()
