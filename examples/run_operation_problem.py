"""Example script for setting up and solving an operation problem, utilizing the high-level API."""

import fledge.api


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'

    # Run operation problem.
    fledge.api.run_operation_problem(
        scenario_name,
        print_results=True,
        recreate_database=True,
        store_results=True
    )


if __name__ == '__main__':
    main()
