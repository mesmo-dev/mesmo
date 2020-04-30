"""Example script for setting up and solving an operation problem."""

import os

import cobmo.database_interface
import fledge.config
import fledge.database_interface
import fledge.optimization_problems


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    results_path = (
        os.path.join(
            fledge.config.config['paths']['results'],
            f'run_operation_problem_{scenario_name}_{fledge.config.get_timestamp()}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()
    cobmo.database_interface.recreate_database()  # TODO: Incorporate cobmo recreate_database into fledge.

    # Obtain operation problem.
    operation_problem = fledge.optimization_problems.OperationProblem(scenario_name)

    # Solve operation problem.
    operation_problem.solve_optimization()

    # Obtain results.
    results = operation_problem.get_optimization_results()

    # Print results.
    print(results)

    # Store results as CSV.
    results.to_csv(results_path)


if __name__ == '__main__':
    main()
