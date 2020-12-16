"""Application programming interface (API) module for high-level interface functions to run FLEDGE."""

import os

import fledge.config
import fledge.data_interface
import fledge.problems
import fledge.utils

logger = fledge.config.get_logger(__name__)


def run_nominal_operation_problem(
        scenario_name: str,
        recreate_database: bool = True,
        print_results: bool = False,
        store_results: bool = True,
        results_path: str = None
) -> fledge.problems.Results:
    """Set up and solve a nominal operation problem for the given scenario."""

    # Instantiate results directory.
    if store_results and (results_path is None):
        results_path = fledge.utils.get_results_path('run_operation_problem', scenario_name)

    # Recreate / overwrite database.
    if recreate_database:
        fledge.data_interface.recreate_database()

    # Obtain operation problem.
    operation_problem = fledge.problems.NominalOperationProblem(scenario_name)

    # Solve operation problem.
    operation_problem.solve()

    # Obtain results.
    results = operation_problem.get_results()

    # Print results.
    if print_results:
        print(f"results = \n{results}")

    # Store results as CSV.
    if store_results:
        results.save(results_path)
        logger.info(f"Results are stored in: {results_path}")

    return results


def run_optimal_operation_problem(
        scenario_name: str,
        recreate_database: bool = True,
        print_results: bool = False,
        store_results: bool = True,
        results_path: str = None
) -> fledge.problems.Results:
    """Set up and solve an optimal operation problem for the given scenario."""

    # Instantiate results directory.
    if store_results and (results_path is None):
        results_path = fledge.utils.get_results_path('run_optimal_operation_problem', scenario_name)

    # Recreate / overwrite database.
    if recreate_database:
        fledge.data_interface.recreate_database()

    # Obtain operation problem.
    operation_problem = fledge.problems.OptimalOperationProblem(scenario_name)

    # Solve operation problem.
    operation_problem.solve()

    # Obtain results.
    results = operation_problem.get_results()

    # Print results.
    if print_results:
        print(f"results = \n{results}")

    # Store results as CSV.
    if store_results:
        results.save(results_path)
        logger.info(f"Results are stored in: {results_path}")

    return results
