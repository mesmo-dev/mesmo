"""Application programming interface (API) module for high-level interface functions to run FLEDGE."""

import os

import cobmo.database_interface
import fledge.config
import fledge.data_interface
import fledge.problems

logger = fledge.config.get_logger(__name__)


def run_nominal_operation_problem(
        scenario_name: str,
        recreate_database: bool = True,
        print_results: bool = False,
        store_results: bool = True,
        results_path: str = None
) -> fledge.data_interface.ResultsDict:
    """Set up and solve a nominal operation problem for the given scenario."""

    # Instantiate results directory.
    if store_results:
        if results_path is None:
            results_path = (
                os.path.join(
                    fledge.config.config['paths']['results'],
                    f'run_operation_problem_{scenario_name}_{fledge.config.get_timestamp()}'
                )
            )
        os.mkdir(results_path)

    # Recreate / overwrite database.
    if recreate_database:
        fledge.data_interface.recreate_database()
        cobmo.database_interface.recreate_database()

    # Obtain operation problem.
    operation_problem = fledge.problems.NominalOperationProblem(scenario_name)

    # Solve operation problem.
    operation_problem.solve()

    # Obtain results.
    results = operation_problem.results

    # Print results.
    if print_results:
        print(f"results = \n{results}")

    # Store results as CSV.
    if store_results:
        results.to_csv(results_path)

    return results


def run_optimal_operation_problem(
        scenario_name: str,
        recreate_database: bool = True,
        print_results: bool = False,
        store_results: bool = True,
        results_path: str = None
) -> fledge.data_interface.ResultsDict:
    """Set up and solve an optimal operation problem for the given scenario."""

    # Instantiate results directory.
    if store_results:
        if results_path is None:
            results_path = (
                os.path.join(
                    fledge.config.config['paths']['results'],
                    f'run_optimal_operation_problem_{scenario_name}_{fledge.config.get_timestamp()}'
                )
            )
        os.mkdir(results_path)

    # Recreate / overwrite database.
    if recreate_database:
        fledge.data_interface.recreate_database()
        cobmo.database_interface.recreate_database()

    # Obtain operation problem.
    operation_problem = fledge.problems.OptimalOperationProblem(scenario_name)

    # Solve operation problem.
    operation_problem.solve_optimization()

    # Obtain results.
    results = operation_problem.get_optimization_results()
    results.update(operation_problem.get_optimization_dlmps())

    # Print results.
    if print_results:
        print(f"results = \n{results}")

    # Store results as CSV.
    if store_results:
        results.to_csv(results_path)

    return results
