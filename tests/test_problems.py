"""Test problems."""

import pyomo.environ as pyo
import time
import unittest

import fledge.config
import fledge.database_interface
import fledge.problems

logger = fledge.config.get_logger(__name__)

# Check availability of optimization solver.
try:
    optimization_solver_available = pyo.SolverFactory(fledge.config.config['optimization']['solver_name']).available()
except Exception:
    optimization_solver_available = False


class TestProblems(unittest.TestCase):

    def test_optimal_operation_problem(self):
        # Get result.
        time_start = time.time()
        fledge.problems.OptimalOperationProblem('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test OptimalOperationProblem: Completed in {time_duration:.6f} seconds.")

    if optimization_solver_available:

        def test_optimal_operation_problem_methods(self):
            # Get result.
            time_start = time.time()
            operation_problem = fledge.problems.OptimalOperationProblem('singapore_tanjongpagar')
            operation_problem.solve_optimization()
            operation_problem.get_optimization_results()
            operation_problem.get_optimization_dlmps()
            time_duration = time.time() - time_start
            logger.info(f"Test OptimalOperationProblem: Completed in {time_duration:.6f} seconds.")
