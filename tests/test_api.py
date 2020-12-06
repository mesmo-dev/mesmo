"""Test API."""

import cvxpy as cp
import time
import unittest

import fledge.config
import fledge.api

logger = fledge.config.get_logger(__name__)

# Check availability of optimization solver.
optimization_solver_available = (
    fledge.config.config['optimization']['solver_name'].upper() in cp.installed_solvers()
)


class TestAPI(unittest.TestCase):

    def test_run_nominal_operation_problem(self):
        # Get result.
        time_start = time.time()
        fledge.api.run_nominal_operation_problem('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test run_nominal_operation_problem: Completed in {time_duration:.6f} seconds.")

    if optimization_solver_available:

        def test_run_optimal_operation_problem(self):
            # Get result.
            time_start = time.time()
            fledge.api.run_optimal_operation_problem('singapore_tanjongpagar')
            time_duration = time.time() - time_start
            logger.info(f"Test run_optimal_operation_problem: Completed in {time_duration:.6f} seconds.")
