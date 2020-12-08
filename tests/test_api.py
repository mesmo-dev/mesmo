"""Test API."""

import cvxpy as cp
import time
import unittest

import fledge.config
import fledge.api

logger = fledge.config.get_logger(__name__)


class TestAPI(unittest.TestCase):

    def test_run_nominal_operation_problem(self):
        # Get result.
        time_start = time.time()
        fledge.api.run_nominal_operation_problem('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test run_nominal_operation_problem: Completed in {time_duration:.6f} seconds.")

    def test_run_optimal_operation_problem(self):
        # Get result.
        time_start = time.time()
        try:
            fledge.api.run_optimal_operation_problem('singapore_tanjongpagar')
        except cp.SolverError:
            # Soft fail: Only raise warning on SolverError, since it may be due to solver not installed.
            logger.warning(f"Test run_optimal_operation_problem failed due to solver error.", exc_info=True)
        time_duration = time.time() - time_start
        logger.info(f"Test run_optimal_operation_problem: Completed in {time_duration:.6f} seconds.")
