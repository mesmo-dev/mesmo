"""Test API."""

import pyomo.environ as pyo
import time
import unittest

import fledge.config
import fledge.database_interface
import fledge.api

logger = fledge.config.get_logger(__name__)

# Check availability of optimization solver.
try:
    optimization_solver_available = pyo.SolverFactory(fledge.config.config['optimization']['solver_name']).available()
except Exception:
    optimization_solver_available = False

if optimization_solver_available:

    class TestAPI(unittest.TestCase):

        def test_run_operation_problem(self):
            # Get result.
            time_start = time.time()
            fledge.api.run_operation_problem('singapore_tanjongpagar')
            time_duration = time.time() - time_start
            logger.info(f"Test run_operation_problem: Completed in {time_duration:.6f} seconds.")
