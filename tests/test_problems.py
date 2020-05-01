"""Test problems."""

import pyomo.environ as pyo
import time
import unittest

import fledge.config
import fledge.database_interface
import fledge.problems

logger = fledge.config.get_logger(__name__)


class TestProblems(unittest.TestCase):

    def test_optimal_operation_problem(self):
        # Get result.
        time_start = time.time()
        fledge.problems.OptimalOperationProblem('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test OptimalOperationProblem: Completed in {time_duration:.6f} seconds.")
