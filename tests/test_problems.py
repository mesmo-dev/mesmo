"""Test problems."""

import time
import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestProblems(unittest.TestCase):

    def test_nominal_operation_problem(self):
        # Get result.
        time_start = time.time()
        fledge.problems.NominalOperationProblem('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test NominalOperationProblem: Completed in {time_duration:.6f} seconds.")

    def test_optimal_operation_problem(self):
        # Get result.
        time_start = time.time()
        fledge.problems.OptimalOperationProblem('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test OptimalOperationProblem: Completed in {time_duration:.6f} seconds.")
