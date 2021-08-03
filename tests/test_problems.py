"""Test problems."""

import gurobipy as gp
import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestProblems(unittest.TestCase):

    def test_nominal_operation_problem(self):
        # Get result.
        fledge.utils.log_time("test_nominal_operation_problem", log_level='info', logger_object=logger)
        problem = fledge.problems.NominalOperationProblem('singapore_tanjongpagar')
        problem.solve()
        problem.get_results()
        fledge.utils.log_time("test_nominal_operation_problem", log_level='info', logger_object=logger)

    def test_optimal_operation_problem(self):
        # Get result.
        fledge.utils.log_time("test_optimal_operation_problem", log_level='info', logger_object=logger)
        problem = fledge.problems.OptimalOperationProblem('singapore_tanjongpagar')
        try:
            problem.solve()
            problem.get_results()
        except gp.GurobiError:
            # Soft fail: Only raise warning on selected errors, since it may be due to solver not installed.
            logger.warning(f"Test test_optimal_operation_problem failed due to solver error.", exc_info=True)
        fledge.utils.log_time("test_optimal_operation_problem", log_level='info', logger_object=logger)
