"""Test problems."""

import gurobipy as gp
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestProblems(unittest.TestCase):

    def test_nominal_operation_problem(self):
        # Get result.
        mesmo.utils.log_time("test_nominal_operation_problem", log_level='info', logger_object=logger)
        problem = mesmo.problems.NominalOperationProblem('singapore_tanjongpagar')
        problem.solve()
        problem.get_results()
        mesmo.utils.log_time("test_nominal_operation_problem", log_level='info', logger_object=logger)

    def test_optimal_operation_problem(self):
        # Get result.
        mesmo.utils.log_time("test_optimal_operation_problem", log_level='info', logger_object=logger)
        problem = mesmo.problems.OptimalOperationProblem('singapore_tanjongpagar')
        problem.solve()
        problem.get_results()
        mesmo.utils.log_time("test_optimal_operation_problem", log_level='info', logger_object=logger)
