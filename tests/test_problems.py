"""Test problems."""

import unittest

import fledge

logger = fledge.config.get_logger(__name__)

found_optimization_solver = False
if fledge.config.config['optimization']['solver_name'] == 'gurobi':
    import gurobipy
    try:
        gurobipy.Model()
        found_optimization_solver = True
    except gurobipy.GurobiError:
        pass


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
        if found_optimization_solver:
            problem.solve()
            problem.get_results()
        fledge.utils.log_time("test_optimal_operation_problem", log_level='info', logger_object=logger)
