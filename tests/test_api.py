"""Test API."""

import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestAPI(unittest.TestCase):
    def test_run_nominal_operation_problem(self):
        # Get result.
        mesmo.utils.log_time("test_run_nominal_operation_problem", log_level="info", logger_object=logger)
        mesmo.api.run_nominal_operation_problem("singapore_tanjongpagar")
        mesmo.utils.log_time("test_run_nominal_operation_problem", log_level="info", logger_object=logger)

    def test_run_optimal_operation_problem(self):
        # Get result.
        mesmo.utils.log_time("test_run_optimal_operation_problem", log_level="info", logger_object=logger)
        mesmo.api.run_optimal_operation_problem("singapore_tanjongpagar")
        mesmo.utils.log_time("test_run_optimal_operation_problem", log_level="info", logger_object=logger)
