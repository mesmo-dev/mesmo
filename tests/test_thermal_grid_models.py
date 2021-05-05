"""Test script for thermal grid models module."""

import time
import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestThermalGridModels(unittest.TestCase):

    def test_thermal_grid_model(self):
        # Get result.
        time_start = time.time()
        fledge.thermal_grid_models.ThermalGridModel('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test ThermalGridModel: Completed in {time_duration:.6f} seconds.")

    def test_thermal_power_flow_solution(self):
        # Get result.
        time_start = time.time()
        fledge.thermal_grid_models.ThermalPowerFlowSolution('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test ThermalPowerFlowSolution: Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
