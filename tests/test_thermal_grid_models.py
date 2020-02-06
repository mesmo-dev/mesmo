"""Test script for thermal grid models module."""

import time
import unittest

import fledge.config
import fledge.thermal_grid_models

logger = fledge.config.get_logger(__name__)


class TestThermalGridModels(unittest.TestCase):

    def test_thermal_grid_model(self):
        # Get result.
        time_start = time.time()
        fledge.thermal_grid_models.ThermalGridModel('singapore_tanjongpagar')
        time_end = time.time()
        logger.info(f"Test ThermalGridModel: Completed in {round(time_end - time_start, 6)} seconds.")


if __name__ == '__main__':
    unittest.main()
