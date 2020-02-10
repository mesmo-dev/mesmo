"""Test electric grid models."""

import time
import unittest

import fledge.config
import fledge.electric_grid_models

logger = fledge.config.get_logger(__name__)


class TestElectricGridModels(unittest.TestCase):

    def test_electric_grid_model(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridModel(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test ElectricGridModel: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_initialize_opendss_model(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.initialize_opendss_model(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test initialize_opendss_model: Completed in {round(time_end - time_start, 6)} seconds.")


if __name__ == '__main__':
    unittest.main()
