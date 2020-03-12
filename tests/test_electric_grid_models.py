"""Test electric grid models."""

import time
import unittest

import fledge.config
import fledge.electric_grid_models

logger = fledge.config.get_logger(__name__)


class TestElectricGridModels(unittest.TestCase):

    def test_electric_grid_model_default(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridModelDefault(fledge.config.test_scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridModelDefault: Completed in {time_duration:.6f} seconds.")

    def test_electric_grid_model_opendss(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.test_scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridModelOpenDSS: Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
