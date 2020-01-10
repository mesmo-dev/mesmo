"""Test electric grid models."""

import time
import unittest

import fledge.config
import fledge.electric_grid_models

logger = fledge.config.get_logger(__name__)


class TestElectricGridModels(unittest.TestCase):

    def test_electric_grid_index(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridIndex(
            scenario_name=fledge.config.test_scenario_name
        )
        time_end = time.time()
        logger.info("Test ElectricGridIndex: Completed in {} seconds.".format(round(time_end - time_start, 6)))

    def test_electric_grid_model(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridModel(
            scenario_name=fledge.config.test_scenario_name
        )
        time_end = time.time()
        logger.info("Test ElectricGridModel: Completed in {} seconds.".format(round(time_end - time_start, 6)))


if __name__ == '__main__':
    unittest.main()