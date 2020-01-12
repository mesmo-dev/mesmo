"""Test linear electric grid models."""

import time
import unittest

import fledge.config
import fledge.linear_electric_grid_models


logger = fledge.config.get_logger(__name__)


class TestLinearElectricGridModels(unittest.TestCase):

    def test_linear_electric_grid_model(self):
        # Get result.
        time_start = time.time()
        fledge.linear_electric_grid_models.LinearElectricGridModel(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test LinearElectricGridModel: Completed in {round(time_end - time_start, 6)} seconds.")


if __name__ == '__main__':
    unittest.main()
