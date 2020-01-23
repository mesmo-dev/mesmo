"""Test linear electric grid models."""

import numpy as np
import pandas as pd
import time
import unittest

import fledge.config
import fledge.linear_electric_grid_models
import fledge.electric_grid_models
import fledge.power_flow_solvers


logger = fledge.config.get_logger(__name__)


class TestLinearElectricGridModels(unittest.TestCase):

    def test_linear_electric_grid_model_global(self):
        # Get result.
        time_start = time.time()
        fledge.linear_electric_grid_models.LinearElectricGridModelGlobal(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test LinearElectricGridModelGlobal: Completed in {round(time_end - time_start, 6)} seconds.")


if __name__ == '__main__':
    unittest.main()
