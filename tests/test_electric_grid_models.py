"""Test electric grid models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import unittest

import fledge.config
import fledge.electric_grid_models

logger = fledge.config.get_logger(__name__)


class TestElectricGridModels(unittest.TestCase):

    def test_electric_grid_model_default(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridModelDefault(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridModelDefault: Completed in {time_duration:.6f} seconds.")

    def test_electric_grid_model_opendss(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridModelOpenDSS: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_fixed_point(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionFixedPoint: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_z_bus(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.PowerFlowSolutionZBus(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionZBus: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_opendss(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.PowerFlowSolutionOpenDSS(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionOpenDSS: Completed in {time_duration:.6f} seconds.")

    def test_linear_electric_grid_model_global(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.LinearElectricGridModelGlobal(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test LinearElectricGridModelGlobal: Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
