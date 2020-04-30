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
        fledge.electric_grid_models.ElectricGridModelDefault(fledge.config.config['testing']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridModelDefault: Completed in {time_duration:.6f} seconds.")

    def test_electric_grid_model_opendss(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.config['testing']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridModelOpenDSS: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_fixed_point(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(fledge.config.config['testing']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionFixedPoint: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_opendss(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.PowerFlowSolutionOpenDSS(fledge.config.config['testing']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionOpenDSS: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_fixed_point_vs_opendss(self):
        # Setup.
        electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(fledge.config.config['testing']['scenario_name'])
        node_voltage_vector_no_load = abs(electric_grid_model.node_voltage_vector_no_load)

        # Get result.
        time_start = time.time()
        node_voltage_vector_fixed_point = abs(
            fledge.electric_grid_models.PowerFlowSolutionFixedPoint(fledge.config.config['testing']['scenario_name']).node_voltage_vector
        )
        node_voltage_vector_opendss = abs(
            fledge.electric_grid_models.PowerFlowSolutionOpenDSS(fledge.config.config['testing']['scenario_name']).node_voltage_vector
        )
        time_duration = time.time() - time_start
        logger.info(
            f"Test PowerFlowSolutionFixedPoint vs. PowerFlowSolutionOpenDSS:"
            f" Completed in {time_duration:.6f} seconds."
        )

        # Display results.
        if fledge.config.config['testing']['show_plots']:
            comparison = pd.DataFrame(
                np.hstack([
                    node_voltage_vector_opendss / node_voltage_vector_no_load,
                    node_voltage_vector_fixed_point / node_voltage_vector_no_load]),
                index=electric_grid_model.nodes,
                columns=['OpenDSS', 'Fixed Point']
            )
            comparison.plot(kind='bar')
            plt.show(block=False)

            absolute_error = pd.DataFrame(
                (node_voltage_vector_fixed_point - node_voltage_vector_opendss),
                index=electric_grid_model.nodes,
                columns=['Absolute error']
            )
            absolute_error.plot(kind='bar')
            plt.show(block=False)

        # Compare expected and actual.
        # TODO: Enable result check.
        # np.testing.assert_array_almost_equal(node_voltage_vector_opendss, node_voltage_vector_fixed_point, decimal=0)

    def test_linear_electric_grid_model_global(self):
        # Get result.
        time_start = time.time()
        fledge.electric_grid_models.LinearElectricGridModelGlobal(fledge.config.config['testing']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test LinearElectricGridModelGlobal: Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
