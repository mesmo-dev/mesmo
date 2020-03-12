"""Test power flow solvers."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from parameterized import parameterized
import scipy.sparse
import time
import unittest

import fledge.config
import fledge.electric_grid_models
import fledge.power_flow_solvers

logger = fledge.config.get_logger(__name__)

as_complex = np.vectorize(np.complex)  # Utility function to convert strings in numpy array to complex numbers.


class TestPowerFlowSolvers(unittest.TestCase):

    def test_get_voltage_opendss(self):
        # Initialize OpenDSS model.
        electric_grid_model_opendss = (
            fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.test_scenario_name)
        )

        # Get result.
        time_start = time.time()
        fledge.power_flow_solvers.get_voltage_opendss()
        time_duration = time.time() - time_start
        logger.info(f"Test get_voltage_opendss: Completed in {time_duration:.6f} seconds.")

    def test_get_branch_power_opendss(self):
        # Initialize OpenDSS model.
        electric_grid_model_opendss = (
            fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.test_scenario_name)
        )

        # Get result.
        time_start = time.time()
        fledge.power_flow_solvers.get_branch_power_opendss()
        time_duration = time.time() - time_start
        logger.info(f"Test get_branch_power_opendss: Completed in {time_duration:.6f} seconds.")

    def test_get_loss_opendss(self):
        # Initialize OpenDSS model.
        electric_grid_model_opendss = (
            fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.test_scenario_name)
        )

        # Get result.
        time_start = time.time()
        fledge.power_flow_solvers.get_loss_opendss()
        time_duration = time.time() - time_start
        logger.info(f"Test get_loss_opendss: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_fixed_point_1(self):
        # Get result.
        time_start = time.time()
        fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(fledge.config.test_scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionFixedPoint #1: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_fixed_point_2(self):
        # Obtain test data.
        electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(fledge.config.test_scenario_name)
        node_voltage_vector_no_load = abs(electric_grid_model.node_voltage_vector_no_load)

        # Define expected result.
        electric_grid_model_opendss = (
            fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.test_scenario_name)
        )
        node_voltage_vector_opendss = abs(fledge.power_flow_solvers.get_voltage_opendss())

        # Get result.
        time_start = time.time()
        node_voltage_vector_fixed_point = abs(
            fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(fledge.config.test_scenario_name).node_voltage_vector
        )
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionFixedPoint #2: Completed in {time_duration:.6f} seconds.")

        # Display results.
        if fledge.config.test_plots:
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
                (node_voltage_vector_fixed_point - node_voltage_vector_opendss) / node_voltage_vector_no_load,
                index=electric_grid_model.nodes,
                columns=['Absolute error']
            )
            absolute_error.plot(kind='bar')
            plt.show(block=False)

        # Compare expected and actual.
        # TODO: Enable result check.
        # np.testing.assert_array_almost_equal(node_voltage_vector_opendss, node_voltage_vector_fixed_point, decimal=0)


if __name__ == '__main__':
    unittest.main()
