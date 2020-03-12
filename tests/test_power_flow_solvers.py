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

    def test_power_flow_solution_fixed_point(self):
        # Get result.
        time_start = time.time()
        fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(fledge.config.test_scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionFixedPoint: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_opendss(self):
        # Get result.
        time_start = time.time()
        fledge.power_flow_solvers.PowerFlowSolutionOpenDSS(fledge.config.test_scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test PowerFlowSolutionOpenDSS: Completed in {time_duration:.6f} seconds.")

    def test_power_flow_solution_fixed_point_vs_opendss(self):
        # Setup.
        electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(fledge.config.test_scenario_name)
        node_voltage_vector_no_load = abs(electric_grid_model.node_voltage_vector_no_load)

        # Get result.
        time_start = time.time()
        node_voltage_vector_fixed_point = abs(
            fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(fledge.config.test_scenario_name).node_voltage_vector
        )
        node_voltage_vector_opendss = abs(
            fledge.power_flow_solvers.PowerFlowSolutionOpenDSS(fledge.config.test_scenario_name).node_voltage_vector
        )
        time_duration = time.time() - time_start
        logger.info(
            f"Test PowerFlowSolutionFixedPoint vs. PowerFlowSolutionOpenDSS:"
            f" Completed in {time_duration:.6f} seconds."
        )

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
