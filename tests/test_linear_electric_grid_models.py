"""Test linear electric grid models."""

import numpy as np
import pandas as pd
import time
import unittest

import fledge.config
import fledge.electric_grid_models
import fledge.linear_electric_grid_models
import fledge.power_flow_solvers


logger = fledge.config.get_logger(__name__)


class TestLinearElectricGridModels(unittest.TestCase):

    def test_linear_electric_grid_model_1(self):
        # Get result.
        time_start = time.time()
        fledge.linear_electric_grid_models.LinearElectricGridModel(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test LinearElectricGridModel #1: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_linear_electric_grid_model_2(self):
        # Obtain electric grid model.
        electric_grid_model = fledge.electric_grid_models.ElectricGridModel(fledge.config.test_scenario_name)

        # Obtain power flow solution for nominal loading conditions.
        power_flow_solution = fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(electric_grid_model)

        # Obtain linear electric grid model for nominal loading conditions.
        linear_electric_grid_model = (
            fledge.linear_electric_grid_models.LinearElectricGridModel(
                electric_grid_model,
                power_flow_solution
            )
        )

        # Define power vector multipliers for testing of linear model at
        # different loading conditions.
        power_multipliers = np.arange(0.8, 1.25, 0.05)

        # Obtain nodal power vectors assuming nominal loading conditions.
        # TODO: Rename to initial.
        node_power_vector_wye = (
            np.transpose([
                electric_grid_model.load_incidence_wye_matrix
                @ electric_grid_model.load_power_vector_nominal
            ])
        )
        node_power_vector_delta = (
            np.transpose([
                electric_grid_model.load_incidence_delta_matrix
                @ electric_grid_model.load_power_vector_nominal
            ])
        )
        node_power_vector_wye_active = (
            np.real(node_power_vector_wye)
        )
        node_power_vector_wye_reactive = (
            np.imag(node_power_vector_wye)
        )
        node_power_vector_delta_active = (
            np.real(node_power_vector_delta)
        )
        node_power_vector_delta_reactive = (
            np.imag(node_power_vector_delta)
        )

        # Obtain initial and no-load nodal voltage vectors.
        node_voltage_vector_initial = power_flow_solution.node_voltage_vector
        branch_power_vector_1_initial = power_flow_solution.branch_power_vector_1
        branch_power_vector_2_initial = power_flow_solution.branch_power_vector_2
        loss_initial = power_flow_solution.loss

        node_voltage_vector_no_load = electric_grid_model.node_voltage_vector_no_load

        # Instantiate testing arrays.
        node_voltage_vector_magnitude_fixed_point = (
            np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.float)
        )
        node_voltage_vector_magnitude_linear_model = (
            np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.float)
        )
        node_voltage_vector_fixed_point = (
            np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.complex)
        )
        node_voltage_vector_linear_model = (
            np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.complex)
        )
        branch_power_vector_1_squared_fixed_point = (
            np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
        )
        branch_power_vector_1_squared_linear_model = (
            np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
        )
        branch_power_vector_2_squared_fixed_point = (
            np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
        )
        branch_power_vector_2_squared_linear_model = (
            np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
        )
        loss_active_fixed_point = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        loss_active_linear_model = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        loss_reactive_fixed_point = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        loss_reactive_linear_model = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        node_voltage_vector_error = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        node_voltage_vector_magnitude_error = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        branch_power_vector_1_squared_error = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        branch_power_vector_2_squared_error = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        loss_active_error = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )
        loss_reactive_error = (
            np.zeros(len(power_multipliers), dtype=np.float)
        )

        # Define error calculation utility function.
        def get_error(actual, approximate):
            return 100.0 * np.max(abs((approximate - actual) / actual))

        # Evaluate linear model errors for each power multiplier.
        for (multiplier_index, power_multiplier) in enumerate(power_multipliers):
            # Obtain nodal power vectors depending on multiplier.
            node_power_vector_wye_candidate = (
                power_multiplier
                * node_power_vector_wye
            )
            node_power_vector_delta_candidate = (
                power_multiplier
                * node_power_vector_delta
            )
            node_power_vector_wye_candidate_active = (
                np.real(node_power_vector_wye_candidate)
            )
            node_power_vector_wye_candidate_reactive = (
                np.imag(node_power_vector_wye_candidate)
            )
            node_power_vector_delta_candidate_active = (
                np.real(node_power_vector_delta_candidate)
            )
            node_power_vector_delta_candidate_reactive = (
                np.imag(node_power_vector_delta_candidate)
            )

            # Obtain fixed-point power flow solution.
            power_flow_solution_fixed_point = (
                fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
                    electric_grid_model,
                    node_power_vector_wye_candidate,
                    node_power_vector_delta_candidate
                )
            )
            node_voltage_vector_fixed_point[:, multiplier_index] = (
                power_flow_solution_fixed_point.node_voltage_vector.flatten()
            )
            node_voltage_vector_magnitude_fixed_point[:, multiplier_index] = (
                abs(power_flow_solution_fixed_point.node_voltage_vector).flatten()
            )
            branch_power_vector_1_squared_fixed_point[:, multiplier_index] = (
                (abs(power_flow_solution_fixed_point.branch_power_vector_1) ** 2).flatten()
            )
            branch_power_vector_2_squared_fixed_point[:, multiplier_index] = (
                (abs(power_flow_solution_fixed_point.branch_power_vector_2) ** 2).flatten()
            )
            loss_active_fixed_point[multiplier_index] = (
                np.real([power_flow_solution_fixed_point.loss]).flatten()
            )
            loss_reactive_fixed_point[multiplier_index] = (
                np.imag([power_flow_solution_fixed_point.loss]).flatten()
            )

            # Calculate nodal power difference / change.
            node_power_vector_wye_active_change = (
                node_power_vector_wye_candidate_active
                - node_power_vector_wye_active
            )
            node_power_vector_wye_reactive_change = (
                node_power_vector_wye_candidate_reactive
                - node_power_vector_wye_reactive
            )
            node_power_vector_delta_active_change = (
                node_power_vector_delta_candidate_active
                - node_power_vector_delta_active
            )
            node_power_vector_delta_reactive_change = (
                node_power_vector_delta_candidate_reactive
                - node_power_vector_delta_reactive
            )

            # Calculate approximate voltage, power vectors and total losses.
            node_voltage_vector_linear_model[:, multiplier_index] = (
                node_voltage_vector_initial
                + linear_electric_grid_model.sensitivity_voltage_by_power_wye_active
                @ node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_voltage_by_power_wye_reactive
                @ node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_voltage_by_power_delta_active
                @ node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_voltage_by_power_delta_reactive
                @ node_power_vector_delta_reactive_change
            ).flatten()
            node_voltage_vector_magnitude_linear_model[:, multiplier_index] = (
                abs(node_voltage_vector_initial)
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_active
                @ node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_reactive
                @ node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_active
                @ node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_reactive
                @ node_power_vector_delta_reactive_change
            ).flatten()
            branch_power_vector_1_squared_linear_model[:, multiplier_index] = (
                (abs(branch_power_vector_1_initial) ** 2)
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_wye_active
                @ node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_wye_reactive
                @ node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_delta_active
                @ node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_delta_reactive
                @ node_power_vector_delta_reactive_change
            ).flatten()
            branch_power_vector_2_squared_linear_model[:, multiplier_index] = (
                (abs(branch_power_vector_2_initial) ** 2)
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_wye_active
                @ node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_wye_reactive
                @ node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_delta_active
                @ node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_delta_reactive
                @ node_power_vector_delta_reactive_change
            ).flatten()
            loss_active_linear_model[multiplier_index] = (
                np.real([loss_initial])
                + linear_electric_grid_model.sensitivity_loss_active_by_power_wye_active
                @ node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_loss_active_by_power_wye_reactive
                @ node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_loss_active_by_power_delta_active
                @ node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_loss_active_by_power_delta_reactive
                @ node_power_vector_delta_reactive_change
            ).flatten()
            loss_reactive_linear_model[multiplier_index] = (
                np.imag([loss_initial])
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_wye_active
                @ node_power_vector_wye_active_change
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_wye_reactive
                @ node_power_vector_wye_reactive_change
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_delta_active
                @ node_power_vector_delta_active_change
                + linear_electric_grid_model.sensitivity_loss_reactive_by_power_delta_reactive
                @ node_power_vector_delta_reactive_change
            ).flatten()

            # Calculate errors for voltage, power vectors and total losses.
            node_voltage_vector_error[multiplier_index] = (
                get_error(
                    node_voltage_vector_fixed_point[:, multiplier_index],
                    node_voltage_vector_linear_model[:, multiplier_index]
                )
            )
            node_voltage_vector_magnitude_error[multiplier_index] = (
                get_error(
                    node_voltage_vector_magnitude_fixed_point[:, multiplier_index],
                    node_voltage_vector_magnitude_linear_model[:, multiplier_index]
                )
            )
            branch_power_vector_1_squared_error[multiplier_index] = (
                get_error(
                    branch_power_vector_1_squared_fixed_point[:, multiplier_index],
                    branch_power_vector_1_squared_linear_model[:, multiplier_index]
                )
            )
            branch_power_vector_2_squared_error[multiplier_index] = (
                get_error(
                    branch_power_vector_2_squared_fixed_point[:, multiplier_index],
                    branch_power_vector_2_squared_linear_model[:, multiplier_index]
                )
            )
            loss_active_error[multiplier_index] = (
                get_error(
                    loss_active_fixed_point[multiplier_index],
                    loss_active_linear_model[multiplier_index]
                )
            )
            loss_reactive_error[multiplier_index] = (
                get_error(
                    loss_reactive_fixed_point[multiplier_index],
                    loss_reactive_linear_model[multiplier_index]
                )
            )

        linear_electric_grid_model_error = (
            pd.DataFrame(
                [
                    node_voltage_vector_error,
                    node_voltage_vector_magnitude_error,
                    branch_power_vector_1_squared_error,
                    branch_power_vector_2_squared_error,
                    loss_active_error,
                    loss_reactive_error
                ],
                index=[
                    'node_voltage_vector_error',
                    'node_voltage_vector_magnitude_error',
                    'branch_power_vector_1_squared_error',
                    'branch_power_vector_2_squared_error',
                    'loss_active_error',
                    'loss_reactive_error'
                ],
                columns=pd.Index(power_multipliers, name='power_multipliers'),
            )
        )
        logger.info(f"linear_electric_grid_model_error = \n{round(linear_electric_grid_model_error, 2).to_string()}")


if __name__ == '__main__':
    unittest.main()
