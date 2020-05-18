"""Example script for testing / validating the linear electric grid model."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models


def main():

    # Settings.
    scenario_name = fledge.config.config['tests']['scenario_name']
    results_path = (
        os.path.join(
            fledge.config.config['paths']['results'],
            f'run_linear_electric_grid_model_validation_{fledge.config.get_timestamp()}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid model.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain power flow solution for nominal power conditions.
    power_flow_solution_initial = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)

    # Obtain linear electric grid model for nominal power conditions.
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution_initial
        )
    )

    # Obtain no load voltage vector.
    node_voltage_vector_no_load = electric_grid_model.node_voltage_vector_no_load

    # Define power vector multipliers for testing of linear model at different power conditions.
    power_multipliers = np.arange(-0.2, 1.2, 0.1)

    # Instantiate testing arrays.
    node_voltage_vector_magnitude_power_flow = (
        np.zeros((len(electric_grid_model.nodes), len(power_multipliers)), dtype=np.float)
    )
    node_voltage_vector_magnitude_linear_model = (
        np.zeros((len(electric_grid_model.nodes), len(power_multipliers)), dtype=np.float)
    )
    node_voltage_vector_power_flow = (
        np.zeros((len(electric_grid_model.nodes), len(power_multipliers)), dtype=np.complex)
    )
    node_voltage_vector_linear_model = (
        np.zeros((len(electric_grid_model.nodes), len(power_multipliers)), dtype=np.complex)
    )
    branch_power_vector_1_squared_power_flow = (
        np.zeros((len(electric_grid_model.branches), len(power_multipliers)), dtype=np.float)
    )
    branch_power_vector_1_squared_linear_model = (
        np.zeros((len(electric_grid_model.branches), len(power_multipliers)), dtype=np.float)
    )
    branch_power_vector_2_squared_power_flow = (
        np.zeros((len(electric_grid_model.branches), len(power_multipliers)), dtype=np.float)
    )
    branch_power_vector_2_squared_linear_model = (
        np.zeros((len(electric_grid_model.branches), len(power_multipliers)), dtype=np.float)
    )
    loss_active_power_flow = (
        np.zeros(len(power_multipliers), dtype=np.float)
    )
    loss_active_linear_model = (
        np.zeros(len(power_multipliers), dtype=np.float)
    )
    loss_reactive_power_flow = (
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
        if np.max(abs(actual)) < 1E-3:
            return np.nan
        else:
            return 100.0 * np.max(abs((approximate - actual) / actual))

    # Evaluate linear model errors for each power multiplier.
    for (multiplier_index, power_multiplier) in enumerate(power_multipliers):

        # Obtain fixed-point power flow solution depending on multiplier.
        power_flow_solution = (
            fledge.electric_grid_models.PowerFlowSolutionFixedPoint(
                electric_grid_model,
                power_multiplier * electric_grid_model.der_power_vector_nominal
            )
        )
        node_voltage_vector_power_flow[:, multiplier_index] = (
            power_flow_solution.node_voltage_vector.ravel()
        )
        node_voltage_vector_magnitude_power_flow[:, multiplier_index] = (
            abs(power_flow_solution.node_voltage_vector).ravel()
        )
        branch_power_vector_1_squared_power_flow[:, multiplier_index] = (
            (abs(power_flow_solution.branch_power_vector_1) ** 2).ravel()
        )
        branch_power_vector_2_squared_power_flow[:, multiplier_index] = (
            (abs(power_flow_solution.branch_power_vector_2) ** 2).ravel()
        )
        loss_active_power_flow[multiplier_index] = (
            np.real([power_flow_solution.loss]).ravel()
        )
        loss_reactive_power_flow[multiplier_index] = (
            np.imag([power_flow_solution.loss]).ravel()
        )

        # Obtain DER power vector change.
        der_power_vector_active_change = (
            (power_multiplier - 1)
            * np.real(power_flow_solution_initial.der_power_vector)
        )
        der_power_vector_reactive_change = (
            (power_multiplier - 1)
            * np.imag(power_flow_solution_initial.der_power_vector)
        )

        # Calculate approximate voltage, power vectors and total losses.
        node_voltage_vector_linear_model[:, multiplier_index] = (
            power_flow_solution_initial.node_voltage_vector
            + linear_electric_grid_model.sensitivity_voltage_by_der_power_active
            @ np.transpose([der_power_vector_active_change])
            + linear_electric_grid_model.sensitivity_voltage_by_der_power_reactive
            @ np.transpose([der_power_vector_reactive_change])
        ).ravel()
        node_voltage_vector_magnitude_linear_model[:, multiplier_index] = (
            np.abs(power_flow_solution_initial.node_voltage_vector)
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
            @ np.transpose([der_power_vector_active_change])
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
            @ np.transpose([der_power_vector_reactive_change])
        ).ravel()
        branch_power_vector_1_squared_linear_model[:, multiplier_index] = (
            np.abs(power_flow_solution_initial.branch_power_vector_1 ** 2)
            + linear_electric_grid_model.sensitivity_branch_power_1_by_der_power_active
            @ np.transpose([der_power_vector_active_change])
            + linear_electric_grid_model.sensitivity_branch_power_1_by_der_power_reactive
            @ np.transpose([der_power_vector_reactive_change])
        ).ravel()
        branch_power_vector_2_squared_linear_model[:, multiplier_index] = (
            np.abs(power_flow_solution_initial.branch_power_vector_2 ** 2)
            + linear_electric_grid_model.sensitivity_branch_power_2_by_der_power_active
            @ np.transpose([der_power_vector_active_change])
            + linear_electric_grid_model.sensitivity_branch_power_2_by_der_power_reactive
            @ np.transpose([der_power_vector_reactive_change])
        ).ravel()
        loss_active_linear_model[multiplier_index] = (
            np.real(power_flow_solution_initial.loss)
            + linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
            @ np.transpose([der_power_vector_active_change])
            + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
            @ np.transpose([der_power_vector_reactive_change])
        ).ravel()
        loss_reactive_linear_model[multiplier_index] = (
            np.imag(power_flow_solution_initial.loss)
            + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
            @ np.transpose([der_power_vector_active_change])
            + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
            @ np.transpose([der_power_vector_reactive_change])
        ).ravel()

        # Calculate errors for voltage, power vectors and total losses.
        node_voltage_vector_error[multiplier_index] = (
            get_error(
                node_voltage_vector_power_flow[:, multiplier_index],
                node_voltage_vector_linear_model[:, multiplier_index]
            )
        )
        node_voltage_vector_magnitude_error[multiplier_index] = (
            get_error(
                node_voltage_vector_magnitude_power_flow[:, multiplier_index],
                node_voltage_vector_magnitude_linear_model[:, multiplier_index]
            )
        )
        branch_power_vector_1_squared_error[multiplier_index] = (
            get_error(
                branch_power_vector_1_squared_power_flow[:, multiplier_index],
                branch_power_vector_1_squared_linear_model[:, multiplier_index]
            )
        )
        branch_power_vector_2_squared_error[multiplier_index] = (
            get_error(
                branch_power_vector_2_squared_power_flow[:, multiplier_index],
                branch_power_vector_2_squared_linear_model[:, multiplier_index]
            )
        )
        loss_active_error[multiplier_index] = (
            get_error(
                loss_active_power_flow[multiplier_index],
                loss_active_linear_model[multiplier_index]
            )
        )
        loss_reactive_error[multiplier_index] = (
            get_error(
                loss_reactive_power_flow[multiplier_index],
                loss_reactive_linear_model[multiplier_index]
            )
        )

    # Construct results table.
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
            columns=pd.Index(np.round(power_multipliers, 2), name='power_multipliers'),
        )
    )
    linear_electric_grid_model_error = linear_electric_grid_model_error.round(2)

    # Print results.
    print(linear_electric_grid_model_error)

    # Store results as CSV.
    linear_electric_grid_model_error.to_csv(os.path.join(results_path, 'linear_electric_grid_model_error.csv'))

    # Plot results.

    # Voltage magnitude.
    for node_index, node in enumerate(electric_grid_model.nodes):
        plt.plot(power_multipliers, node_voltage_vector_magnitude_power_flow[node_index, :], label='Power flow')
        plt.plot(power_multipliers, node_voltage_vector_magnitude_linear_model[node_index, :], label='Linear model')
        plt.scatter([0.0], [abs(node_voltage_vector_no_load[node_index])], label='No load')
        plt.scatter([1.0], [abs(power_flow_solution_initial.node_voltage_vector[node_index])], label='Initial point')
        plt.legend()
        plt.title(f"Voltage magnitude node/phase: {node}")
        plt.savefig(os.path.join(results_path, f'voltage_magnitude_{node}.png'))
        plt.close()

    # Branch flow.
    for branch_index, branch in enumerate(electric_grid_model.branches):
        plt.plot(power_multipliers, branch_power_vector_1_squared_power_flow[branch_index, :], label='Power flow')
        plt.plot(power_multipliers, branch_power_vector_1_squared_linear_model[branch_index, :], label='Linear model')
        plt.scatter([0.0], [0.0], label='No load')
        plt.scatter([1.0], [abs(power_flow_solution_initial.branch_power_vector_1[branch_index] ** 2)], label='Initial point')
        plt.legend()
        plt.title(f"Branch flow 1 branch/phase/type: {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_1_{branch}.png'))
        plt.close()

        plt.plot(power_multipliers, branch_power_vector_2_squared_power_flow[branch_index, :], label='Power flow')
        plt.plot(power_multipliers, branch_power_vector_2_squared_linear_model[branch_index, :], label='Linear model')
        plt.scatter([0.0], [0.0], label='No load')
        plt.scatter([1.0], [abs(power_flow_solution_initial.branch_power_vector_2[branch_index] ** 2)], label='Initial point')
        plt.legend()
        plt.title(f"Branch flow 2 branch/phase/type: {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_2_{branch}.png'))
        plt.close()

    # Loss.
    plt.plot(power_multipliers, loss_active_power_flow, label='Power flow')
    plt.plot(power_multipliers, loss_active_linear_model, label='Linear model')
    plt.scatter([0.0], [0.0], label='No load')
    plt.scatter([1.0], [np.real([power_flow_solution_initial.loss])], label='Initial point')
    plt.legend()
    plt.title("Loss active")
    plt.savefig(os.path.join(results_path, f'loss_active.png'))
    plt.close()

    plt.plot(power_multipliers, loss_reactive_power_flow, label='Power flow')
    plt.plot(power_multipliers, loss_reactive_linear_model, label='Linear model')
    plt.scatter([0.0], [0.0], label='No load')
    plt.scatter([1.0], [np.imag([power_flow_solution_initial.loss])], label='Initial point')
    plt.legend()
    plt.title("Loss reactive")
    plt.savefig(os.path.join(results_path, f'loss_reactive.png'))
    plt.close()

    # Print results path.
    print("Results are stored in: " + results_path)


if __name__ == '__main__':
    main()
