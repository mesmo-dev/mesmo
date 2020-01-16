"""Example script for testing the linear electric grid models."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import fledge.config
import fledge.linear_electric_grid_models
import fledge.electric_grid_models
import fledge.power_flow_solvers


# Settings.
scenario_name = 'singapore_6node'
plots = True  # If True, script may produce plots.

results_path = os.path.join(fledge.config.results_path, f'run_linear_electric_grid_model_{fledge.config.timestamp}')
os.mkdir(results_path) if plots else None  # Instantiate results directory.

# Obtain electric grid model.
electric_grid_model = fledge.electric_grid_models.ElectricGridModel(scenario_name)

# Obtain power flow solution for nominal loading conditions.
power_flow_solution_initial = fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(electric_grid_model)

# Obtain linear electric grid model for nominal loading conditions.
linear_electric_grid_model = (
    fledge.linear_electric_grid_models.LinearElectricGridModel(
        electric_grid_model,
        power_flow_solution_initial
    )
)

# Obtain nodal power vectors assuming nominal loading conditions.
node_power_vector_wye_initial = (
    np.transpose([
        electric_grid_model.load_incidence_wye_matrix
        @ electric_grid_model.load_power_vector_nominal
    ])
)
node_power_vector_delta_initial = (
    np.transpose([
        electric_grid_model.load_incidence_delta_matrix
        @ electric_grid_model.load_power_vector_nominal
    ])
)
node_power_vector_wye_active_initial = (
    np.real(node_power_vector_wye_initial)
)
node_power_vector_wye_reactive_initial = (
    np.imag(node_power_vector_wye_initial)
)
node_power_vector_delta_active_initial = (
    np.real(node_power_vector_delta_initial)
)
node_power_vector_delta_reactive_initial = (
    np.imag(node_power_vector_delta_initial)
)

# Obtain initial and no-load nodal voltage vectors.
node_voltage_vector_initial = power_flow_solution_initial.node_voltage_vector
branch_power_vector_1_initial = power_flow_solution_initial.branch_power_vector_1
branch_power_vector_2_initial = power_flow_solution_initial.branch_power_vector_2
loss_initial = power_flow_solution_initial.loss

# Define power vector multipliers for testing of linear model at different loading conditions.
power_multipliers = np.arange(0.1, 1.5, 0.1)

# Instantiate testing arrays.
node_voltage_vector_magnitude_power_flow = (
    np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.float)
)
node_voltage_vector_magnitude_linear_model = (
    np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.float)
)
node_voltage_vector_power_flow = (
    np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.complex)
)
node_voltage_vector_linear_model = (
    np.zeros((electric_grid_model.index.node_dimension, len(power_multipliers)), dtype=np.complex)
)
branch_power_vector_1_squared_power_flow = (
    np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
)
branch_power_vector_1_squared_linear_model = (
    np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
)
branch_power_vector_2_squared_power_flow = (
    np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
)
branch_power_vector_2_squared_linear_model = (
    np.zeros((electric_grid_model.index.branch_dimension, len(power_multipliers)), dtype=np.float)
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
    return 100.0 * np.max(abs((approximate - actual) / actual))

# Evaluate linear model errors for each power multiplier.
for (multiplier_index, power_multiplier) in enumerate(power_multipliers):
    # Obtain nodal power vectors depending on multiplier.
    node_power_vector_wye_candidate = (
        power_multiplier
        * node_power_vector_wye_initial
    )
    node_power_vector_delta_candidate = (
        power_multiplier
        * node_power_vector_delta_initial
    )
    node_power_vector_wye_active_candidate = (
        np.real(node_power_vector_wye_candidate)
    )
    node_power_vector_wye_reactive_candidate = (
        np.imag(node_power_vector_wye_candidate)
    )
    node_power_vector_delta_active_candidate = (
        np.real(node_power_vector_delta_candidate)
    )
    node_power_vector_delta_reactive_candidate = (
        np.imag(node_power_vector_delta_candidate)
    )

    # Obtain fixed-point power flow solution.
    power_flow_solution = (
        fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
            electric_grid_model,
            node_power_vector_wye_candidate,
            node_power_vector_delta_candidate
        )
    )
    node_voltage_vector_power_flow[:, multiplier_index] = (
        power_flow_solution.node_voltage_vector.flatten()
    )
    node_voltage_vector_magnitude_power_flow[:, multiplier_index] = (
        abs(power_flow_solution.node_voltage_vector).flatten()
    )
    branch_power_vector_1_squared_power_flow[:, multiplier_index] = (
        (abs(power_flow_solution.branch_power_vector_1) ** 2).flatten()
    )
    branch_power_vector_2_squared_power_flow[:, multiplier_index] = (
        (abs(power_flow_solution.branch_power_vector_2) ** 2).flatten()
    )
    loss_active_power_flow[multiplier_index] = (
        np.real([power_flow_solution.loss]).flatten()
    )
    loss_reactive_power_flow[multiplier_index] = (
        np.imag([power_flow_solution.loss]).flatten()
    )

    # Calculate nodal power difference / change.
    node_power_vector_wye_active_change = (
        node_power_vector_wye_active_candidate
        - node_power_vector_wye_active_initial
    )
    node_power_vector_wye_reactive_change = (
        node_power_vector_wye_reactive_candidate
        - node_power_vector_wye_reactive_initial
    )
    node_power_vector_delta_active_change = (
        node_power_vector_delta_active_candidate
        - node_power_vector_delta_active_initial
    )
    node_power_vector_delta_reactive_change = (
        node_power_vector_delta_reactive_candidate
        - node_power_vector_delta_reactive_initial
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
print(round(linear_electric_grid_model_error, 2).to_string())

# Plot results.
if plots:

    # Voltage magnitude.
    for node_phase_index, node_phase in enumerate(electric_grid_model.index.nodes_phases):
        plt.plot(power_multipliers, node_voltage_vector_magnitude_power_flow[node_phase_index, :], label='Power flow')
        plt.plot(power_multipliers, node_voltage_vector_magnitude_linear_model[node_phase_index, :], label='Linear model')
        plt.scatter([1.0], [abs(node_voltage_vector_initial[node_phase_index])], label='Initial point')
        plt.legend()
        plt.title(f"Voltage magnitude node/phase: {node_phase}")
        plt.savefig(os.path.join(results_path, f'voltage_magnitude_{node_phase}.png'))
        plt.close()

    # Branch flow.
    for branch_phase_index, branch_phase in enumerate(electric_grid_model.index.branches_phases):
        plt.plot(power_multipliers, branch_power_vector_1_squared_power_flow[branch_phase_index, :], label='Power flow')
        plt.plot(power_multipliers, branch_power_vector_1_squared_linear_model[branch_phase_index, :], label='Linear model')
        plt.scatter([1.0], [abs(branch_power_vector_1_initial[branch_phase_index] ** 2)], label='Initial point')
        plt.legend()
        plt.title(f"Branch flow 1 branch/phase/type: {branch_phase}")
        plt.savefig(os.path.join(results_path, f'branch_power_1_{branch_phase}.png'))
        plt.close()

        plt.plot(power_multipliers, branch_power_vector_2_squared_power_flow[branch_phase_index, :], label='Power flow')
        plt.plot(power_multipliers, branch_power_vector_2_squared_linear_model[branch_phase_index, :], label='Linear model')
        plt.scatter([1.0], [abs(branch_power_vector_2_initial[branch_phase_index] ** 2)], label='Initial point')
        plt.legend()
        plt.title(f"Branch flow 2 branch/phase/type: {branch_phase}")
        plt.savefig(os.path.join(results_path, f'branch_power_2_{branch_phase}.png'))
        plt.close()

    # Loss.
    plt.plot(power_multipliers, loss_active_power_flow, label='Power flow')
    plt.plot(power_multipliers, loss_active_linear_model, label='Linear model')
    plt.scatter([1.0], [np.real([loss_initial])], label='Initial point')
    plt.legend()
    plt.title("Loss active")
    plt.savefig(os.path.join(results_path, f'loss_active.png'))
    plt.close()

    plt.plot(power_multipliers, loss_reactive_power_flow, label='Power flow')
    plt.plot(power_multipliers, loss_reactive_linear_model, label='Linear model')
    plt.scatter([1.0], [np.imag([loss_initial])], label='Initial point')
    plt.legend()
    plt.title("Loss reactive")
    plt.savefig(os.path.join(results_path, f'loss_reactive.png'))
    plt.close()

    # Store CSV file.
    linear_electric_grid_model_error.to_csv(os.path.join(results_path, 'linear_electric_grid_model_error.csv'))
