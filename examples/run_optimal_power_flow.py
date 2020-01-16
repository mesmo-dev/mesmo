"""Example script for setting up and solving an optimal power flow problem."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.linear_electric_grid_models
import fledge.electric_grid_models
import fledge.power_flow_solvers


def main():

    # Settings.
    scenario_name = "singapore_6node"

    # Get model.
    electric_grid_data = (
        fledge.database_interface.ElectricGridData(scenario_name)
    )
    electric_grid_index = (
        fledge.electric_grid_models.ElectricGridIndex(scenario_name)
    )
    electric_grid_model = (
        fledge.electric_grid_models.ElectricGridModel(scenario_name)
    )

    # Instantiate iteration variables.
    load_power_vector_reference = electric_grid_model.load_power_vector_nominal
    power_flow_solutions = []
    linear_electric_grid_models = []
    trust_region_iteration_count = 0

    # Get power flow solution and linear grid model.
    power_flow_solution = (
        fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
            electric_grid_model,
            load_power_vector_reference
        )
    )
    power_flow_solutions.append(power_flow_solution)
    linear_electric_grid_model = (
        fledge.linear_electric_grid_models.LinearElectricGridModel(
            electric_grid_model,
            power_flow_solution
        )
    )
    linear_electric_grid_models.append(linear_electric_grid_model)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)

    # Define variables.

    # Load.
    optimization_problem.load_active_power_vector_change = (
        pyo.Var(electric_grid_index.load_names.to_list())
    )
    optimization_problem.load_reactive_power_vector_change = (
        pyo.Var(electric_grid_index.load_names.to_list())
    )

    # Power.
    optimization_problem.node_power_vector_wye_active_change = (
        pyo.Var(electric_grid_index.nodes_phases.to_list())
    )
    optimization_problem.node_power_vector_wye_reactive_change = (
        pyo.Var(electric_grid_index.nodes_phases.to_list())
    )
    optimization_problem.node_power_vector_delta_active_change = (
        pyo.Var(electric_grid_index.nodes_phases.to_list())
    )
    optimization_problem.node_power_vector_delta_reactive_change = (
        pyo.Var(electric_grid_index.nodes_phases.to_list())
    )

    # Voltage.
    optimization_problem.voltage_magnitude_vector_change = (
        pyo.Var(electric_grid_index.nodes_phases.to_list())
    )

    # Branch flows.
    optimization_problem.branch_power_vector_1_squared_change = (
        pyo.Var(electric_grid_index.branches_phases.to_list())
    )
    optimization_problem.branch_power_vector_2_squared_change = (
        pyo.Var(electric_grid_index.branches_phases.to_list())
    )

    # Loss.
    optimization_problem.loss_active_change = pyo.Var()
    optimization_problem.loss_reactive_change = pyo.Var()

    # Trust-region.
    optimization_problem.change_limit = pyo.Var()

    # Define constraints.
    optimization_problem.constraints = pyo.ConstraintList()

    # Load.
    for load_index, load_name in enumerate(electric_grid_index.load_names):
        optimization_problem.constraints.add(
            optimization_problem.load_active_power_vector_change[load_name]
            >=
            -0.5 * np.real(electric_grid_model.load_power_vector_nominal[load_index])
        )
    for load_index, load_name in enumerate(electric_grid_index.load_names):
        optimization_problem.constraints.add(
            optimization_problem.load_active_power_vector_change[load_name]
            <=
            0.5 * np.real(electric_grid_model.load_power_vector_nominal[load_index])
        )
    for load_index, load_name in enumerate(electric_grid_index.load_names):
        optimization_problem.constraints.add(
            optimization_problem.load_reactive_power_vector_change[load_name]
            >=
            -0.5 * np.imag(electric_grid_model.load_power_vector_nominal[load_index])
        )
    for load_index, load_name in enumerate(electric_grid_index.load_names):
        optimization_problem.constraints.add(
            optimization_problem.load_reactive_power_vector_change[load_name]
            <=
            0.5 * np.imag(electric_grid_model.load_power_vector_nominal[load_index])
        )

    # Power.
    for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):
        optimization_problem.constraints.add(
            optimization_problem.node_power_vector_wye_active_change[node_phase]
            ==
            pyo.quicksum(
                electric_grid_model.load_incidence_wye_matrix[node_phase_index, load_index]
                * optimization_problem.load_active_power_vector_change[load_name]
                for load_index, load_name in enumerate(electric_grid_index.load_names)
            )
        )
    for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):
        optimization_problem.constraints.add(
            optimization_problem.node_power_vector_wye_reactive_change[node_phase]
            ==
            pyo.quicksum(
                electric_grid_model.load_incidence_wye_matrix[node_phase_index, load_index]
                * optimization_problem.load_reactive_power_vector_change[load_name]
                for load_index, load_name in enumerate(electric_grid_index.load_names)
            )
        )
    for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):
        optimization_problem.constraints.add(
            optimization_problem.node_power_vector_delta_active_change[node_phase]
            ==
            pyo.quicksum(
                electric_grid_model.load_incidence_delta_matrix[node_phase_index, load_index]
                * optimization_problem.load_active_power_vector_change[load_name]
                for load_index, load_name in enumerate(electric_grid_index.load_names)
            )
        )
    for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):
        optimization_problem.constraints.add(
            optimization_problem.node_power_vector_delta_reactive_change[node_phase]
            ==
            pyo.quicksum(
                electric_grid_model.load_incidence_delta_matrix[node_phase_index, load_index]
                * optimization_problem.load_reactive_power_vector_change[load_name]
                for load_index, load_name in enumerate(electric_grid_index.load_names)
            )
        )

    # Voltage.
    for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):
        optimization_problem.constraints.add(
            optimization_problem.voltage_magnitude_vector_change[node_phase]
            ==
            pyo.quicksum(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_active[node_phase_index, node_phase_index_other]
                * optimization_problem.node_power_vector_wye_active_change[node_phase_other]
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_reactive[node_phase_index, node_phase_index_other]
                * optimization_problem.node_power_vector_wye_reactive_change[node_phase_other]
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_active[node_phase_index, node_phase_index_other]
                * optimization_problem.node_power_vector_delta_active_change[node_phase_other]
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_reactive[node_phase_index, node_phase_index_other]
                * optimization_problem.node_power_vector_delta_reactive_change[node_phase_other]
                for node_phase_index_other, node_phase_other in enumerate(electric_grid_index.nodes_phases)
            )
        )

    # Branch flows.
    for branch_phase_index, branch_phase in enumerate(electric_grid_index.branches_phases):
        optimization_problem.constraints.add(
            optimization_problem.branch_power_vector_1_squared_change[branch_phase]
            ==
            pyo.quicksum(
                linear_electric_grid_model.sensitivity_branch_power_1_by_power_wye_active[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_wye_active_change[node_phase]
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_wye_reactive[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_wye_reactive_change[node_phase]
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_delta_active[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_delta_active_change[node_phase]
                + linear_electric_grid_model.sensitivity_branch_power_1_by_power_delta_reactive[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_delta_reactive_change[node_phase]
                for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases)
            )
        )
    for branch_phase_index, branch_phase in enumerate(electric_grid_index.branches_phases):
        optimization_problem.constraints.add(
            optimization_problem.branch_power_vector_2_squared_change[branch_phase]
            ==
            pyo.quicksum(
                linear_electric_grid_model.sensitivity_branch_power_2_by_power_wye_active[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_wye_active_change[node_phase]
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_wye_reactive[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_wye_reactive_change[node_phase]
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_delta_active[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_delta_active_change[node_phase]
                + linear_electric_grid_model.sensitivity_branch_power_2_by_power_delta_reactive[branch_phase_index, node_phase_index]
                * optimization_problem.node_power_vector_delta_reactive_change[node_phase]
                for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases)
            )
        )

    # Loss.
    optimization_problem.constraints.add(
        optimization_problem.loss_active_change
        ==
        pyo.quicksum(
            linear_electric_grid_model.sensitivity_loss_active_by_power_wye_active[0, node_phase_index]
            * optimization_problem.node_power_vector_wye_active_change[node_phase]
            + linear_electric_grid_model.sensitivity_loss_active_by_power_wye_reactive[0, node_phase_index]
            * optimization_problem.node_power_vector_wye_reactive_change[node_phase]
            + linear_electric_grid_model.sensitivity_loss_active_by_power_delta_active[0, node_phase_index]
            * optimization_problem.node_power_vector_delta_active_change[node_phase]
            + linear_electric_grid_model.sensitivity_loss_active_by_power_delta_reactive[0, node_phase_index]
            * optimization_problem.node_power_vector_delta_reactive_change[node_phase]
            for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases)
        )
    )
    optimization_problem.constraints.add(
        optimization_problem.loss_reactive_change
        ==
        pyo.quicksum(
            linear_electric_grid_model.sensitivity_loss_reactive_by_power_wye_active[0, node_phase_index]
            * optimization_problem.node_power_vector_wye_active_change[node_phase]
            + linear_electric_grid_model.sensitivity_loss_reactive_by_power_wye_reactive[0, node_phase_index]
            * optimization_problem.node_power_vector_wye_reactive_change[node_phase]
            + linear_electric_grid_model.sensitivity_loss_reactive_by_power_delta_active[0, node_phase_index]
            * optimization_problem.node_power_vector_delta_active_change[node_phase]
            + linear_electric_grid_model.sensitivity_loss_reactive_by_power_delta_reactive[0, node_phase_index]
            * optimization_problem.node_power_vector_delta_reactive_change[node_phase]
            for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases)
        )
    )

    # Trust region.
    for node_phase in electric_grid_index.nodes_phases:
        optimization_problem.constraints.add(
            optimization_problem.voltage_magnitude_vector_change[node_phase]
            >=
            -1.0 * optimization_problem.change_limit
        )
    for node_phase in electric_grid_index.nodes_phases:
        optimization_problem.constraints.add(
            optimization_problem.voltage_magnitude_vector_change[node_phase]
            <=
            optimization_problem.change_limit
        )
    for load_name in electric_grid_index.load_names:
        optimization_problem.constraints.add(
            optimization_problem.load_active_power_vector_change[load_name]
            >=
            -1.0 * optimization_problem.change_limit
        )
    for load_name in electric_grid_index.load_names:
        optimization_problem.constraints.add(
            optimization_problem.load_active_power_vector_change[load_name]
            <=
            optimization_problem.change_limit
        )
    for load_name in electric_grid_index.load_names:
        optimization_problem.constraints.add(
            optimization_problem.load_reactive_power_vector_change[load_name]
            >=
            -1.0 * optimization_problem.change_limit
        )
    for load_name in electric_grid_index.load_names:
        optimization_problem.constraints.add(
            optimization_problem.load_reactive_power_vector_change[load_name]
            <=
            optimization_problem.change_limit
        )

    # Define objective.
    cost = 0.0
    cost += (
        pyo.quicksum(
            optimization_problem.load_active_power_vector_change[load_name]
            for load_name in electric_grid_index.load_names
        )
    )
    cost += (
        pyo.quicksum(
            optimization_problem.load_active_power_vector_change[load_name]
            for load_name in electric_grid_index.load_names
        )
    )
    optimization_problem.objective = (
        pyo.Objective(
            expr=cost,
            sense=pyo.minimize
        )
    )

    # Solve optimization problem.
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    optimization_problem.display()

    # Instantiate results variables.

    # Load.
    load_active_power_vector_change = (
        pd.DataFrame(columns=electric_grid_index.load_names, index=[0], dtype=np.float)
    )
    load_reactive_power_vector_change = (
        pd.DataFrame(columns=electric_grid_index.load_names, index=[0], dtype=np.float)
    )

    # Power.
    node_power_vector_wye_active_change = (
        pd.DataFrame(columns=electric_grid_index.nodes_phases, index=[0], dtype=np.float)
    )
    node_power_vector_wye_reactive_change = (
        pd.DataFrame(columns=electric_grid_index.nodes_phases, index=[0], dtype=np.float)
    )
    node_power_vector_delta_active_change = (
        pd.DataFrame(columns=electric_grid_index.nodes_phases, index=[0], dtype=np.float)
    )
    node_power_vector_delta_reactive_change = (
        pd.DataFrame(columns=electric_grid_index.nodes_phases, index=[0], dtype=np.float)
    )

    # Voltage.
    voltage_magnitude_vector_change = (
        pd.DataFrame(columns=electric_grid_index.nodes_phases, index=[0], dtype=np.float)
    )

    # Branch flows.
    branch_power_vector_1_squared_change = (
        pd.DataFrame(columns=electric_grid_index.branches_phases, index=[0], dtype=np.float)
    )
    branch_power_vector_2_squared_change = (
        pd.DataFrame(columns=electric_grid_index.branches_phases, index=[0], dtype=np.float)
    )

    # Loss.
    loss_active_change = pd.DataFrame(columns=['total'], index=[0], dtype=np.float)
    loss_reactive_change = pd.DataFrame(columns=['total'], index=[0], dtype=np.float)

    # Obtain results.

    # Load.
    for load_name in electric_grid_index.load_names:
        load_active_power_vector_change[load_name] = (
            optimization_problem.load_active_power_vector_change[load_name].value
        )
        load_reactive_power_vector_change[load_name] = (
            optimization_problem.load_reactive_power_vector_change[load_name].value
        )

    for node_phase in electric_grid_index.nodes_phases:
        # Power.
        node_power_vector_wye_active_change[node_phase] = (
            optimization_problem.node_power_vector_wye_active_change[node_phase].value
        )
        node_power_vector_wye_reactive_change[node_phase] = (
            optimization_problem.node_power_vector_wye_reactive_change[node_phase].value
        )
        node_power_vector_delta_active_change[node_phase] = (
            optimization_problem.node_power_vector_delta_active_change[node_phase].value
        )
        node_power_vector_delta_reactive_change[node_phase] = (
            optimization_problem.node_power_vector_delta_reactive_change[node_phase].value
        )

        # Voltage.
        voltage_magnitude_vector_change[node_phase] = (
            optimization_problem.voltage_magnitude_vector_change[node_phase].value
        )

    # Branch flows.
    for branch_phase in electric_grid_index.branches_phases:
        branch_power_vector_1_squared_change[branch_phase] = (
            optimization_problem.branch_power_vector_1_squared_change[branch_phase].value
        )
        branch_power_vector_2_squared_change[branch_phase] = (
            optimization_problem.branch_power_vector_2_squared_change[branch_phase].value
        )

    # Loss.
    loss_active_change['total'] = optimization_problem.loss_active_change.value
    loss_reactive_change['total'] = optimization_problem.loss_reactive_change.value

    # Post-processing results.
    voltage_magnitude_vector_per_unit = (
        1.0
        + voltage_magnitude_vector_change
        / abs(electric_grid_model.node_voltage_vector_no_load.transpose())
    )
    voltage_magnitude_vector_per_unit['mean'] = voltage_magnitude_vector_per_unit.mean(axis=1)
    load_active_power_vector_per_unit = (
        1.0
        + load_active_power_vector_change
        / np.real(electric_grid_model.load_power_vector_nominal.transpose())
    )
    load_active_power_vector_per_unit['mean'] = load_active_power_vector_per_unit.mean(axis=1)
    branch_power_vector_1_squared_per_unit = (
        1.0
        + branch_power_vector_1_squared_change
        / abs(power_flow_solution.branch_power_vector_1.transpose() ** 2)
    )
    branch_power_vector_1_squared_per_unit['mean'] = branch_power_vector_1_squared_per_unit.mean(axis=1)
    loss_active_per_unit = (
        1.0
        + loss_active_change
        / np.real(power_flow_solution.loss)
    )

    # Print some results.
    print(f"voltage_magnitude_vector_per_unit = \n{voltage_magnitude_vector_per_unit.to_string()}")
    print(f"load_active_power_vector_per_unit = \n{load_active_power_vector_per_unit.to_string()}")
    print(f"branch_power_vector_1_squared_per_unit = \n{branch_power_vector_1_squared_per_unit.to_string()}")
    print(f"loss_active_per_unit = \n{loss_active_per_unit.to_string()}")


if __name__ == '__main__':
    main()
