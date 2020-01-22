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

    # Define initial load vector and power flow solution.
    load_power_vector_reference_candidate = electric_grid_model.load_power_vector_nominal
    power_flow_solution_candidate = (
        fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
            electric_grid_model,
            load_power_vector_reference_candidate
        )
    )

    # Instantiate iteration variables.
    sigma = 0.0
    load_power_vector_change_max = np.inf
    trust_region_iteration_count = 0
    power_flow_solutions = []
    linear_electric_grid_models = []
    objective_power_flows = []
    optimization_problems = []

    # Define trust-region parameters.
    delta = 0.1  # 3.0 / 0.5 / range: (0, delta_max] / If too big, no power flow solution.
    delta_max = 0.2  # 4.0 / 1.0
    gamma = 0.5  # 0.5 / range: (0, 1)
    eta = 0.1  # 0.1 / range: (0, 0.5]
    tau = 0.1  # 0.1 / range: [0, 0.25)
    epsilon = 1.0e-3  # 1e-3 / 1e-4
    trust_region_iteration_limit = 100

    while (
            (load_power_vector_change_max > epsilon)
            and (trust_region_iteration_count < trust_region_iteration_limit)
    ):

        # Print progress.
        print(f"Starting trust-region iteration #{trust_region_iteration_count}")

        # Check trust-region solution acceptance conditions.
        if (trust_region_iteration_count == 0) or (sigma > tau):

            # Accept load vector and power flow solution candidate.
            load_power_vector_reference = load_power_vector_reference_candidate
            power_flow_solution = power_flow_solution_candidate
            power_flow_solutions.append(power_flow_solution)

            # Obtain new
            linear_electric_grid_model = (
                fledge.linear_electric_grid_models.LinearElectricGridModel(
                    electric_grid_model,
                    power_flow_solution
                )
            )
            linear_electric_grid_models.append(linear_electric_grid_model)

            # Store objective value.
            objective_power_flow = (
                np.sum(np.real(load_power_vector_reference_candidate).ravel())
                + np.sum(np.real(power_flow_solution_candidate.loss.ravel()))
            )
            objective_power_flows.append(objective_power_flow)

        # Instantiate / reset optimization problem.
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

        # Define constraints.
        optimization_problem.constraints = pyo.ConstraintList()

        # Load.
        for load_index, load_name in enumerate(electric_grid_index.load_names):
            optimization_problem.constraints.add(
                optimization_problem.load_active_power_vector_change[load_name]
                >=
                0.5 * np.real(electric_grid_model.load_power_vector_nominal[load_index])
                - np.real(load_power_vector_reference[load_index])
            )
            optimization_problem.constraints.add(
                optimization_problem.load_active_power_vector_change[load_name]
                <=
                1.5 * np.real(electric_grid_model.load_power_vector_nominal[load_index])
                - np.real(load_power_vector_reference[load_index])
            )
            # Fixed power factor for reactive loads based on nominal load power factor.
            optimization_problem.constraints.add(
                optimization_problem.load_reactive_power_vector_change[load_name]
                ==
                optimization_problem.load_active_power_vector_change[load_name]
                * np.imag(electric_grid_model.load_power_vector_nominal[load_index])
                / np.real(electric_grid_model.load_power_vector_nominal[load_index])
            )

        for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):

            # Power.
            optimization_problem.constraints.add(
                optimization_problem.node_power_vector_wye_active_change[node_phase]
                ==
                pyo.quicksum(
                    electric_grid_model.load_incidence_wye_matrix[node_phase_index, load_index]
                    * optimization_problem.load_active_power_vector_change[load_name]
                    for load_index, load_name in enumerate(electric_grid_index.load_names)
                )
            )
            optimization_problem.constraints.add(
                optimization_problem.node_power_vector_wye_reactive_change[node_phase]
                ==
                pyo.quicksum(
                    electric_grid_model.load_incidence_wye_matrix[node_phase_index, load_index]
                    * optimization_problem.load_reactive_power_vector_change[load_name]
                    for load_index, load_name in enumerate(electric_grid_index.load_names)
                )
            )
            optimization_problem.constraints.add(
                optimization_problem.node_power_vector_delta_active_change[node_phase]
                ==
                pyo.quicksum(
                    electric_grid_model.load_incidence_delta_matrix[node_phase_index, load_index]
                    * optimization_problem.load_active_power_vector_change[load_name]
                    for load_index, load_name in enumerate(electric_grid_index.load_names)
                )
            )
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

        # Load.
        for load_index, load_name in enumerate(electric_grid_index.load_names):
            optimization_problem.constraints.add(
                optimization_problem.load_active_power_vector_change[load_name]
                >=
                -delta * np.real(electric_grid_model.load_power_vector_nominal.ravel()[load_index])
            )
            optimization_problem.constraints.add(
                optimization_problem.load_active_power_vector_change[load_name]
                <=
                delta * np.real(electric_grid_model.load_power_vector_nominal.ravel()[load_index])
            )

        # Voltage.
        for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):
            optimization_problem.constraints.add(
                optimization_problem.voltage_magnitude_vector_change[node_phase]
                >=
                -delta * np.abs(electric_grid_model.node_voltage_vector_no_load.ravel()[node_phase_index])
            )
            optimization_problem.constraints.add(
                optimization_problem.voltage_magnitude_vector_change[node_phase]
                <=
                delta * np.abs(electric_grid_model.node_voltage_vector_no_load.ravel()[node_phase_index])
            )

        # Branch flows.
        for branch_phase_index, branch_phase in enumerate(electric_grid_index.branches_phases):
            optimization_problem.constraints.add(
                optimization_problem.branch_power_vector_1_squared_change[branch_phase]
                >=
                -delta * np.abs(power_flow_solutions[0].branch_power_vector_1.ravel()[branch_phase_index] ** 2)
            )
            optimization_problem.constraints.add(
                optimization_problem.branch_power_vector_1_squared_change[branch_phase]
                <=
                delta * np.abs(power_flow_solutions[0].branch_power_vector_1.ravel()[branch_phase_index] ** 2)
            )
            optimization_problem.constraints.add(
                optimization_problem.branch_power_vector_2_squared_change[branch_phase]
                >=
                -delta * np.abs(power_flow_solutions[0].branch_power_vector_2.ravel()[branch_phase_index] ** 2)
            )
            optimization_problem.constraints.add(
                optimization_problem.branch_power_vector_2_squared_change[branch_phase]
                <=
                delta * np.abs(power_flow_solutions[0].branch_power_vector_2.ravel()[branch_phase_index] ** 2)
            )

        # Loss.
        optimization_problem.constraints.add(
            optimization_problem.loss_active_change
            >=
            -delta * np.sum(np.real(power_flow_solutions[0].loss))
        )
        optimization_problem.constraints.add(
            optimization_problem.loss_active_change
            <=
            delta * np.sum(np.real(power_flow_solutions[0].loss))
        )
        optimization_problem.constraints.add(
            optimization_problem.loss_reactive_change
            >=
            -delta * np.sum(np.imag(power_flow_solutions[0].loss))
        )
        optimization_problem.constraints.add(
            optimization_problem.loss_reactive_change
            <=
            delta * np.sum(np.imag(power_flow_solutions[0].loss))
        )

        # Define objective.
        cost = 0.0
        cost += (
            # Active load power.
            pyo.quicksum(
                optimization_problem.load_active_power_vector_change[load_name]
                + np.real(load_power_vector_reference[load_index])
                for load_index, load_name in enumerate(electric_grid_index.load_names)
            )
        )
        cost += (
            # Active loss.
            optimization_problem.loss_active_change
            + np.sum(np.real(power_flow_solution.loss))
        )
        optimization_problem.objective = (
            pyo.Objective(
                expr=cost,
                sense=pyo.minimize
            )
        )

        # Solve optimization problem.
        optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
        optimization_problems.append(optimization_problem)
        print(f"optimization_result.solver.termination_condition = {optimization_result.solver.termination_condition}")
        # optimization_problem.display()

        # Obtain load change value.
        load_active_power_vector_change = (
            np.zeros((len(electric_grid_index.load_names), 1), dtype=np.float)
        )
        load_reactive_power_vector_change = (
            np.zeros((len(electric_grid_index.load_names), 1), dtype=np.float)
        )
        for load_index, load_name in enumerate(electric_grid_index.load_names):
            load_active_power_vector_change[load_index] = (
                optimization_problem.load_active_power_vector_change[load_name].value
            )
            load_reactive_power_vector_change[load_index] = (
                optimization_problem.load_reactive_power_vector_change[load_name].value
            )
        load_power_vector_change_max = (
            max(
                np.max(abs(load_active_power_vector_change).ravel()),
                np.max(abs(load_reactive_power_vector_change).ravel())
            )
        )

        # Print change variables.
        print(f"load_active_power_vector_change = {load_active_power_vector_change.ravel()}")
        print(f"load_reactive_power_vector_change = {load_reactive_power_vector_change.ravel()}")
        print(f"load_power_vector_change_max = {load_power_vector_change_max}")

        # Check trust-region conditions and obtain load vector and power flow solution candidates for next iteration.
        # - Only if termination condition is not met, otherwise risk of division by zero.
        if load_power_vector_change_max > epsilon:
            # Get new load vector and power flow solution candidate.
            load_power_vector_reference_candidate = (
                load_power_vector_reference
                + load_active_power_vector_change.ravel()
                + 1.0j * load_reactive_power_vector_change.ravel()
            )
            power_flow_solution_candidate = (
                fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
                    electric_grid_model,
                    load_power_vector_reference_candidate
                )
            )

            # Obtain objective values.
            objective_power_flow = (
                np.sum(np.real(load_power_vector_reference_candidate).ravel())
                + np.sum(np.real(power_flow_solution_candidate.loss.ravel()))
            )
            objective_linear_model = (
                pyo.value(optimization_problem.objective)
            )

            # Check trust-region range conditions.
            sigma = (
                (objective_power_flows[-1] - objective_power_flow)
                / (objective_power_flows[-1] - objective_linear_model)
            )
            if sigma <= eta:
                delta *= gamma
            elif sigma > (1.0 - eta):
                delta = min(2 * delta, delta_max)

            # Print trust-region parameters.
            print(f"objective_power_flow = {objective_power_flow}")
            print(f"objective_linear_model = {objective_linear_model}")
            print(f"objective_power_flows[-1] = {objective_power_flows[-1]}")
            print(f"sigma = {sigma}")
            print(f"delta = {delta}")

        # Iterate counter.
        trust_region_iteration_count += 1

    # Instantiate results variables.

    # Load.
    load_active_power_vector = (
        pd.DataFrame(columns=electric_grid_index.load_names, index=[0], dtype=np.float)
    )
    load_reactive_power_vector = (
        pd.DataFrame(columns=electric_grid_index.load_names, index=[0], dtype=np.float)
    )

    # Voltage.
    voltage_magnitude_vector = (
        pd.DataFrame(columns=electric_grid_index.nodes_phases, index=[0], dtype=np.float)
    )

    # Branch flows.
    branch_power_vector_1_squared = (
        pd.DataFrame(columns=electric_grid_index.branches_phases, index=[0], dtype=np.float)
    )
    branch_power_vector_2_squared = (
        pd.DataFrame(columns=electric_grid_index.branches_phases, index=[0], dtype=np.float)
    )

    # Loss.
    loss_active = pd.DataFrame(columns=['total'], index=[0], dtype=np.float)
    loss_reactive = pd.DataFrame(columns=['total'], index=[0], dtype=np.float)

    # Obtain results.

    # Load.
    for load_index, load_name in enumerate(electric_grid_index.load_names):
        load_active_power_vector[load_name] = (
            optimization_problem.load_active_power_vector_change[load_name].value
            + np.real(load_power_vector_reference[load_index])
        )
        load_reactive_power_vector[load_name] = (
            optimization_problem.load_reactive_power_vector_change[load_name].value
            + np.imag(load_power_vector_reference[load_index])
        )

    # Voltage.
    for node_phase_index, node_phase in enumerate(electric_grid_index.nodes_phases):
        voltage_magnitude_vector[node_phase] = (
            optimization_problem.voltage_magnitude_vector_change[node_phase].value
            + np.abs(power_flow_solution.node_voltage_vector[node_phase_index])
        )

    # Branch flows.
    for branch_phase_index, branch_phase in enumerate(electric_grid_index.branches_phases):
        branch_power_vector_1_squared[branch_phase] = (
            optimization_problem.branch_power_vector_1_squared_change[branch_phase].value
            + np.abs(power_flow_solution.branch_power_vector_1[branch_phase_index] ** 2)
        )
        branch_power_vector_2_squared[branch_phase] = (
            optimization_problem.branch_power_vector_2_squared_change[branch_phase].value
            + np.abs(power_flow_solution.branch_power_vector_2[branch_phase_index] ** 2)
        )

    # Loss.
    loss_active['total'] = (
        optimization_problem.loss_active_change.value
        + np.real(power_flow_solution.loss)
    )
    loss_reactive['total'] = (
        optimization_problem.loss_reactive_change.value
        + np.imag(power_flow_solution.loss)
    )

    # Post-processing results.
    voltage_magnitude_vector_per_unit = (
            voltage_magnitude_vector
            / abs(power_flow_solutions[0].node_voltage_vector.transpose())
    )
    voltage_magnitude_vector_per_unit['mean'] = voltage_magnitude_vector_per_unit.mean(axis=1)
    load_active_power_vector_per_unit = (
            load_active_power_vector
            / np.real(electric_grid_model.load_power_vector_nominal.transpose())
    )
    load_active_power_vector_per_unit['mean'] = load_active_power_vector_per_unit.mean(axis=1)
    load_reactive_power_vector_per_unit = (
            load_reactive_power_vector
            / np.imag(electric_grid_model.load_power_vector_nominal.transpose())
    )
    load_reactive_power_vector_per_unit['mean'] = load_reactive_power_vector_per_unit.mean(axis=1)
    branch_power_vector_1_squared_per_unit = (
            branch_power_vector_1_squared
            / abs(power_flow_solutions[0].branch_power_vector_1.transpose() ** 2)
    )
    branch_power_vector_1_squared_per_unit['mean'] = branch_power_vector_1_squared_per_unit.mean(axis=1)
    loss_active_per_unit = (
            loss_active
            / np.real(power_flow_solutions[0].loss)
    )

    # Print some results.
    print(f"voltage_magnitude_vector_per_unit = \n{voltage_magnitude_vector_per_unit.to_string()}")
    print(f"load_active_power_vector_per_unit = \n{load_active_power_vector_per_unit.to_string()}")
    print(f"load_reactive_power_vector_per_unit = \n{load_reactive_power_vector_per_unit.to_string()}")
    print(f"branch_power_vector_1_squared_per_unit = \n{branch_power_vector_1_squared_per_unit.to_string()}")
    print(f"loss_active_per_unit = \n{loss_active_per_unit.to_string()}")


if __name__ == '__main__':
    main()
