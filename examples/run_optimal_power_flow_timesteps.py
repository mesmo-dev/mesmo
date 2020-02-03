"""Example script for setting up and solving an optimal power flow problem."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.linear_electric_grid_models
import fledge.electric_grid_models
import fledge.power_flow_solvers


def main():

    # Settings.
    scenario_name = "singapore_6node"

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    fixed_load_data = fledge.database_interface.FixedLoadData(scenario_name)
    ev_charger_data = fledge.database_interface.EVChargerData(scenario_name)
    flexible_load_data = fledge.database_interface.FlexibleLoadData(scenario_name)

    # Obtain models.
    electric_grid_index = (
        fledge.electric_grid_models.ElectricGridIndex(scenario_name)
    )
    electric_grid_model = (
        fledge.electric_grid_models.ElectricGridModel(scenario_name)
    )
    der_models = dict.fromkeys(electric_grid_index.der_names)
    flexible_load_der_names = []
    for der_name in electric_grid_index.der_names:
        if der_name in fixed_load_data.fixed_loads['der_name']:
            der_models[der_name] = (
                fledge.der_models.FixedLoadModel(
                    fixed_load_data,
                    der_name
                )
            )
        elif der_name in ev_charger_data.ev_chargers['der_name']:
            der_models[der_name] = (
                fledge.der_models.EVChargerModel(
                    ev_charger_data,
                    der_name
                )
            )
        elif der_name in flexible_load_data.flexible_loads['der_name']:
            der_models[der_name] = (
                fledge.der_models.FlexibleLoadModel(
                    flexible_load_data,
                    der_name
                )
            )
            flexible_load_der_names.append(der_name)
        else:
            raise ValueError(f"Cannot determine DER type of DER: {der_name}")

    # Obtain reference DER power vector and power flow solution.
    power_flow_solution = fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(electric_grid_model)

    # Obtain linear electric grid model.
    linear_electric_grid_model = (
        fledge.linear_electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(optimization_problem, scenario_data.timesteps)

    # Define linear electric grid model constraints.
    linear_electric_grid_model.define_optimization_constraints(optimization_problem, scenario_data.timesteps)

    # Define flexible DER variables.
    der_state_names = [
        (der_name, state_name)
        for der_name in flexible_load_der_names
        for state_name in der_models[der_name].state_names
    ]
    der_control_names = [
        (der_name, control_name)
        for der_name in flexible_load_der_names
        for control_name in der_models[der_name].control_names
    ]
    der_output_names = [
        (der_name, output_name)
        for der_name in flexible_load_der_names
        for output_name in der_models[der_name].output_names
    ]
    optimization_problem.state_vector = pyo.Var(scenario_data.timesteps, der_state_names)
    optimization_problem.control_vector = pyo.Var(scenario_data.timesteps, der_control_names)
    optimization_problem.output_vector = pyo.Var(scenario_data.timesteps, der_output_names)

    # Define DER constraints.
    for der_model in der_models.values():
        # Internal DER constraints (for flexible DERs).
        der_model.define_optimization_constraints(
            optimization_problem
        )
        # Constraints for the connection with the DER power vector of the electric grid.
        der_model.define_optimization_connection_electric_grid(
            optimization_problem,
            power_flow_solution,
            electric_grid_index
        )

    # Define branch limit constraints.
    # TODO: This is an arbitrary limit on the minimum branch flow, just to demonstrate the functionality.
    optimization_problem.branch_limit_constraints = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_index.branches_phases.to_list(),
        rule=lambda optimization_problem, timestep, *branch_phase: (
            optimization_problem.branch_power_vector_1_squared_change[timestep, branch_phase]
            >=
            0.3 * np.abs(power_flow_solution.branch_power_vector_1.ravel()[electric_grid_index.branches_phases.get_loc(branch_phase)] ** 2)
            - np.abs(power_flow_solution.branch_power_vector_1.ravel()[electric_grid_index.branches_phases.get_loc(branch_phase)] ** 2)
        )
    )

    # Define objective.
    cost = 0.0
    cost += (
        # DER active power.
        # TODO: DERs are currently assumed to be only loads, hence negative values.
        -1.0 * pyo.quicksum(
            optimization_problem.der_active_power_vector_change[timestep, der_name]
            + np.real(power_flow_solution.der_power_vector[der_index])
            for timestep in scenario_data.timesteps
            for der_index, der_name in enumerate(electric_grid_index.der_names)
        )
    )
    cost += (
        # Active loss.
        pyo.quicksum(
            optimization_problem.loss_active_change[timestep]
            for timestep in scenario_data.timesteps
        )
        + np.sum(np.real(power_flow_solution.loss))
    )
    optimization_problem.objective = (
        pyo.Objective(
            expr=cost,
            sense=pyo.minimize
        )
    )

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    if optimization_result.solver.termination_condition is not pyo.TerminationCondition.optimal:
        raise Exception(f"Invalid solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    (
        der_active_power_vector,
        der_reactive_power_vector,
        voltage_magnitude_vector,
        branch_power_vector_1_squared,
        branch_power_vector_2_squared,
        loss_active,
        loss_reactive
    ) = linear_electric_grid_model.get_optimization_results(
        optimization_problem,
        scenario_data.timesteps
    )

    # Post-processing results.
    voltage_magnitude_vector_per_unit = (
        voltage_magnitude_vector
        / abs(power_flow_solution.node_voltage_vector.transpose())
    )
    voltage_magnitude_vector_per_unit['mean'] = voltage_magnitude_vector_per_unit.mean(axis=1)
    der_active_power_vector_per_unit = (
        der_active_power_vector
        / np.real(electric_grid_model.der_power_vector_nominal.transpose())
    )
    der_active_power_vector_per_unit['mean'] = der_active_power_vector_per_unit.mean(axis=1)
    der_reactive_power_vector_per_unit = (
        der_reactive_power_vector
        / np.imag(electric_grid_model.der_power_vector_nominal.transpose())
    )
    der_reactive_power_vector_per_unit['mean'] = der_reactive_power_vector_per_unit.mean(axis=1)
    branch_power_vector_1_squared_per_unit = (
        branch_power_vector_1_squared
        / abs(power_flow_solution.branch_power_vector_1.transpose() ** 2)
    )
    branch_power_vector_1_squared_per_unit['mean'] = branch_power_vector_1_squared_per_unit.mean(axis=1)
    loss_active_per_unit = (
        loss_active
        / np.real(power_flow_solution.loss)
    )

    # Print some results.
    print(f"voltage_magnitude_vector_per_unit = \n{voltage_magnitude_vector_per_unit.to_string()}")
    print(f"der_active_power_vector_per_unit = \n{der_active_power_vector_per_unit.to_string()}")
    print(f"der_reactive_power_vector_per_unit = \n{der_reactive_power_vector_per_unit.to_string()}")
    print(f"branch_power_vector_1_squared_per_unit = \n{branch_power_vector_1_squared_per_unit.to_string()}")
    print(f"loss_active_per_unit = \n{loss_active_per_unit.to_string()}")

    # Obtain duals.
    branch_limit_duals = (
        pd.DataFrame(columns=electric_grid_index.branches_phases, index=scenario_data.timesteps, dtype=np.float)
    )
    for timestep in scenario_data.timesteps:
        for branch_phase_index, branch_phase in enumerate(electric_grid_index.branches_phases):
            branch_limit_duals.at[timestep, branch_phase] = (
                optimization_problem.dual[optimization_problem.branch_limit_constraints[timestep, branch_phase]]
            )
    print(f"branch_limit_duals = {branch_limit_duals.to_string()}")


if __name__ == '__main__':
    main()
