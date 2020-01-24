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
    electric_grid_index = (
        fledge.electric_grid_models.ElectricGridIndex(scenario_name)
    )
    electric_grid_model = (
        fledge.electric_grid_models.ElectricGridModel(scenario_name)
    )

    # Obtain reference DER power vector and power flow solution.
    der_power_vector_reference = electric_grid_model.der_power_vector_nominal
    power_flow_solution = (
        fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(
            electric_grid_model,
            der_power_vector_reference
        )
    )

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
    linear_electric_grid_model.define_optimization_variables(optimization_problem)

    # Define linear electric grid model constraints.
    linear_electric_grid_model.define_optimization_constraints(optimization_problem)

    # Define DER constraints.
    # TODO: DERs are currently assumed to be only loads, hence negative values.
    optimization_problem.der_constraints = pyo.ConstraintList()
    for der_index, der_name in enumerate(electric_grid_index.der_names):
        optimization_problem.der_constraints.add(
            optimization_problem.der_active_power_vector_change[der_name]
            <=
            0.5 * np.real(electric_grid_model.der_power_vector_nominal[der_index])
            - np.real(der_power_vector_reference[der_index])
        )
        optimization_problem.der_constraints.add(
            optimization_problem.der_active_power_vector_change[der_name]
            >=
            1.5 * np.real(electric_grid_model.der_power_vector_nominal[der_index])
            - np.real(der_power_vector_reference[der_index])
        )
        # Fixed power factor for reactive power based on nominal power factor.
        optimization_problem.der_constraints.add(
            optimization_problem.der_reactive_power_vector_change[der_name]
            ==
            optimization_problem.der_active_power_vector_change[der_name]
            * np.imag(electric_grid_model.der_power_vector_nominal[der_index])
            / np.real(electric_grid_model.der_power_vector_nominal[der_index])
        )

    # Define objective.
    cost = 0.0
    cost += (
        # DER active power.
        # TODO: DERs are currently assumed to be only loads, hence negative values.
        -1.0 * pyo.quicksum(
            optimization_problem.der_active_power_vector_change[der_name]
            + np.real(der_power_vector_reference[der_index])
            for der_index, der_name in enumerate(electric_grid_index.der_names)
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
        optimization_problem
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


if __name__ == '__main__':
    main()
