"""Example script for setting up and solving an optimal power flow problem."""

import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.electric_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    results_path = (
        os.path.join(
            fledge.config.results_path,
            f'run_electric_grid_optimal_power_flow_simple_{fledge.config.timestamp}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
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
    for der_index, der in enumerate(electric_grid_model.ders):
        optimization_problem.der_constraints.add(
            optimization_problem.der_active_power_vector_change[0, der]
            <=
            0.5 * np.real(electric_grid_model.der_power_vector_nominal[der_index])
            - np.real(power_flow_solution.der_power_vector[der_index])
        )
        optimization_problem.der_constraints.add(
            optimization_problem.der_active_power_vector_change[0, der]
            >=
            1.5 * np.real(electric_grid_model.der_power_vector_nominal[der_index])
            - np.real(power_flow_solution.der_power_vector[der_index])
        )
        # Fixed power factor for reactive power based on nominal power factor.
        optimization_problem.der_constraints.add(
            optimization_problem.der_reactive_power_vector_change[0, der]
            ==
            optimization_problem.der_active_power_vector_change[0, der]
            * np.imag(electric_grid_model.der_power_vector_nominal[der_index])
            / np.real(electric_grid_model.der_power_vector_nominal[der_index])
        )

    # Define voltage limit constraints.
    linear_electric_grid_model.define_optimization_limits(
        optimization_problem,
        voltage_magnitude_vector_minimum=0.5 * np.abs(power_flow_solution.node_voltage_vector),
        voltage_magnitude_vector_maximum=1.5 * np.abs(power_flow_solution.node_voltage_vector),
    )

    # Define objective.
    optimization_problem.objective = (
        pyo.Objective(
            expr=0.0,
            sense=pyo.minimize
        )
    )
    optimization_problem.objective.expr += (
        # DER active power.
        # TODO: DERs are currently assumed to be only loads, hence negative values.
        -1.0 * sum(
            optimization_problem.der_active_power_vector_change[0, der]
            + np.real(power_flow_solution.der_power_vector[der_index])
            for der_index, der in enumerate(electric_grid_model.ders)
        )
    )
    optimization_problem.objective.expr += (
        # Active loss.
        optimization_problem.loss_active_change[0]
        + np.sum(np.real(power_flow_solution.loss))
    )

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
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
        power_flow_solution,
        in_per_unit=True,
        with_mean=True
    )

    # Print results.
    print(f"der_active_power_vector = \n{der_active_power_vector.to_string()}")
    print(f"der_reactive_power_vector = \n{der_reactive_power_vector.to_string()}")
    print(f"voltage_magnitude_vector = \n{voltage_magnitude_vector.to_string()}")
    print(f"branch_power_vector_1_squared = \n{branch_power_vector_1_squared.to_string()}")
    print(f"branch_power_vector_2_squared = \n{branch_power_vector_2_squared.to_string()}")
    print(f"loss_active = \n{loss_active.to_string()}")
    print(f"loss_reactive = \n{loss_reactive.to_string()}")

    # Store results as CSV.
    der_active_power_vector.to_csv(os.path.join(results_path, 'der_active_power_vector.csv'))
    der_reactive_power_vector.to_csv(os.path.join(results_path, 'der_reactive_power_vector.csv'))
    voltage_magnitude_vector.to_csv(os.path.join(results_path, 'voltage_magnitude_vector.csv'))
    branch_power_vector_1_squared.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared.csv'))
    branch_power_vector_2_squared.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared.csv'))
    loss_active.to_csv(os.path.join(results_path, 'loss_active.csv'))
    loss_reactive.to_csv(os.path.join(results_path, 'loss_reactive.csv'))

    # Obtain / print duals.
    (
        voltage_magnitude_vector_minimum_dual,
        voltage_magnitude_vector_maximum_dual,
        branch_power_vector_1_squared_maximum_dual,
        branch_power_vector_2_squared_maximum_dual
    ) = linear_electric_grid_model.get_optimization_limits_duals(
        optimization_problem
    )
    print(f"voltage_magnitude_vector_minimum_dual = \n{voltage_magnitude_vector_minimum_dual.to_string()}")
    print(f"voltage_magnitude_vector_maximum_dual = \n{voltage_magnitude_vector_maximum_dual.to_string()}")
    print(f"branch_power_vector_1_squared_maximum_dual = \n{branch_power_vector_1_squared_maximum_dual.to_string()}")
    print(f"branch_power_vector_2_squared_maximum_dual = \n{branch_power_vector_2_squared_maximum_dual.to_string()}")


if __name__ == '__main__':
    main()
