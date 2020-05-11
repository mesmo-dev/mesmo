"""Example script for setting up and solving an optimal power flow problem."""

import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    results_path = (
        os.path.join(
            fledge.config.config['paths']['results'],
            f'run_electric_grid_optimal_power_flow_simple_{fledge.config.get_timestamp()}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

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
    # TODO: DERs are currently assumed to be only loads, hence negative power values.
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

    # Define objective.
    optimization_problem.objective = (
        pyo.Objective(
            expr=0.0,
            sense=pyo.minimize
        )
    )
    optimization_problem.objective.expr += (
        # DER active power.
        # TODO: DERs are currently assumed to be only loads, hence negative power values.
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
    optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    results = (
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            power_flow_solution,
            in_per_unit=True,
            with_mean=True
        )
    )

    # Print results.
    print(results)

    # Store results as CSV.
    results.to_csv(results_path)

    # Print results path.
    print("Results are stored in: " + results_path)


if __name__ == '__main__':
    main()
