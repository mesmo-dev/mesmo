"""Example script for setting up and solving an electric grid optimal operation problem."""

import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    results_path = (
        os.path.join(
            fledge.config.config['paths']['results'],
            f'run_electric_grid_optimal_operation_{fledge.config.get_timestamp()}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_type = 'singapore_wholesale'
    price_timeseries = price_data.price_timeseries_dict[price_type]

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    voltage_magnitude_vector_minimum = 0.5 * np.abs(power_flow_solution.node_voltage_vector)
    voltage_magnitude_vector_maximum = 1.5 * np.abs(power_flow_solution.node_voltage_vector)
    branch_power_vector_squared_maximum = 2.5 * np.abs(power_flow_solution.branch_power_vector_1 ** 2)
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem,
        electric_grid_model=electric_grid_model,
        power_flow_solution=power_flow_solution
    )

    # Define objective (DER operation cost minimization).
    der_model_set.define_optimization_objective(optimization_problem, price_timeseries)

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
            scenario_data.timesteps,
            in_per_unit=True,
            with_mean=True
        )
    )
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Store results as CSV.
    results.to_csv(results_path)

    # Obtain DLMPs.
    dlmps = (
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_timeseries,
            scenario_data.timesteps
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs as CSV.
    dlmps.to_csv(results_path)

    # Print results path.
    print("Results are stored in: " + results_path)


if __name__ == '__main__':
    main()
