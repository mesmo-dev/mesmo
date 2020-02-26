"""Example script for setting up and solving a flexible building optimal operation problem."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.power_flow_solvers


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    plots = True  # If True, script may produce plots.

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    der_data = fledge.database_interface.DERData(scenario_name)
    price_data = fledge.database_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_name = 'energy'
    price_timeseries = price_data.price_timeseries_dict[price_name]

    # Obtain model.
    der_name = der_data.flexible_buildings['der_name'][0]  # Pick first `der_name`.
    flexible_building_model = fledge.der_models.FlexibleBuildingModel(der_data, der_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define variables.
    flexible_building_model.define_optimization_variables(optimization_problem)

    # Define constraints.
    flexible_building_model.define_optimization_constraints(optimization_problem)

    # Disable thermal grid connection.
    optimization_problem.der_connection_constraints = pyo.ConstraintList()
    for timestep in scenario_data.timesteps:
        optimization_problem.der_connection_constraints.add(
            0.0
            ==
            optimization_problem.output_vector[timestep, der_name, 'grid_thermal_power_cooling']
        )

    # Define objective.
    flexible_building_model.define_optimization_objective(optimization_problem, price_timeseries)

    # Solve optimization problem.
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    (
        state_vector,
        control_vector,
        output_vector
    ) = flexible_building_model.get_optimization_results(
        optimization_problem
    )

    # Print results.
    print(f"state_name = \n{state_vector.to_string()}")
    print(f"control_name = \n{control_vector.to_string()}")
    print(f"output_name = \n{output_vector.to_string()}")

    # Plot results.
    if plots:

        for output_name in flexible_building_model.output_names:
            plt.plot(flexible_building_model.output_maximum_timeseries[output_name], label="Maximum", drawstyle='steps-post')
            plt.plot(flexible_building_model.output_minimum_timeseries[output_name], label="Minimum", drawstyle='steps-post')
            plt.plot(output_vector[output_name], label="Optimal", drawstyle='steps-post')
            plt.legend()
            plt.title(f"Output: {output_name}")
            plt.show()
            plt.close()

        plt.plot(price_timeseries['price_value'], drawstyle='steps-post')
        plt.title(f"Price: {price_name}")
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
