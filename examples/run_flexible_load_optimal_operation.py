"""Example script for setting up and solving a flexible load optimal operation problem."""

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
    scenario_name = "singapore_6node"
    plots = True  # If True, script may produce plots.

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    der_data = fledge.database_interface.ElectricGridDERData(scenario_name)
    price_data = fledge.database_interface.PriceData(scenario_name)

    # Obtain model.
    der_name = der_data.flexible_loads['der_name'][0]  # Pick first `der_name`.
    flexible_load_model = fledge.der_models.FlexibleLoadModel(der_data, der_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define variables.
    flexible_load_model.define_optimization_variables(optimization_problem)

    # Define constraints.
    flexible_load_model.define_optimization_constraints(optimization_problem)

    # Define objective.
    price_name = 'energy'

    cost = 0.0
    cost += (
        pyo.quicksum(
            -1.0
            * price_data.price_timeseries_dict[price_name].at[timestep, 'price_value']
            * optimization_problem.output_vector[timestep, der_name, output_name]
            for timestep in scenario_data.timesteps
            for output_name in ['active_power', 'reactive_power']
        )
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
        state_vector,
        control_vector,
        output_vector
    ) = flexible_load_model.get_optimization_results(
        optimization_problem
    )

    # Print results.
    print(f"state_name = \n{state_vector.to_string()}")
    print(f"control_name = \n{control_vector.to_string()}")
    print(f"output_name = \n{output_vector.to_string()}")

    # Plot results.
    if plots:

        for output_name in flexible_load_model.output_names:
            plt.plot(flexible_load_model.output_maximum_timeseries[output_name], label="Maximum", drawstyle='steps-post')
            plt.plot(flexible_load_model.output_minimum_timeseries[output_name], label="Minimum", drawstyle='steps-post')
            plt.plot(output_vector[output_name], label="Optimal", drawstyle='steps-post')
            plt.legend()
            plt.title(f"Output: {output_name}")
            plt.show()
            plt.close()

        plt.plot(price_data.price_timeseries_dict[price_name]['price_value'], drawstyle='steps-post')
        plt.title(f"Price: {price_name}")
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
