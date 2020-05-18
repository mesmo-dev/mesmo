"""Example script for setting up and solving a flexible load optimal operation problem."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    plots = True  # If True, script may produce plots.

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    der_data = fledge.data_interface.DERData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_type = 'singapore_wholesale'
    price_timeseries = price_data.price_timeseries_dict[price_type]

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
    flexible_load_model.define_optimization_objective(optimization_problem, price_timeseries)

    # Solve optimization problem.
    optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    results = (
        flexible_load_model.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Plot results.
    if plots:

        for output in flexible_load_model.outputs:
            plt.plot(flexible_load_model.output_maximum_timeseries[output], label="Maximum", drawstyle='steps-post')
            plt.plot(flexible_load_model.output_minimum_timeseries[output], label="Minimum", drawstyle='steps-post')
            plt.plot(results['output_vector'][output], label="Optimal", drawstyle='steps-post')
            plt.legend()
            plt.title(f"Output: {output}")
            plt.show()
            plt.close()

        plt.plot(price_timeseries['price_value'], drawstyle='steps-post')
        plt.title(f"Price: {price_type}")
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
