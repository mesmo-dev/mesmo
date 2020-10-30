"""Example script for setting up and solving a flexible DER optimal operation problem."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    der_name = '4_2'  # Must be valid flexible DER from given scenario.
    plots = True  # If True, script may produce plots.

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    der_data = fledge.data_interface.DERData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain model.
    flexible_der_model = fledge.der_models.make_der_model(der_data, der_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define variables.
    flexible_der_model.define_optimization_variables(
        optimization_problem
    )

    # Define constraints.
    flexible_der_model.define_optimization_constraints(
        optimization_problem
    )

    # Define objective.
    flexible_der_model.define_optimization_objective(
        optimization_problem,
        price_data
    )

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
        flexible_der_model.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Plot results.
    if plots:

        for output in flexible_der_model.outputs:
            plt.plot(flexible_der_model.output_maximum_timeseries[output], label="Maximum", drawstyle='steps-post')
            plt.plot(flexible_der_model.output_minimum_timeseries[output], label="Minimum", drawstyle='steps-post')
            plt.plot(results['output_vector'][output], label="Optimal", drawstyle='steps-post')
            plt.legend()
            plt.title(f"Output: {output}")
            plt.show()
            plt.close()

        plt.plot(price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')], drawstyle='steps-post')
        plt.title(f"Price")
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
