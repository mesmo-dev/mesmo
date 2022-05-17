"""Example script for setting up and solving a gas grid optimal operation problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = 'gas_grid_test'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    price_data = mesmo.data_interface.PriceData(scenario_name)

    # Obtain models.
    gas_grid_model = mesmo.gas_grid_models.GasGridModel(scenario_name)
    gas_flow_solution = mesmo.gas_grid_models.GasFlowSolution(gas_grid_model)
    linear_gas_grid_model_set = (
        mesmo.gas_grid_models.LinearGasGridModelSet(
            gas_grid_model,
            gas_flow_solution
        )
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = mesmo.utils.OptimizationProblem()

    # Define thermal grid problem.
    node_pressure_vector_minimum = 1.5 * gas_flow_solution.node_pressure_vector
    gas_branch_flow_vector_maximum = 10.0 * gas_flow_solution.gas_branch_flow_vector
    linear_gas_grid_model_set.define_optimization_problem(
        optimization_problem,
        price_data,
        node_pressure_vector_minimum=node_pressure_vector_minimum,
        gas_branch_flow_vector_maximum=gas_branch_flow_vector_maximum
    )

    # Define DER problem.
    der_model_set.define_optimization_problem(optimization_problem, price_data)

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = mesmo.problems.Results()
    results.update(linear_gas_grid_model_set.get_optimization_results(optimization_problem))
    results.update(der_model_set.get_optimization_results(optimization_problem))

    # Print results.
    print(results)

    # Store results to CSV.
    results.save(results_path)

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
