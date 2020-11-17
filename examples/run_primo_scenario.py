"""Project PRIMO scenario run script.

- This script relies on the PRIMO scenario definitions which are not included in this repository. If you have the
  scenario definition files, add the path to the definition in `config.yml` at `additional_data: []`.
"""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.plots
import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_pdd'
    results_path = fledge.utils.get_results_path('run_primo', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data / models.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Define optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define DER connection variables.
    optimization_problem.der_active_power_vector_change = (
        pyo.Var(scenario_data.timesteps.to_list(), electric_grid_model.ders.to_list())
    )
    optimization_problem.der_reactive_power_vector_change = (
        pyo.Var(scenario_data.timesteps.to_list(), electric_grid_model.ders.to_list())
    )

    # Define other DER variables / constraints.
    der_model_set.define_optimization_variables(
        optimization_problem
    )
    der_model_set.define_optimization_constraints(
        optimization_problem,
        electric_grid_model=electric_grid_model,
        power_flow_solution=power_flow_solution
    )

    # Define objective.
    optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    for timestep in scenario_data.timesteps:
        for der_index, der in enumerate(electric_grid_model.ders):
            optimization_problem.objective.expr += (
                optimization_problem.der_active_power_vector_change[timestep, der]
            )

    # Solve optimization problem.
    fledge.utils.solve_optimization(optimization_problem)

    # Obtain results.
    results = der_model_set.get_optimization_results(optimization_problem)
    der_active_power_vector = (
            pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    der_reactive_power_vector = (
        pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    for timestep in scenario_data.timesteps:
        for der_index, der in enumerate(electric_grid_model.ders):
            der_active_power_vector.at[timestep, der] = (
                optimization_problem.der_active_power_vector_change[timestep, der].value
                + np.real(power_flow_solution.der_power_vector[der_index])
            )
            der_reactive_power_vector.at[timestep, der] = (
                optimization_problem.der_reactive_power_vector_change[timestep, der].value
                + np.imag(power_flow_solution.der_power_vector[der_index])
            )
    results.update(
        fledge.data_interface.ResultsDict(
            der_active_power_vector=der_active_power_vector,
            der_reactive_power_vector=der_reactive_power_vector,
        )
    )

    # Plot results.
    figure = px.line(-results['der_active_power_vector'].sum(axis='columns'), line_shape='hv')
    figure.update_traces(fill='tozeroy')
    # figure.show(
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, 'total_demand'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
