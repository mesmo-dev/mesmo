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

    # Baseline optimization.

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
                -1.0 * optimization_problem.der_active_power_vector_change[timestep, der]
            )

    # Solve optimization problem.
    fledge.utils.solve_optimization(optimization_problem)

    # Obtain results.
    results_baseline = der_model_set.get_optimization_results(optimization_problem)
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
    results_baseline.update(
        fledge.data_interface.ResultsDict(
            der_active_power_vector=der_active_power_vector,
            der_reactive_power_vector=der_reactive_power_vector,
        )
    )

    # Peak shaving optimization.

    # Define peak power variable / constraint / objective.
    if optimization_problem.find_component('active_power_peak') is not None:
        optimization_problem.del_component('active_power_peak')
        optimization_problem.del_component('active_power_peak_index')
    optimization_problem.active_power_peak = (
        pyo.Var([0])
    )
    if optimization_problem.find_component('peak_power_constraints') is not None:
        optimization_problem.del_component('peak_power_constraints')
        optimization_problem.del_component('peak_power_constraints_index')
    optimization_problem.peak_power_constraints = pyo.ConstraintList()
    for timestep in scenario_data.timesteps:
        optimization_problem.peak_power_constraints.add(
            optimization_problem.active_power_peak[0]
            <=
            sum(
                optimization_problem.der_active_power_vector_change[timestep, der]
                + np.real(power_flow_solution.der_power_vector[der_index])
                for der_index, der in enumerate(electric_grid_model.ders)
            )
        )
    optimization_problem.peak_power_constraints.add(
        optimization_problem.active_power_peak[0]
        >=
        0.95 * results_baseline['der_active_power_vector'].sum(axis='columns').min()
    )

    # Solve optimization problem.
    fledge.utils.solve_optimization(optimization_problem)

    # Obtain results.
    results_peak_shaving = der_model_set.get_optimization_results(optimization_problem)
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
    results_peak_shaving.update(
        fledge.data_interface.ResultsDict(
            der_active_power_vector=der_active_power_vector,
            der_reactive_power_vector=der_reactive_power_vector,
        )
    )

    # Plots.

    for file_name, values in [
        ('total_demand_baseline', results_baseline.copy()),
        ('total_demand_peak_shaving', results_peak_shaving.copy())
    ]:

        values = values['der_active_power_vector']
        values = values.groupby(axis='columns', level='der_type').sum()
        values = values.reindex(['fixed_load', 'flexible_building', 'fixed_ev_charger'], axis='columns')
        values = values.rename(
            {
                'fixed_load': 'Non-cooling load',
                'flexible_building': 'Cooling load',
                'fixed_ev_charger': 'EV charging'
            },
            axis='columns'
        )
        values *= -1e-6

        figure = go.Figure()
        for column in values.columns:
            figure.add_trace(go.Scatter(
                x=values.index,
                y=values.loc[:, :column].sum(axis='columns').values,
                name=column,
                fill=('tozeroy' if column == values.columns[0] else 'tonexty'),
                line=go.scatter.Line(shape='hv')
            ))
        if file_name == 'total_demand_peak_shaving':
            figure.add_trace(go.Scatter(
                x=values.index,
                y=(-1e-6 * results_baseline['der_active_power_vector'].sum(axis='columns').values),
                name='Baseline',
                line=go.scatter.Line(shape='hv', color='dimgrey')
            ))
        figure.update_layout(
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            yaxis_title='Load [MW]',
            legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
            # legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, file_name))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
