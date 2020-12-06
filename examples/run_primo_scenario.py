"""Project PRIMO scenario run script.

- This script relies on the PRIMO scenario definitions which are not included in this repository. If you have the
  scenario definition files, add the path to the definition in `config.yml` at `additional_data: []`.
"""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    results_path = fledge.utils.get_results_path(os.path.basename(__file__)[:-3], scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data / models.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Baseline optimization.

    # Define optimization problem.
    optimization_problem_baseline = fledge.utils.OptimizationProblem()

    # Define DER connection variables.
    optimization_problem_baseline.der_active_power_vector_change = (
        cp.Variable((len(scenario_data.timesteps), len(electric_grid_model.ders)))
    )
    optimization_problem_baseline.der_reactive_power_vector_change = (
        cp.Variable((len(scenario_data.timesteps), len(electric_grid_model.ders)))
    )

    # Define other DER variables / constraints.
    der_model_set.define_optimization_variables(
        optimization_problem_baseline
    )
    der_model_set.define_optimization_constraints(
        optimization_problem_baseline,
        electric_grid_model=electric_grid_model,
        power_flow_solution=power_flow_solution
    )

    # Define objective.
    optimization_problem_baseline.objective += (
        -1.0 * sum(sum(optimization_problem_baseline.der_active_power_vector_change))
    )

    # Solve optimization problem.
    optimization_problem_baseline.solve()

    # Obtain results.
    results_baseline = der_model_set.get_optimization_results(optimization_problem_baseline)
    results_baseline.update(
        fledge.data_interface.ResultsDict(
            der_active_power_vector=pd.DataFrame(
                (
                    optimization_problem_baseline.der_active_power_vector_change.value
                    + np.array([np.real(power_flow_solution.der_power_vector.ravel())])
                ),
                columns=electric_grid_model.ders,
                index=scenario_data.timesteps
            ),
            der_reactive_power_vector=pd.DataFrame(
                (
                        optimization_problem_baseline.der_reactive_power_vector_change.value
                        + np.array([np.imag(power_flow_solution.der_power_vector.ravel())])
                ),
                columns=electric_grid_model.ders,
                index=scenario_data.timesteps
            ),
        )
    )

    # Peak shaving optimization.

    # Instantiate optimization problem.
    optimization_problem_peak_shaving = fledge.utils.OptimizationProblem()

    # Define DER connection variables.
    optimization_problem_peak_shaving.der_active_power_vector_change = (
        cp.Variable((len(scenario_data.timesteps), len(electric_grid_model.ders)))
    )
    optimization_problem_peak_shaving.der_reactive_power_vector_change = (
        cp.Variable((len(scenario_data.timesteps), len(electric_grid_model.ders)))
    )

    # Define other DER variables / constraints.
    der_model_set.define_optimization_variables(
        optimization_problem_peak_shaving
    )
    der_model_set.define_optimization_constraints(
        optimization_problem_peak_shaving,
        electric_grid_model=electric_grid_model,
        power_flow_solution=power_flow_solution
    )

    # Define objective.
    optimization_problem_peak_shaving.objective += (
        -1.0 * sum(sum(optimization_problem_baseline.der_active_power_vector_change))
    )

    # Define peak power variable / constraint / objective.
    optimization_problem_peak_shaving.active_power_peak = cp.Variable()
    optimization_problem_peak_shaving.constraints.append(
        optimization_problem_peak_shaving.active_power_peak
        * np.ones(len(scenario_data.timesteps))
        <=
        cp.sum(
            (
                optimization_problem_peak_shaving.der_active_power_vector_change
                + np.array([np.real(power_flow_solution.der_power_vector.ravel())])
            ),
            axis=1  # Sum along DERs, i.e. sum for each timestep.
        )
    )
    optimization_problem_peak_shaving.constraints.append(
        optimization_problem_peak_shaving.active_power_peak
        >=
        0.95 * results_baseline['der_active_power_vector'].sum(axis='columns').min()
    )

    # Solve optimization problem.
    optimization_problem_peak_shaving.solve()

    # Obtain results.
    results_peak_shaving = der_model_set.get_optimization_results(optimization_problem_peak_shaving)
    results_peak_shaving.update(
        dict(
            der_active_power_vector=pd.DataFrame(
                (
                    optimization_problem_peak_shaving.der_active_power_vector_change.value
                    + np.array([np.real(power_flow_solution.der_power_vector.ravel())])
                ),
                columns=electric_grid_model.ders,
                index=scenario_data.timesteps
            ),
            der_reactive_power_vector=pd.DataFrame(
                (
                    optimization_problem_peak_shaving.der_reactive_power_vector_change.value
                    + np.array([np.imag(power_flow_solution.der_power_vector.ravel())])
                ),
                columns=electric_grid_model.ders,
                index=scenario_data.timesteps
            ),
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
