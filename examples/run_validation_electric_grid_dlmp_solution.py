"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    results_path = fledge.utils.get_results_path('run_electric_grid_optimal_operation', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name, price_type='singapore_wholesale')

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

    # Instantiate centralized optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    voltage_magnitude_vector_minimum[
        fledge.utils.get_index(electric_grid_model.nodes, node_name='4')
    ] *= 0.965 / 0.5
    voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_vector_squared_maximum = 10.0 * np.abs(electric_grid_model.branch_power_vector_magnitude_reference ** 2)
    branch_power_vector_squared_maximum[
        fledge.utils.get_index(electric_grid_model.branches, branch_type='line', branch_name='2')
    ] *= 1.2 / 10.0
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
    )

    # Define grid  / centralized objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_data,
        scenario_data.timesteps
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

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data,
        electric_grid_model=electric_grid_model
    )

    # Solve centralized optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")

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
            price_data,
            scenario_data.timesteps
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs as CSV.
    dlmps.to_csv(results_path)

    # Validate DLMPs.
    der_name = '4_2'
    price_data_dlmps = price_data.copy()
    price_data_dlmps.price_timeseries = dlmps['electric_grid_total_dlmp_price_timeseries']

    # Instantiate decentralized DER optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define DER variables.
    der_model_set.der_models[der_name].define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.der_models[der_name].define_optimization_constraints(
        optimization_problem
    )

    # Define objective (DER operation cost minimization).
    der_model_set.der_models[der_name].define_optimization_objective(
        optimization_problem,
        price_data_dlmps
    )

    # Solve decentralized DER optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")

    # Obtain results.
    results_validation = (
        der_model_set.der_models[der_name].get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results_validation)

    # Define Plotly default options.
    pio.templates.default = go.layout.Template(pio.templates['simple_white'])
    pio.templates.default.layout.update(
        font_family=fledge.config.config['plots']['font_family'][0],
        legend=go.layout.Legend(borderwidth=1),
        xaxis=go.layout.XAxis(showgrid=True),
        yaxis=go.layout.YAxis(showgrid=True)
    )

    # Plot: Price comparison.
    values_1 = price_data.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    values_2 = price_data_dlmps.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    values_3 = price_data_dlmps.price_timeseries.loc[:, ('reactive_power', slice(None), der_name)].iloc[:, 0]

    title = 'Price comparison'
    filename = 'price_timeseries_comparison'
    y_label = 'Price'
    value_unit = 'S$/kWh'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Wholesale price',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='DLMP (active power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_3.index,
        y=values_3.values,
        name='DLMP (reactive power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.update_layout(
        title=title,
        yaxis_title=f'{y_label} [{value_unit}]',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.5, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

    # Plot: Active power comparison.
    values_1 = -1e-6 * results['output_vector'].loc[:, (der_name, 'active_power')]
    values_2 = -1e-6 * results_validation['output_vector'].loc[:, 'active_power']

    title = 'Active power comparison'
    filename = 'active_power_comparison'
    y_label = 'Active power'
    value_unit = 'MW'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Centralized solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='DER (decentralized) solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.update_layout(
        title=title,
        yaxis_title=f'{y_label} [{value_unit}]',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
