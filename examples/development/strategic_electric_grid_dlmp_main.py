"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mesmo.kkt_conditions_with_state_space import StrategicMarket
import mesmo


def main():
    # TODO: Currently not working. Review limits below.

    # scenarios = [None]
    # scenario_name = "strategic_dso_market"
    # global strategic_der_model_set
    scenario_name = 'strategic_market_19_node'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name
                                                # , price_type='singapore_wholesale'
                                                )
    price_data.price_sensitivity_coefficient = 1e-6

    # Obtain models.
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model_set = (
        mesmo.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model,
            power_flow_solution
        )
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Instantiate centralized optimization problem.
    optimization_non_strategic = mesmo.utils.OptimizationProblem()
    optimization_strategic = mesmo.utils.OptimizationProblem()

    # Define electric grid problem.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.91 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.05 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 1.2 * electric_grid_model.branch_power_vector_magnitude_reference
    # active_power_vector_minimum = 0.0 * np.real(electric_grid_model.der_power_vector_reference)
    # active_power_vector_maximum = 1.3 * np.real(electric_grid_model.der_power_vector_reference)
    # reactive_power_vector_minimum = 0.0 * np.imag(electric_grid_model.der_power_vector_reference)
    # reactive_power_vector_maximum = 1.1 * np.imag(electric_grid_model.der_power_vector_reference)

    grid_cost_coefficient = 1.0

    der_model_set.define_optimization_problem(optimization_non_strategic,
                                              price_data,
                                              state_space_model=True,
                                              kkt_conditions=False,
                                              grid_cost_coefficient=grid_cost_coefficient
                                              )

    linear_electric_grid_model_set.define_optimization_problem(
        optimization_non_strategic,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        kkt_conditions=False,
        grid_cost_coefficient=grid_cost_coefficient
    )

    der_model_set.define_optimization_problem(optimization_strategic,
                                              price_data,
                                              state_space_model=True,
                                              kkt_conditions=False,
                                              grid_cost_coefficient=grid_cost_coefficient
                                              )

    linear_electric_grid_model_set.define_optimization_problem(
        optimization_strategic,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        kkt_conditions=False,
        grid_cost_coefficient=grid_cost_coefficient
    )

    strategic_scenario = True
    if strategic_scenario:
        strategic_der_model_set = StrategicMarket(scenario_name)
        strategic_der_model_set.strategic_optimization_problem(
            optimization_strategic,
            price_data,
            node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
            node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
            branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
            # active_power_vector_minimum=active_power_vector_minimum,
            # active_power_vector_maximum=active_power_vector_maximum,
            # reactive_power_vector_minimum=reactive_power_vector_minimum,
            # reactive_power_vector_maximum=reactive_power_vector_maximum,
            big_m=100,
            grid_cost_coefficient=grid_cost_coefficient
        )

    # Define DER problem.

    # Solve centralized optimization problem.
    optimization_non_strategic.solve()
    optimization_strategic.solve()

    # Obtain results.
    flexible_der_type = ['flexible_generator', 'flexible_load']


    results_non_strategic = mesmo.problems.Results()
    results_strategic = mesmo.problems.Results()

    results_non_strategic.update(linear_electric_grid_model_set.get_optimization_results(optimization_non_strategic))
    results_non_strategic.update(der_model_set.get_optimization_results(optimization_non_strategic))
    results_strategic.update(linear_electric_grid_model_set.get_optimization_results(optimization_strategic))
    results_strategic.update(der_model_set.get_optimization_results(optimization_strategic))

    # Print results.
    # print(results_centralized)

    # Store results to CSV.
    results_non_strategic.save(results_path)
    results_strategic.save(results_path)

    # Obtain DLMPs.
    dlmps_non_strategic = linear_electric_grid_model_set.get_optimization_dlmps(optimization_non_strategic, price_data)
    dlmps_strategic = strategic_der_model_set.get_optimization_dlmps(optimization_strategic, price_data)

    dlmps_non_strategic_active_power = dlmps_non_strategic.electric_grid_total_dlmp_node_active_power
    dlmps_strategic_active_power = dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power

    dlmp_difference = dlmps_strategic_active_power - dlmps_non_strategic_active_power

    strategic_flexible_der_active_power_vector = results_strategic.der_active_power_vector[flexible_der_type]
    non_strategic_flexible_der_active_power_vector = results_non_strategic.der_active_power_vector[flexible_der_type]
    active_power_vector_difference = strategic_flexible_der_active_power_vector - non_strategic_flexible_der_active_power_vector




    # Print DLMPs.
    # print(dlmps_non_strategic)

    # Store DLMPs as CSV.
    dlmps_non_strategic.save(results_path)
    dlmps_strategic.save(results_path)

    der_name = '01_10'
    price_data_dlmps = price_data.copy()
    price_data_dlmps.price_timeseries = dlmps_non_strategic['electric_grid_total_dlmp_price_timeseries']

    # Plot: Price comparison.
    price_active_wholesale = (
            1e6 / scenario_data.scenario.at['base_apparent_power']
            * price_data.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    )
    price_active_dlmp = (
            1e6 / scenario_data.scenario.at['base_apparent_power']
            * price_data_dlmps.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    )
    price_reactive_dlmp = (
            1e6 / scenario_data.scenario.at['base_apparent_power']
            * price_data_dlmps.price_timeseries.loc[:, ('reactive_power', slice(None), der_name)].iloc[:, 0]
    )

    title = 'Price comparison'
    filename = 'price_timeseries_comparison'
    y_label = 'Price'
    value_unit = 'S$/MWh'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=price_active_wholesale.index,
        y=price_active_wholesale.values,
        name='Wholesale price',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=price_active_dlmp.index,
        y=price_active_dlmp.values,
        name='DLMP (active power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=price_reactive_dlmp.index,
        y=price_reactive_dlmp.values,
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
    mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, filename))

    # Plot: Active power comparison.
    active_power_centralized = (
            1e-6 * scenario_data.scenario.at['base_apparent_power']
            * results_non_strategic['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
    )
    active_power_decentralized = (
            1e-6 * scenario_data.scenario.at['base_apparent_power']
            * results_non_strategic['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
    )

    title = 'Active power comparison'
    filename = 'active_power_comparison'
    y_label = 'Active power'
    value_unit = 'MW'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=active_power_centralized.index,
        y=active_power_centralized.values,
        name='Centralized solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=active_power_decentralized.index,
        y=active_power_decentralized.values,
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
    mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, filename))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
