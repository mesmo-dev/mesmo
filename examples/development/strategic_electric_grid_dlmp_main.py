"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
pd.options.plotting.backend = "matplotlib"
import plotly.express as px
import plotly.graph_objects as go
# pd.options.plotting.backend = "plotly"
from PIL._util import deferred_error

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
    node_voltage_magnitude_vector_minimum = 0.8 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.05 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 1.2 * electric_grid_model.branch_power_vector_magnitude_reference

    grid_cost_coefficient = 1

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
    # results_non_strategic.save(results_path)
    # results_strategic.save(results_path)

    # Obtain DLMPs.
    dlmps_non_strategic = linear_electric_grid_model_set.get_optimization_dlmps(optimization_non_strategic, price_data)
    dlmps_strategic = strategic_der_model_set.get_optimization_dlmps(optimization_strategic, price_data)

    # DLMPs in non-strategic scenario Timeseries for Node 10 for three phases
    # Energy portion of DLMP:
    node_10_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_energy_dlmps_non_strategic_active_power.columns = \
        node_10_energy_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Loss portion of DLMP:
    node_10_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_loss_dlmps_non_strategic_active_power.columns = \
        node_10_loss_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Voltage portion of DLMP:
    node_10_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_voltage_dlmps_non_strategic_active_power.columns = \
        node_10_voltage_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Congestion portion of DLMP:
    node_10_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_congestion_dlmps_non_strategic_active_power.columns = \
        node_10_congestion_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Total DLMP:
    node_10_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_total_dlmps_non_strategic_active_power.columns = \
        node_10_total_dlmps_non_strategic_active_power.columns.to_flat_index()


    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    nodal_energy_dlmps_non_strategic_active_power.index = \
        nodal_energy_dlmps_non_strategic_active_power.index.to_flat_index()

    # Loss portion of DLMPs:
    nodal_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    nodal_loss_dlmps_non_strategic_active_power.index = \
        nodal_loss_dlmps_non_strategic_active_power.index.to_flat_index()

    # Voltage portion of DLMPs:
    nodal_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    nodal_voltage_dlmps_non_strategic_active_power.index = \
        nodal_voltage_dlmps_non_strategic_active_power.index.to_flat_index()

    # Congestion portion of DLMPs:
    nodal_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    nodal_congestion_dlmps_non_strategic_active_power.index = \
        nodal_congestion_dlmps_non_strategic_active_power.index.to_flat_index()

    # Total DLMPs:
    nodal_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    nodal_total_dlmps_non_strategic_active_power.index = \
        nodal_total_dlmps_non_strategic_active_power.index.to_flat_index()



    # ------------------DLMPs in Strategic scenario Timeseries for Node 10 for three phases------------------------:
    # Energy portion of DLMP:
    node_10_energy_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_energy_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_energy_dlmps_non_strategic_active_power.columns = \
        node_10_energy_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Loss portion of DLMP:
    node_10_loss_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_loss_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_loss_dlmps_non_strategic_active_power.columns = \
        node_10_loss_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Voltage portion of DLMP:
    node_10_voltage_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_voltage_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_voltage_dlmps_non_strategic_active_power.columns = \
        node_10_voltage_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Congestion portion of DLMP:
    node_10_congestion_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_congestion_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_congestion_dlmps_non_strategic_active_power.columns = \
        node_10_congestion_dlmps_non_strategic_active_power.columns.to_flat_index()

    # Total DLMP:
    node_10_total_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power.loc[:, ('no_source', '10', slice(None))]
    node_10_total_dlmps_non_strategic_active_power.columns = \
        node_10_total_dlmps_non_strategic_active_power.columns.to_flat_index()

    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_energy_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    # nodal_energy_dlmps_non_strategic_active_power.index = \
    #     nodal_energy_dlmps_non_strategic_active_power.index.to_flat_index()

    # Loss portion of DLMP:
    nodal_loss_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_loss_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    # nodal_loss_dlmps_non_strategic_active_power.index = \
    #     nodal_loss_dlmps_non_strategic_active_power.index.to_flat_index()

    # Voltage portion of DLMP:
    nodal_voltage_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_voltage_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    # nodal_voltage_dlmps_non_strategic_active_power.index = \
    #     nodal_voltage_dlmps_non_strategic_active_power.index.to_flat_index()

    # Congestion portion of DLMP:
    nodal_congestion_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_congestion_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    # nodal_congestion_dlmps_non_strategic_active_power.index = \
    #     nodal_congestion_dlmps_non_strategic_active_power.index.to_flat_index()

    # Total DLMP:
    nodal_total_dlmps_non_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power.loc['2021-02-22 15:00:00', :]
    # nodal_total_dlmps_non_strategic_active_power.index = \
    #     nodal_total_dlmps_non_strategic_active_power.index.to_flat_index()

    nodal_non_strategic_stacked_bar_data_phase_1 = {
        'Energy': nodal_energy_dlmps_non_strategic_active_power[('no_source', slice(None), 1)].values,
        'Loss': nodal_loss_dlmps_non_strategic_active_power[('no_source', slice(None), 1)].values,
        'Voltage': nodal_voltage_dlmps_non_strategic_active_power[('no_source', slice(None), 1)].values,
        'Congestion': nodal_congestion_dlmps_non_strategic_active_power[('no_source', slice(None), 1)].values
    }
    nodal_non_strategic_stacked_bar_data_phase_1 = pd.DataFrame(nodal_non_strategic_stacked_bar_data_phase_1)

    nodal_non_strategic_stacked_bar_data_phase_2 = {
        'Energy': nodal_energy_dlmps_non_strategic_active_power[('no_source', slice(None), 2)].values,
        'Loss': nodal_loss_dlmps_non_strategic_active_power[('no_source', slice(None), 2)].values,
        'Voltage': nodal_voltage_dlmps_non_strategic_active_power[('no_source', slice(None), 2)].values,
        'Congestion': nodal_congestion_dlmps_non_strategic_active_power[('no_source', slice(None), 2)].values
    }
    nodal_non_strategic_stacked_bar_data_phase_2 = pd.DataFrame(nodal_non_strategic_stacked_bar_data_phase_2)

    nodal_non_strategic_stacked_bar_data_phase_3 = {
        'Energy': nodal_energy_dlmps_non_strategic_active_power[('no_source', slice(None), 3)].values,
        'Loss': nodal_loss_dlmps_non_strategic_active_power[('no_source', slice(None), 3)].values,
        'Voltage': nodal_voltage_dlmps_non_strategic_active_power[('no_source', slice(None), 3)].values,
        'Congestion': nodal_congestion_dlmps_non_strategic_active_power[('no_source', slice(None), 3)].values
    }
    nodal_non_strategic_stacked_bar_data_phase_3 = pd.DataFrame(nodal_non_strategic_stacked_bar_data_phase_3)

    fig, ax = plt.subplots()
    nodal_total_dlmps_non_strategic_active_power[('no_source', slice(None), 1)].plot(
        ax=ax,
        label='DER active power marginal cost',
        color='g',
        marker='s'
    )
    nodal_non_strategic_stacked_bar_data_phase_1.loc['Loss'].plot(
        kind='bar',
        # stacked=True,
        ax=ax,
        label='DER offered marginal cost',
        # marker='^'
    )
    # nodal_total_dlmps_non_strategic_active_power[('no_source', slice(None), 3)].plot(
    #     ax=ax,
    #     label='DER offered marginal cost',
    #     color='b',
    #     marker='*'
    # )
    fig.suptitle('Offer comparison for the strategic generator')
    ax.set_ylabel('Offer price [$/kWh]')
    ax.set_xlabel('Time [h]')
    # ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.show()












    dlmps_strategic_active_power = dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power

    dlmp_difference = dlmps_strategic_active_power - dlmps_non_strategic_active_power

    strategic_flexible_der_active_power_vector = results_strategic.der_active_power_vector_per_unit[flexible_der_type]
    non_strategic_flexible_der_active_power_vector = results_non_strategic.der_active_power_vector_per_unit[flexible_der_type]
    active_power_vector_difference = strategic_flexible_der_active_power_vector - non_strategic_flexible_der_active_power_vector

    der_active_power_marginal_offers_timeseries = results_non_strategic.der_active_power_marginal_offers_timeseries.loc[
                                                  ('flexible_generator', '01_10'), :]
    der_active_power_strategic_marginal_offers_timeseries = dlmps_strategic.strategic_der_marginal_price_offers

    # Print DLMPs.
    # print(dlmps_non_strategic)

    # Store DLMPs as CSV.
    # dlmps_non_strategic.save(results_path)
    # dlmps_strategic.save(results_path)


    fig, ax = plt.subplots()
    der_active_power_marginal_offers_timeseries.plot(
        ax=ax,
        label='DER active power marginal cost',
        color='g',
        marker='s'
    )
    der_active_power_strategic_marginal_offers_timeseries[('flexible_generator', '01_10')].plot(
        ax=ax,
        label='DER offered marginal cost',
        color='r',
        marker='^'
    )
    fig.suptitle('Offer comparison for the strategic generator')
    ax.set_ylabel('Offer price [$/kWh]')
    ax.set_xlabel('Time [h]')
    # ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.show()
    fig.savefig('Offers.svg')

    fig, ax = plt.subplots()




















    # pd.options.plotting.backend = "plotly"
    # figure = go.Figure()
    # figure.add_scatter(
    #     x=der_active_power_strategic_marginal_offers_timeseries.index,
    #     y=der_active_power_marginal_offers_timeseries.values,
    #     name='no_reserve',
    #     line=go.scatter.Line()
    # )
    # figure.add_scatter(
    #     x=der_active_power_strategic_marginal_offers_timeseries.index,
    #     y=der_active_power_strategic_marginal_offers_timeseries.values.ravel(),
    #     name='no_reserve',
    #     line=go.scatter.Line()
    # )
    # figure.show()

    # figure.add_scatter(
    #     x=up_reserve_stage_1.index,
    #     y=up_reserve_stage_1.values,
    #     name='up_reserve',
    #     line=go.scatter.Line(shape='hv', width=4, dash='dot')
    # )
    # figure.add_scatter(
    #     x=down_reserve_stage_1.index,
    #     y=down_reserve_stage_1.values,
    #     name='down_reserve',
    #     line=go.scatter.Line(shape='hv', width=3, dash='dot')
    # )
    # figure.add_scatter(
    #     x=up_reserve_stage_1.index,
    #     y=(energy_stage_1 + up_reserve_stage_1).values,
    #     name='no_reserve + up_reserve',
    #     line=go.scatter.Line(shape='hv', width=2, dash='dot')
    # )
    # figure.add_scatter(
    #     x=up_reserve_stage_1.index,
    #     y=(energy_stage_1 - down_reserve_stage_1).values,
    #     name='no_reserve - down_reserve',
    #     line=go.scatter.Line(shape='hv', width=1, dash='dot')
    # )
    # figure.update_layout(
    #     title=f'Power balance',
    #     xaxis=go.layout.XAxis(tickformat='%H:%M'),
    #     legend=go.layout.Legend(x=0.01, xanchor='auto', y=0.99, yanchor='auto')
    # )
    # # figure.show()
    # mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, f'0_power_balance'))
    #










































    # strategic_der_marginal_and_strategic_offers_timeseries = pd.concat(
    #     [der_active_power_marginal_offers_timeseries, der_active_power_strategic_marginal_offers_timeseries],
    #     axis=1, keys=['marginal_offer', 'strategic_offer'])
    # fig = strategic_der_marginal_and_strategic_offers_timeseries.plot
    # fig.show()
    # a = np.array([der_active_power_strategic_marginal_offers_timeseries, der_active_power_marginal_offers_timeseries])

    title = 'Price comparison'
    filename = 'price_timeseries_comparison'
    y_label = 'Offer Price'
    value_unit = 'S$/MWh'

    figure = go.Figure()
    trace1 = go.Scatter(
        x=results_non_strategic.der_active_power_marginal_offers_timeseries.index,
        y=der_active_power_marginal_offers_timeseries,
    )
    trace2 = go.Scatter(
        x=results_non_strategic.der_active_power_marginal_offers_timeseries.index,
        y=der_active_power_strategic_marginal_offers_timeseries,
    )
    data = [trace1, trace2]
    fig=go.Figure(data)
    fig.show()

    # figure.add_trace(go.Line(
    #     x=strategic_der_marginal_and_strategic_offers_timeseries.index,
    #     y=strategic_der_marginal_and_strategic_offers_timeseries.values[0],
    #     name='Wholesale price',
    #     # fill='tozeroy',
    #     line=go.scatter.Line(shape='hv')
    # ))
    # figure.add_trace(go.Line(
    #     x=strategic_der_marginal_and_strategic_offers_timeseries.index,
    #     y=strategic_der_marginal_and_strategic_offers_timeseries.values[1],
    #     name='Wholesale price',
    #     # fill='tozeroy',
    #     line=go.scatter.Line(shape='hv')
    # ))
    # figure.update_layout(
    #     title=title,
    #     yaxis_title=f'{y_label} [{value_unit}]',
    #     xaxis=go.layout.XAxis(tickformat='%H:%M'),
    #     legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.5, yanchor='auto')
    # )
    # figure.show()
"""
    offer_figs = px.line(strategic_der_marginal_and_strategic_offers_timeseries.values.transpose(), x='timestep', y='offer', markers=True)

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
"""

if __name__ == '__main__':
    main()
