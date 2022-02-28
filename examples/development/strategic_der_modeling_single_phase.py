"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
    # scenario_name = 'strategic_market_19_node'
    scenario_name = 'polimi_test_case'
    strategic_scenario = True

    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name
                                                # , price_type='singapore_wholesale'
                                                )
    # price_data.price_sensitivity_coefficient = 1e-6
    # Run nominal operational problem:
    nominal_operation = mesmo.api.run_nominal_operation_problem(scenario_name, store_results=False)
    nominal_voltage = nominal_operation.node_voltage_magnitude_vector_per_unit
    nominal_branch_power_1 = nominal_operation.branch_power_magnitude_vector_1_per_unit
    nominal_branch_power_2 = nominal_operation.branch_power_magnitude_vector_2_per_unit

    # optimal_operation = mesmo.api.run_optimal_operation_problem(scenario_name, store_results=False)
    # optimal_voltage = optimal_operation.node_voltage_magnitude_vector_per_unit.min()
    # optimal_branch_power_1 = optimal_operation.branch_power_magnitude_vector_1_per_unit.max()
    # optimal_branch_power_2 = optimal_operation.branch_power_magnitude_vector_2_per_unit.max()
    # flexible_generator_optimal_dispatch = optimal_operation.der_active_power_vector_per_unit['flexible_generator']

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

    # Define electric grid problem.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.95 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.05 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 1 * electric_grid_model.branch_power_vector_magnitude_reference

    grid_cost_coefficient = 0.75

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

    if strategic_scenario:
        optimization_strategic = mesmo.utils.OptimizationProblem()

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

        strategic_der_model_set = StrategicMarket(scenario_name, strategic_der='pv_b10_strategic')
        strategic_der_model_set.strategic_optimization_problem(
            optimization_strategic,
            price_data,
            node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
            node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
            branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
            big_m=2500,
            kkt_conditions=False,
            grid_cost_coefficient=grid_cost_coefficient
        )

    # Define DER problem.

    # Solve centralized optimization problem.
    optimization_non_strategic.solve()
    optimization_strategic.solve()
    # a=1
    # a = optimization_non_strategic.duals['output_equation']
    # b = optimization_strategic.results['output_equation_mu']
    # c = a-b

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
    dlmp_difference = dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power - \
                      dlmps_non_strategic.electric_grid_total_dlmp_node_active_power

    flexible_der_active_power_non_strategic = results_non_strategic.der_active_power_vector_per_unit[flexible_der_type]
    flexible_der_active_power_strategic = results_strategic.der_active_power_vector_per_unit[flexible_der_type]

    flexible_der_reactive_power_non_strategic = results_non_strategic.der_reactive_power_vector_per_unit[
        flexible_der_type]
    flexible_der_reactive_power_strategic = results_strategic.der_reactive_power_vector_per_unit[flexible_der_type]

    report_time = '2021-02-22 14:00:00'

    x = np.arange(len(flexible_der_active_power_non_strategic.columns))
    width = 0.35
    fig, axes = plt.subplots(1, figsize=(12, 6))
    axes.bar(x + width / 2,
             flexible_der_active_power_non_strategic.loc[report_time],
             width=width,
             color='b',
             label='non_strategic')
    axes.bar(x - width / 2,
             flexible_der_active_power_strategic.loc[report_time],
             width=width,
             color='r',
             label='strategic')
    axes.set_xticks(x, flexible_der_active_power_strategic.columns)
    plt.xlabel('DER name')
    fig.set_tight_layout(True)
    axes.set_ylabel('Power dispatch [p.u]')
    axes.title.set_text(f"Flexible DER's active power dispatch at {report_time}")
    plt.xticks(rotation=-40, fontsize=8)
    axes.legend()
    axes.grid()
    fig.show()
    # fig.savefig('flexible_der_active_power_dispatch.svg')



    # DLMPs in non-strategic scenario Timeseries for Node 10 for three phases
    # Energy portion of DLMP:
    node_10_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # Loss portion of DLMP:
    node_10_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # Voltage portion of DLMP:
    node_10_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # Congestion portion of DLMP:
    node_10_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # TODO plotting the results
    # Total DLMP:
    node_10_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # ------------------DLMPs in Strategic scenario Timeseries for Node 10 for three phases------------------------:
    # Energy portion of DLMP:
    node_10_energy_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_energy_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # Loss portion of DLMP:
    node_10_loss_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_loss_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # Voltage portion of DLMP:
    node_10_voltage_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_voltage_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # Congestion portion of DLMP:
    node_10_congestion_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_congestion_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    # TODO plotting the results
    # Total DLMP:
    node_10_total_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power.loc[:, ('no_source', 'b10', 1)]

    fig, axes = plt.subplots(1, sharex=True, sharey=True, figsize=(12, 6))
    node_10_total_dlmps_non_strategic_active_power.plot(
        ax=axes,
        label='Non-strategic',
        y=('no_source', 'b10', 1),
        color='b',
        marker='s'
    )
    node_10_total_dlmps_strategic_active_power.plot(
        ax=axes,
        label='Strategic',
        y=('no_source', 'b10', 1),
        color='r',
        marker='^'
    )
    fig.suptitle('DLMP timeseries at node 10')
    plt.xlabel('Time [h]')
    fig.set_tight_layout(True)
    axes.set_ylabel('DLMP [$/kWh]')
    axes.title.set_text('Node 10 total DLMP')
    axes.legend()
    axes.grid()
    fig.show()
    # fig.savefig('dlmp_timeseries_node_10.svg')

    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc[report_time]

    # Loss portion of DLMPs:
    nodal_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc[report_time]

    # Voltage portion of DLMPs:
    nodal_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc[report_time]

    # Congestion portion of DLMPs:
    nodal_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc[report_time]

    # Total DLMPs:
    nodal_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc[report_time]

    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_energy_dlmp_node_active_power.loc[report_time]

    # Loss portion of DLMP:
    nodal_loss_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_loss_dlmp_node_active_power.loc[report_time]

    # Voltage portion of DLMP:
    nodal_voltage_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_voltage_dlmp_node_active_power.loc[report_time]

    # Congestion portion of DLMP:
    nodal_congestion_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_congestion_dlmp_node_active_power.loc[report_time]

    # Total DLMP:
    nodal_total_dlmps_strategic_active_power = \
        dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power.loc[report_time]

    x = np.arange(len(nodal_energy_dlmps_non_strategic_active_power.index))
    fig, axes = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
    axes[0].bar(x, nodal_energy_dlmps_non_strategic_active_power, label='Energy', color='g')
    axes[1].bar(x, nodal_energy_dlmps_strategic_active_power, label='Energy', color='g')
    axes[0].bar(x, nodal_loss_dlmps_non_strategic_active_power,
                bottom=nodal_energy_dlmps_non_strategic_active_power,
                label='Loss',
                color='b')
    axes[1].bar(x, nodal_loss_dlmps_strategic_active_power,
                bottom=nodal_energy_dlmps_strategic_active_power,
                label='Loss',
                color='b')
    axes[0].bar(x, nodal_voltage_dlmps_non_strategic_active_power,
                bottom=nodal_energy_dlmps_non_strategic_active_power + nodal_loss_dlmps_non_strategic_active_power,
                label='Voltage',
                color='r')
    axes[1].bar(x, nodal_voltage_dlmps_strategic_active_power,
                bottom=nodal_energy_dlmps_strategic_active_power + nodal_loss_dlmps_strategic_active_power,
                label='Voltage',
                color='r')
    axes[0].bar(x, nodal_congestion_dlmps_non_strategic_active_power,
                bottom=nodal_energy_dlmps_non_strategic_active_power + nodal_loss_dlmps_non_strategic_active_power +
                       nodal_voltage_dlmps_non_strategic_active_power,
                label='Congestion',
                color='y')
    axes[1].bar(x, nodal_congestion_dlmps_strategic_active_power,
                bottom=nodal_energy_dlmps_strategic_active_power + nodal_loss_dlmps_strategic_active_power +
                        nodal_voltage_dlmps_strategic_active_power,
                label='Congestion',
                color='y')
    axes[0].plot(x, nodal_total_dlmps_non_strategic_active_power, color='k', label='Total', marker='*')
    axes[1].plot(x, nodal_total_dlmps_strategic_active_power, color='k', label='Total', marker='*')
    axes[1].set_xticks(x, nodal_energy_dlmps_non_strategic_active_power.index)
    plt.xlabel('Node name')
    fig.set_tight_layout(True)
    axes[0].set_ylabel('DLMP [$/MW]')
    axes[1].set_ylabel('DLMP [$/MW]')
    axes[0].title.set_text(f"Non-strategic nodal DLMPs at {report_time}")
    axes[1].title.set_text(f"Strategic nodal DLMPs at {report_time}")
    plt.xticks(rotation=-40, fontsize=8)
    axes[0].legend()
    axes[1].legend()
    axes[0].grid()
    axes[1].grid()
    fig.show()
    # fig.savefig('contributions_to_DLMP.svg')


    fig, axes = plt.subplots(figsize=(12, 6))
    nodal_total_dlmps_non_strategic_active_power.plot(
        ax=axes,
        label='Non-strategic',
        # y=(slice(None), slice(None), 3),
        color='b',
        marker='*'
    )
    nodal_total_dlmps_strategic_active_power.plot(
        ax=axes,
        label='Strategic',
        # y=(slice(None), slice(None), 3),
        color='r',
        marker='^'
    )
    x = np.arange(len(nodal_total_dlmps_strategic_active_power.index))
    axes.set_xticks(x, nodal_total_dlmps_strategic_active_power.index)
    plt.xticks(rotation=-30, fontsize=8)
    fig.suptitle(f'Nodal DLMPs at {report_time}')
    plt.xlabel('Node name')
    fig.set_tight_layout(True)
    axes.set_ylabel('DLMP [$/kWh]')
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes.legend()
    axes.grid()
    axes.title.set_text('Nodal DLMPs')
    fig.show()
    # fig.savefig('Nodal_DLMPs_at_14.svg')

    # Figures for strategic DER offers:
    dlmps_strategic_active_power1 = dlmps_strategic.strategic_electric_grid_total_dlmp_node_active_power
    dlmps_non_strategic_active_power1 = dlmps_non_strategic.electric_grid_total_dlmp_node_active_power

    dlmp_difference = dlmps_strategic_active_power1 - dlmps_non_strategic_active_power1

    der_01_10_strategic_active_power_vector = results_strategic.der_active_power_vector_per_unit[
        ('flexible_generator', 'pv_b10_strategic')]
    der_01_10_non_strategic_active_power_vector = results_non_strategic.der_active_power_vector_per_unit[
        ('flexible_generator', 'pv_b10_strategic')]

    der_active_power_marginal_offers_timeseries = \
        results_non_strategic.der_active_power_marginal_offers_timeseries.loc[
        ('flexible_generator', 'pv_b10_strategic'), :] / der_model_set.der_models[
            'pv_b10_strategic'].active_power_nominal
    der_active_power_strategic_marginal_offers_timeseries = \
        dlmps_strategic.strategic_der_marginal_price_offers / der_model_set.der_models[
            'pv_b10_strategic'].active_power_nominal

    # Print DLMPs.
    # print(dlmps_non_strategic)

    # Store DLMPs as CSV.
    # dlmps_non_strategic.save(results_path)
    # dlmps_strategic.save(results_path)

    fig, ax = plt.subplots()
    der_active_power_marginal_offers_timeseries.plot(
        ax=ax,
        label='DER active power marginal cost',
        color='b',
        marker='s'
    )
    der_active_power_strategic_marginal_offers_timeseries[('flexible_generator', 'pv_b10_strategic')].plot(
        ax=ax,
        label='DER offered marginal cost',
        color='r',
        marker='^'
    )
    fig.suptitle('Offer comparison for the strategic generator')
    ax.set_ylabel('Offer price [$/MWh]')
    ax.set_xlabel('Time [h]')
    # ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('strategic_offers.svg')

    fig, ax = plt.subplots()
    der_01_10_non_strategic_active_power_vector.plot(
        ax=ax,
        label="Non-strategic",
        color='b',
        marker='s'
    )
    der_01_10_strategic_active_power_vector.plot(
        ax=ax,
        label='Strategic',
        color='r',
        marker='^'
    )
    fig.suptitle("Strategic der's active power generation")
    ax.set_ylabel('Active power dispatched [p.u.]')
    ax.set_xlabel('Time [h]')
    # ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    plt.xticks(rotation=-90, fontsize=8)
    fig.show()
    # fig.savefig('strategic_der_active_power.svg')

    # Plot Offer and power dispatch together:
    fig, ax = plt.subplots()
    der_01_10_strategic_active_power_vector.plot(
        ax=ax,
        label="Strategic DER power dispatch",
        color='b',
        marker='s'
    )
    ax2 = ax.twinx()
    der_active_power_strategic_marginal_offers_timeseries[('flexible_generator', 'pv_b10_strategic')].plot(
        ax=ax2,
        label='DER offered marginal cost',
        color='r',
        marker='^'
    )
    fig.suptitle("Strategic der's active power generation")
    ax.set_ylabel('Active power dispatched [p.u.]')
    ax2.set_ylabel('Offer price [$/MWh]')
    ax.set_xlabel('Time [h]')
    ax.legend(loc=0)
    ax2.legend(loc=0)
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    plt.xticks(rotation=-90, fontsize=8)
    fig.show()
    # fig.savefig('DER_01_10_active_power_offer.svg')

    voltage_profile_non_strategic = results_non_strategic.node_voltage_magnitude_vector_per_unit.loc[
        '2021-02-22 14:00:00']
    voltage_profile_strategic = results_strategic.node_voltage_magnitude_vector_per_unit.loc[report_time]

    fig, axes = plt.subplots(figsize=(12, 6))
    voltage_profile_non_strategic.plot(
        ax=axes,
        label='non-strategic',
        # y=(slice(None), slice(None), 3),
        color='b',
        marker='*'
    )
    voltage_profile_strategic.plot(
        ax=axes,
        label='strategic',
        # y=(slice(None), slice(None), 3),
        color='r',
        marker='*'
    )
    x = np.arange(len(voltage_profile_strategic.index))
    axes.set_xticks(x, voltage_profile_strategic.index)
    plt.xticks(rotation=-30, fontsize=8)
    fig.suptitle(f'Nodal Voltage Profile at {report_time}')
    plt.xlabel('Node name')
    fig.set_tight_layout(True)
    axes.set_ylabel('Voltage [p.u]')
    # axes[1].ticklabel_format(useOffset=False)
    axes.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('Voltage profile.svg')

    line_loading_non_strategic = results_non_strategic.branch_power_magnitude_vector_1_per_unit.loc[
        report_time]
    line_loading_strategic = results_strategic.branch_power_magnitude_vector_1_per_unit.loc[report_time]

    fig, axes = plt.subplots(figsize=(12, 6))
    line_loading_non_strategic.plot(
        ax=axes,
        label='non-strategic',
        # x=np.arange(len(line_loading_strategic.index)),
        color='b',
        marker='*'
    )
    line_loading_strategic.plot(
        ax=axes,
        label='strategic',
        y=line_loading_strategic.values,
        # x=np.arange(len(line_loading_strategic.index)),
        color='r',
        marker='*'
    )
    x = np.arange(len(line_loading_non_strategic.index))
    axes.set_xticks(x, line_loading_non_strategic.index)
    fig.suptitle(f'Line Loading at {report_time}')
    plt.xticks(rotation=-30, fontsize=8)
    plt.xlabel('Branch name')
    fig.set_tight_layout(True)
    axes.set_ylabel('Loading [p.u]')
    axes.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('line_loading.svg')

    print(1)


if __name__ == '__main__':
    main()
