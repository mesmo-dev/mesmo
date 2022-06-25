"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
pd.options.plotting.backend = "matplotlib"

from mesmo.kkt_conditions_with_state_space import StrategicMarket
import mesmo


def main():
    # TODO: Currently not working. Review limits below.

    # scenarios = [None]
    # scenario_name = "strategic_dso_market"
    # global strategic_der_model_set
    # scenario_name = 'strategic_market_19_node'
    scenario_name = 'ieee_34node'
    strategic_scenario = True
    kkt_conditions = True
    strategic_der_name = 'pv_860_strategic'

    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    scenario_profiles = mesmo.data_interface.DERData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name)
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

    # max_branch_power = np.array([0.808, 0.784, 0.726, 0.532,
    #                              0.926, 0.803, 0.810, 0.708, 0.708,
    #                              0.789, 0.789, 0.789, 0.789, 0.789, 0.789,
    #                              0.538, 0.538, 0.538, 0.538])
    max_branch_power = 1
    max_branch_power = pd.Series(1.0, index=electric_grid_model.branches)
    max_branch_power['transformer'] = 10

    # Define electric grid problem.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.9 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.1 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = max_branch_power.values * electric_grid_model.branch_power_vector_magnitude_reference

    jump_here = True
    grid_cost_coefficient = 1
    report_time = '2019-07-17 16:00:00'

    der_model_set.define_optimization_problem(optimization_non_strategic,
                                              price_data,
                                              grid_cost_coefficient=grid_cost_coefficient,
                                              kkt_conditions=False,
                                              state_space_model=True
                                              )

    linear_electric_grid_model_set.define_optimization_problem(
        optimization_non_strategic,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        grid_cost_coefficient=grid_cost_coefficient,
        kkt_conditions=False
    )

    if strategic_scenario:
        optimization_strategic = mesmo.utils.OptimizationProblem()

        der_model_set.define_optimization_problem(optimization_strategic,
                                                  price_data,
                                                  kkt_conditions=False,
                                                  grid_cost_coefficient=grid_cost_coefficient,
                                                  state_space_model=True
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

        # strategic_der_model_set = StrategicMarket(scenario_name, strategic_der=strategic_der_name)
        # strategic_der_model_set.strategic_optimization_problem(
        #     optimization_strategic,
        #     price_data,
        #     node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        #     node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        #     branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
        #     big_m=1e5,
        #     kkt_conditions=True,
        #     grid_cost_coefficient=grid_cost_coefficient
        # )

    # Define DER problem.

    # Solve centralized optimization problem.
    optimization_non_strategic.solve()
    optimization_strategic.solve()

    # Obtain results.
    flexible_der_type = ['flexible_generator', 'flexible_load']
    time_index = [scenario_data.timesteps]
    price_correction = 1e3/scenario_data.scenario.at['base_apparent_power']


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
    dlmps_strategic = linear_electric_grid_model_set.get_optimization_dlmps(optimization_strategic, price_data)
    dlmp_difference = dlmps_strategic.electric_grid_total_dlmp_node_active_power - \
                      dlmps_non_strategic.electric_grid_total_dlmp_node_active_power

    flexible_der_active_power_non_strategic = results_non_strategic.der_active_power_vector_per_unit[flexible_der_type]
    flexible_der_active_power_strategic = results_strategic.der_active_power_vector_per_unit[flexible_der_type]

    flexible_der_reactive_power_non_strategic = results_non_strategic.der_reactive_power_vector_per_unit[
        flexible_der_type]
    flexible_der_reactive_power_strategic = results_strategic.der_reactive_power_vector_per_unit[flexible_der_type]

    # =========================================================
    load_profile = 100 * scenario_profiles.der_definitions[('schedule_per_unit', 'mixed_commercial_residential')]
    pv_profile = 100 * scenario_profiles.der_definitions[('schedule_per_unit', 'photovoltaic_generic')]

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(8, 8))
    load_profile.plot.area(
        ax=axes[0],
        label='Mixed-commercial residential load profile',
        color='b'
        # marker='s'
    )
    pv_profile.plot.area(
        ax=axes[1],
        label='Generic PV profile',
        color='g'
        # marker='^'
    )
    # x = np.arange(len(load_profile.index))
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(load_profile.index,
    #                             rotation=-30, fontsize=8, minor=False)
    axes[0].title.set_text('Mixed-commercial residential load profile')
    axes[0].title.set_fontsize(18)
    axes[1].title.set_text('Generic PV profile')
    axes[1].title.set_fontsize(18)
    axes[1].set_xlabel('Time [h]', fontsize=18)
    # fig.suptitle(f'Nodal Voltage Profile at {sample_time}')
    # axes[i - 1].set_ylim([0.5, 1.05])
    axes[0].set_ylabel('%', fontsize=18)
    axes[1].set_ylabel('%', fontsize=18)
    # axes[0].legend()
    axes[0].grid(axis='y')
    axes[0].grid(axis='x')
    axes[1].grid(axis='y')
    axes[1].grid(axis='x')
    axes[0].tick_params(axis='both', which='minor', labelsize=14)
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[1].tick_params(axis='both', which='minor', labelsize=14)
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    fig.set_tight_layout(True)
    # fig.suptitle('Mixed-commercial residential load and Generic PV profile', fontsize=18)
    # fig.show()
    # fig.savefig('results kkt/load_and_pv_profiles.svg')

    gsp_price_timeseries = price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')] \
                           * 1e3/ scenario_data.scenario.at['base_apparent_power']

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 3))
    gsp_price_timeseries.plot.area(
        ax=axes,
        # label='Energy price at GSP',
        color='g'
        # marker='s'
    )
    # x = np.arange(len(load_profile.index))
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(load_profile.index,
    #                             rotation=-30, fontsize=8, minor=False)
    axes.title.set_text('Energy price at GSP')
    axes.title.set_fontsize(14)
    axes.set_xlabel('Time [h]', fontsize=12)
    # fiuptitle(f'Nodal Voltage Profile at {sample_time}')
    # axi - 1].set_ylim([0.5, 1.05])
    axes.set_ylabel(r"$c_t^{0,p}$  [\$/kWh]", fontsize=12)
    # ax0].legend()
    axes.grid(axis='y')
    axes.grid(axis='x')
    axes.tick_params(axis='both', which='minor', labelsize=10)
    axes.tick_params(axis='both', which='major', labelsize=10)
    fig.set_tight_layout(True)
    # fig.suptitle('Mixed-commercial residential load and Generic PV profile', fontsize=18)
    # fig.show()
    # fig.savefig('results kkt/energy_price_at_gsp.svg')



    primal_active_losses = results_non_strategic.loss_active
    kkt_active_losses = results_strategic.loss_active
    fig, ax = plt.subplots(figsize=(8,4))
    primal_active_losses.plot(
        ax=ax,
        label='Primal active loss',
        color='b',
        marker='s'
    )
    kkt_active_losses.plot(
        ax=ax,
        label='KKT active loss',
        color='r',
        marker='^'
    )
    fig.suptitle('Active losses primal problem vs KKT problem', fontsize=18)
    x = np.arange(len(kkt_active_losses.index))
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=10, minor=False)
    ax.set_ylabel('Losses [p.u.]', fontsize=16)
    ax.set_xlabel('Time [h]', fontsize=16)
    # axes.tick_params(axis='both', which='major', labelsize=14)
    # axes.tick_params(axis='both', which='minor', labelsize=14)
    plt.legend()
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('results kkt/kkt_active_losses.svg')


    x = np.arange(len(flexible_der_active_power_non_strategic.columns))
    width = 0.35
    fig, axes = plt.subplots(1, figsize=(12, 6))
    axes.bar(x + width / 3,
             flexible_der_active_power_non_strategic.loc[report_time],
             width=width,
             color='b',
             label='DSO primal problem')
    axes.bar(x - width / 3,
             flexible_der_active_power_strategic.loc[report_time],
             width=width,
             color='r',
             label='DSO KKT problem')
    axes.set_xticks(x, flexible_der_active_power_strategic.columns)
    plt.xlabel('DER name', fontsize=18)
    axes.set_ylabel('Power dispatch [p.u]', fontsize=18)
    axes.title.set_text(f"Flexible DER's active power dispatch at {report_time}")
    axes.title.set_fontsize(18)
    fig.set_tight_layout(True)
    plt.xticks(rotation=-90, fontsize=8)
    plt.yticks(fontsize=10)
    axes.legend()
    axes.grid()
    fig.show()
    # fig.savefig('results kkt/kkt_flexible_der_active_power_dispatch_report_time.svg')

    # DLMPs in non-strategic scenario Timeseries for Node 10 for three phases
    # Energy portion of DLMP:

    node_860_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # Loss portion of DLMP:
    node_860_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # Voltage portion of DLMP:
    node_860_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # Congestion portion of DLMP:
    node_860_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))]

    # TODO plotting the results
    # Total DLMP:
    node_860_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # ------------------DLMPs in Strategic scenario Timeseries for Node 10 for three phases------------------------:
    # Energy portion of DLMP:
    node_860_energy_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_energy_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # Loss portion of DLMP:
    node_860_loss_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_loss_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # Voltage portion of DLMP:
    node_860_voltage_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_voltage_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # Congestion portion of DLMP:
    node_860_congestion_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_congestion_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    # TODO plotting the results
    # Total DLMP:
    node_860_total_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_total_dlmp_node_active_power.loc[:, (slice(None), '860', slice(None))] * price_correction

    phases = [1, 2, 3]
    fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(12, 12))
    for i in phases:
        node_860_total_dlmps_non_strategic_active_power[('no_source', '860', i)].plot(
            ax=axes[i-1],
            label=f'Primal_phase_{i}',
            color='b',
            marker='s'
        )
        node_860_total_dlmps_strategic_active_power[('no_source', '860', i)].plot(
            ax=axes[i-1],
            label=f'KKT_phase_{i}',
            color='r',
            marker='^'
        )
        # x = np.arange(len(node_860_total_dlmps_strategic_active_power.index))
        # axes[i - 1].set_xticks(x)
        # axes[i - 1].set_xticklabels(time_index,
        #                             rotation=-30, fontsize=8, minor=False)
        axes[i - 1].title.set_text(f'Node 860 phase {i} total DLMP')
        axes[i - 1].title.set_fontsize(18)
        axes[i - 1].set_xlabel('Time [h]', fontsize=18)
        # fig.suptitle(f'Nodal Voltage Profile at {sample_time}')
        # axes[i - 1].set_ylim([0.5, 1.05])
        axes[i - 1].set_ylabel('DLMP [$/kWh]', fontsize=18)
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
        fig.set_tight_layout(True)
    fig.suptitle('DLMP timeseries at strategic node 860', fontsize=20)
    fig.show()
    # fig.savefig('results kkt/kkt_DLMP_time_series_at_strategic_node_860.svg')


    # fig.savefig('dlmp_timeseries_node_10.svg')

    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_energy_dlmp_node_active_power.loc[report_time] * price_correction

    # Loss portion of DLMPs:
    nodal_loss_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_loss_dlmp_node_active_power.loc[report_time] * price_correction

    # Voltage portion of DLMPs:
    nodal_voltage_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_voltage_dlmp_node_active_power.loc[report_time] * price_correction

    # Congestion portion of DLMPs:
    nodal_congestion_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_congestion_dlmp_node_active_power.loc[report_time] * price_correction

    # Total DLMPs:
    nodal_total_dlmps_non_strategic_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.loc[report_time] * price_correction

    # ______________________Nodal DLMPs at time 15:00:00 for non strategic scenario:_________________________________
    nodal_energy_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_energy_dlmp_node_active_power.loc[report_time] * price_correction

    # Loss portion of DLMP:
    nodal_loss_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_loss_dlmp_node_active_power.loc[report_time] * price_correction

    # Voltage portion of DLMP:
    nodal_voltage_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_voltage_dlmp_node_active_power.loc[report_time] * price_correction

    # Congestion portion of DLMP:
    nodal_congestion_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_congestion_dlmp_node_active_power.loc[report_time] * price_correction

    # Total DLMP:
    nodal_total_dlmps_strategic_active_power = \
        dlmps_strategic.electric_grid_total_dlmp_node_active_power.loc[report_time] * price_correction

    for i in phases:
        x = np.arange(len(nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)].index))
        fig, axes = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
        axes[0].bar(x, nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)]
                    , label='Energy', color='g')
        axes[1].bar(x, nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)]
                    , label='Energy', color='g')
        axes[0].bar(x, nodal_loss_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Loss',
                    color='b')
        axes[1].bar(x, nodal_loss_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Loss',
                    color='b')
        axes[0].bar(x, nodal_voltage_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Voltage',
                    color='r')
        axes[1].bar(x, nodal_voltage_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Voltage',
                    color='r')
        axes[0].bar(x, nodal_congestion_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_voltage_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Congestion',
                    color='y')
        axes[1].bar(x, nodal_congestion_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    bottom=nodal_energy_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_loss_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)] +
                           nodal_voltage_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                    label='Congestion',
                    color='y')
        axes[0].plot(x, nodal_total_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)],
                     color='k', label='Total', marker='*')
        axes[1].plot(x, nodal_total_dlmps_strategic_active_power.loc[(slice(None), slice(None), i)],
                     color='k', label='Total', marker='*')
        axes[1].set_xticks(x, nodal_energy_dlmps_non_strategic_active_power.loc[(slice(None), slice(None), i)].index)
        plt.xlabel('Node name')
        fig.set_tight_layout(True)
        axes[0].set_ylabel('DLMP [$/MW]', fontsize=18)
        axes[1].set_ylabel('DLMP [$/MW]', fontsize=18)
        axes[0].set_ylim([0., 0.3])
        axes[1].set_ylim([0., 0.3])
        axes[0].title.set_text(f"DSO primal problem nodal DLMPs at {report_time} for phase {i}")
        axes[0].title.set_fontsize(18)
        axes[1].title.set_text(f"DSO KKT problem nodal DLMPs at {report_time} for phase {i}")
        axes[1].title.set_fontsize(18)
        plt.xticks(rotation=-90, fontsize=10)
        plt.yticks(fontsize=10)
        axes[0].legend()
        axes[1].legend()
        axes[0].grid()
        axes[1].grid()
        # axes[0].set_ylim([0, 0.2])
        # axes[1].set_ylim([0, 0.2])
        fig.show()
        # fig.savefig('results kkt/kkt_contributions_to_DLMP.svg')


    fig, axes = plt.subplots(3, sharex=False, sharey=True, figsize=(12, 12))
    for i in phases:
        nodal_total_dlmps_non_strategic_active_power[(slice(None), slice(None), i)].plot(
            ax=axes[i - 1],
            label=f'Primal phase {i}',
            color='b',
            marker='s'
        )
        nodal_total_dlmps_non_strategic_active_power[(slice(None), slice(None), i)].plot(
            ax=axes[i - 1],
            label=f'KKT phase {i}',
            color='r',
            marker='^'
        )
        x = np.arange(len(nodal_total_dlmps_non_strategic_active_power[(slice(None), slice(None), i)].index))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(nodal_total_dlmps_non_strategic_active_power[(slice(None), slice(None), i)].index,
                                    rotation=-90, fontsize=10, minor=False)
        plt.yticks(fontsize=10)
        axes[i - 1].title.set_text(f'Nodal DLMPs at time {report_time} for phase {i} DSO primal vs KKT')
        axes[i - 1].set_xlabel('Node name')
        # fig.suptitle(f'Nodal Voltage Profile at {sample_time}')
        # axes[i - 1].set_ylim([0.5, 1.05])
        axes[i - 1].set_ylabel('DLMP [$/kWh]')
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
        fig.set_tight_layout(True)
    fig.suptitle(f'Nodal DLMPs at {report_time}')
    fig.show()
    # fig.savefig('results kkt/kkt_nodal_dlmp_at_18_pm.svg')


    # Figures for strategic DER offers:

    strategic_der_active_power_vector_strategic_scenario = results_strategic.der_active_power_vector_per_unit[
        ('flexible_generator', strategic_der_name)]
    strategic_der_active_power_vector_non_strategic_scenario = results_non_strategic.der_active_power_vector_per_unit[
        ('flexible_generator', strategic_der_name)]

    der_active_power_marginal_offers_timeseries = \
        results_non_strategic.der_active_power_marginal_offers_timeseries.loc[
        ('flexible_generator', strategic_der_name), :] / der_model_set.der_models[
            strategic_der_name].active_power_nominal * price_correction
    der_active_power_strategic_marginal_offers_timeseries = \
        results_strategic.der_active_power_marginal_offers_timeseries.loc[
        ('flexible_generator', strategic_der_name), :] / der_model_set.der_models[
            strategic_der_name].active_power_nominal * price_correction


    fig, ax = plt.subplots()
    der_active_power_marginal_offers_timeseries.plot(
        ax=ax,
        label='DER active power marginal cost',
        color='b',
        marker='s'
    )
    der_active_power_strategic_marginal_offers_timeseries.plot(
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
    # fig.show()
    # fig.savefig('strategic_offers.svg')

    fig, ax = plt.subplots()
    strategic_der_active_power_vector_non_strategic_scenario.plot(
        ax=ax,
        label="DSO primal",
        color='b',
        marker='s'
    )
    strategic_der_active_power_vector_strategic_scenario.plot(
        ax=ax,
        label='DSO KKT',
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
    fig, ax = plt.subplots(figsize=(8,6))
    strategic_der_active_power_vector_strategic_scenario.plot(
        ax=ax,
        label="Strategic DER power dispatch",
        color='b',
        marker='s'
    )
    ax2 = ax.twinx()
    der_active_power_strategic_marginal_offers_timeseries.plot(
        ax=ax2,
        label='DER marginal offer',
        color='r',
        marker='^'
    )
    fig.suptitle("Strategic der's active power generation", fontsize=18)
    ax.set_ylabel('Active power dispatched [p.u.]', fontsize=18)
    ax2.set_ylabel('Marginal cost [$/MWh]', fontsize=18)
    ax.set_xlabel('Time [h]', fontsize=18)
    ax.legend(loc=0)
    ax2.legend(loc=0)
    plt.grid(axis='y')
    plt.grid(axis='x')
    # ax2.tick_params(axis='both', which='major', labelsize=14)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    fig.set_tight_layout(True)
    plt.xticks(rotation=-90, fontsize=12)
    fig.show()
    # fig.savefig('results kkt/kkt_DER_890_active_power_offer.svg')

    voltage_profile_non_strategic = results_non_strategic.node_voltage_magnitude_vector_per_unit.min()
    voltage_profile_strategic = results_strategic.node_voltage_magnitude_vector_per_unit.min()

    fig, axes = plt.subplots(3, figsize=(12, 12))
    for i in [1, 2, 3]:
        voltage_profile_non_strategic[:, :, i].plot(
            ax=axes[i - 1],
            label=f'DSO primal min voltage profile of phase {i}',
            # y=(slice(None), slice(None), 3),
            color='b',
            marker='s'
        )
        voltage_profile_strategic[:, :, i].plot(
            ax=axes[i - 1],
            label=f'DSO kkt min voltage profile of phase {i}',
            # y=(slice(None), slice(None), 3),
            color='r',
            marker='^'
        )
        x = np.arange(len(voltage_profile_non_strategic[:, :, i].index))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(voltage_profile_non_strategic[:, :, i].index, rotation=-90, fontsize=10, minor=False)
        plt.yticks(fontsize=10)
        axes[i - 1].set_ylim([0.75, 1.1])
        axes[i - 1].set_ylabel('Voltage [p.u]', fontsize=18)
        axes[i - 1].set_xlabel('Node name', fontsize=18)
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
    fig.set_tight_layout(True)
    fig.set_tight_layout(True)
    fig.suptitle('Minimum Nodal Voltage Profile', fontsize=20)
    fig.show()
    # fig.savefig('results kkt/kkt_min_voltage_profile.svg')

    line_loading_non_strategic1 = results_non_strategic.branch_power_magnitude_vector_1_per_unit.max()
    line_loading_strategic1 = results_strategic.branch_power_magnitude_vector_1_per_unit.max()
    fig, axes = plt.subplots(3, figsize=(12, 12))
    for i in [1, 2, 3]:
        line_loading_non_strategic1.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max line loading of phase {i} for DSO primal',
            color='b',
            marker='s'
        )
        line_loading_strategic1.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max line loading of phase {i} for DSO kkt',
            color='r',
            marker='^'
        )
        x = np.arange(len(line_loading_non_strategic1[:, :, i].index[:-2]))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(line_loading_non_strategic1[:, :, i].index[:-2], rotation=-90, fontsize=10, minor=False)
        # axes[i-1].set_ylim([0, 7])
        plt.yticks(fontsize=10)
        axes[i - 1].set_ylabel('Loading [p.u]', fontsize=18)
        axes[i - 1].set_xlabel('Line name', fontsize=18)
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
    fig.set_tight_layout(True)
    fig.suptitle('Maximum Line  loading in "From" direction', fontsize=20)
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('results kkt/kkt_line_loading_from.svg')

    line_loading_non_strategic2 = results_non_strategic.branch_power_magnitude_vector_2_per_unit.max()
    line_loading_strategic2 = results_strategic.branch_power_magnitude_vector_2_per_unit.max()
    fig, axes = plt.subplots(3, figsize=(12, 12))
    for i in [1, 2, 3]:
        line_loading_non_strategic2.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max line loading of phase {i} for DSO primal',
            color='b',
            marker='s'
        )
        line_loading_strategic2.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max line loading of phase {i} for DSO kkt',
            color='r',
            marker='^'
        )
        x = np.arange(len(line_loading_non_strategic2[:, :, i].index[:-2]))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(line_loading_non_strategic2[:, :, i].index[:-2], rotation=-90, fontsize=10,
                                    minor=False)
        # axes[i-1].set_ylim([0, 7])
        plt.yticks(fontsize=10)
        axes[i - 1].set_ylabel('Loading [p.u]', fontsize=18)
        axes[i - 1].set_xlabel('Line name', fontsize=18)
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
    fig.set_tight_layout(True)
    fig.suptitle('Maximum Line  loading in "To" direction', fontsize=20)
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('results kkt/kkt_line_loading_to.svg')

    # ******************************************
    dlmps_non_strategic.electric_grid_total_dlmp_node_active_power.index = np.arange(25)
    dlmps_non_strategic.electric_grid_total_dlmp_node_active_power = \
        dlmps_non_strategic.electric_grid_total_dlmp_node_active_power * 1e3 / scenario_data.scenario.at['base_apparent_power']
    dlmps_strategic.electric_grid_total_dlmp_node_active_power.index = np.arange(25)
    dlmps_strategic.electric_grid_total_dlmp_node_active_power = \
        dlmps_strategic.electric_grid_total_dlmp_node_active_power * 1e3/ scenario_data.scenario.at['base_apparent_power']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    axes = sns.heatmap(dlmps_non_strategic.electric_grid_total_dlmp_node_active_power, annot=False, cmap="YlGnBu")
    axes.set_ylabel('Time [h]', fontsize=20)
    axes.set_xlabel('Node', fontsize=20)
    axes.collections[0].colorbar.set_label("DLMP level [$/kWh]", fontsize=20)
    fig.suptitle('Nodal DLMPs over the time horizon in non-strategic scenario', fontsize=30)
    axes.tick_params(axis='both', which='minor', labelsize=14)
    axes.tick_params(axis='both', which='major', labelsize=14)
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('results kkt/heatmap_DLMP nodal_timeseries_non_strategic.svg')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    axes = sns.heatmap(dlmps_strategic.electric_grid_total_dlmp_node_active_power, annot=False, cmap="YlGnBu")
    axes.set_ylabel('Time [h]', fontsize=20)
    axes.set_xlabel('Node', fontsize=20)
    axes.collections[0].colorbar.set_label("DLMP level [$/kWh]", fontsize=20)
    fig.suptitle('Nodal DLMPs over the time horizon in non-strategic scenario', fontsize=30)
    axes.tick_params(axis='both', which='minor', labelsize=14)
    axes.tick_params(axis='both', which='major', labelsize=14)
    fig.set_tight_layout(True)
    fig.show()
    # fig.savefig('results kkt/heatmap_DLMP_nodal_timeseries_strategic.svg')



    print(1)


if __name__ == '__main__':
    main()
