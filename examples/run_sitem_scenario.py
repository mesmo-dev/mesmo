"""Project SITEM scenario evaluation script."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re

import fledge.data_interface
import fledge.der_models
import fledge.plots
import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_district25'
    results_path = fledge.utils.get_results_path('run_sitem_baseline', scenario_name)
    plot_grid = False
    plot_detailed_grid = True

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain nominal operation problem & solution.
    problem = fledge.problems.NominalOperationProblem(scenario_name)
    problem.solve()
    results = problem.get_results()

    # Obtain additional results.
    branch_power_vector_magnitude_per_unit = (
        (np.abs(results['branch_power_vector_1']) + np.abs(results['branch_power_vector_2'])) / 2
        / problem.electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_per_unit.loc['maximum', :] = branch_power_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit = (
        np.abs(results['node_voltage_vector'])
        / np.abs(problem.electric_grid_model.node_voltage_vector_reference)
    )
    new_columns = []
    for branch_index, branch in enumerate(problem.electric_grid_model.branches):
        if branch[0] == 'transformer':
            if problem.electric_grid_model.branch_power_vector_magnitude_reference[branch_index] > 1e6:
                new_columns.extend(
                    int(np.ceil(problem.electric_grid_model.branch_power_vector_magnitude_reference[branch_index] / 1e6) - 1)
                    * [branch_power_vector_magnitude_per_unit.loc[:, branch]]
                )
    branch_power_vector_magnitude_per_unit = pd.concat([branch_power_vector_magnitude_per_unit, *new_columns], axis='columns')
    node_voltage_vector_magnitude_per_unit.loc['maximum', :] = node_voltage_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit.loc['minimum', :] = node_voltage_vector_magnitude_per_unit.min(axis='rows')
    results.update({
        'branch_power_vector_magnitude_per_unit': branch_power_vector_magnitude_per_unit,
        'node_voltage_vector_magnitude_per_unit': node_voltage_vector_magnitude_per_unit
    })

    # Remove bus chargers and re-run simulation.
    for der_name in problem.der_model_set.der_models:
        if type(problem.der_model_set.der_models[der_name]) is fledge.der_models.FixedEVChargerModel:
            if 'bus_charger' in der_name:
                problem.der_model_set.der_models[der_name].active_power_nominal_timeseries *= 0
                problem.der_model_set.der_models[der_name].active_power_nominal_timeseries *= 0
    problem.solve()
    results_1 = problem.get_results()
    branch_power_vector_magnitude_per_unit_1 = (
        (np.abs(results_1['branch_power_vector_1']) + np.abs(results_1['branch_power_vector_2'])) / 2
        / problem.electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_per_unit_1.loc['maximum', :] = branch_power_vector_magnitude_per_unit_1.max(axis='rows')
    new_columns = []
    for branch_index, branch in enumerate(problem.electric_grid_model.branches):
        if branch[0] == 'transformer':
            if problem.electric_grid_model.branch_power_vector_magnitude_reference[branch_index] > 1e6:
                new_columns.extend(
                    int(np.ceil(problem.electric_grid_model.branch_power_vector_magnitude_reference[branch_index] / 1e6) - 1)
                    * [branch_power_vector_magnitude_per_unit_1.loc[:, branch]]
                )
    branch_power_vector_magnitude_per_unit_1 = pd.concat([branch_power_vector_magnitude_per_unit_1, *new_columns], axis='columns')

    # Remove private EV chargers and re-run simulation.
    for der_name in problem.der_model_set.der_models:
        if type(problem.der_model_set.der_models[der_name]) is fledge.der_models.FixedEVChargerModel:
            problem.der_model_set.der_models[der_name].active_power_nominal_timeseries *= 0
            problem.der_model_set.der_models[der_name].active_power_nominal_timeseries *= 0
    problem.solve()
    results_2 = problem.get_results()
    branch_power_vector_magnitude_per_unit_2 = (
        (np.abs(results_2['branch_power_vector_1']) + np.abs(results_2['branch_power_vector_2'])) / 2
        / problem.electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_per_unit_2.loc['maximum', :] = branch_power_vector_magnitude_per_unit_2.max(axis='rows')
    new_columns = []
    for branch_index, branch in enumerate(problem.electric_grid_model.branches):
        if branch[0] == 'transformer':
            if problem.electric_grid_model.branch_power_vector_magnitude_reference[branch_index] > 1e6:
                new_columns.extend(
                    int(np.ceil(problem.electric_grid_model.branch_power_vector_magnitude_reference[branch_index] / 1e6) - 1)
                    * [branch_power_vector_magnitude_per_unit_2.loc[:, branch]]
                )
    branch_power_vector_magnitude_per_unit_2 = pd.concat([branch_power_vector_magnitude_per_unit_2, *new_columns], axis='columns')

    # Print results.
    print(results)

    # Store results to CSV.
    results.to_csv(results_path)

    # Obtain electric grid graph.
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)

    # Plot utilization.
    fledge.plots.plot_line_utilization(
        problem.electric_grid_model,
        electric_grid_graph,
        branch_power_vector_magnitude_per_unit.loc['maximum', :] * 100.0,
        results_path,
        value_unit='%',
        horizontal_line_value=100.0
    )
    fledge.plots.plot_transformer_utilization(
        problem.electric_grid_model,
        electric_grid_graph,
        branch_power_vector_magnitude_per_unit.loc['maximum', :] * 100.0,
        results_path,
        value_unit='%',
        horizontal_line_value=100.0
    )
    fledge.plots.plot_node_utilization(
        problem.electric_grid_model,
        electric_grid_graph,
        (node_voltage_vector_magnitude_per_unit.loc['maximum', :] - 1.0) * - 100.0,
        results_path,
        value_unit='%',
        suffix='drop',
        horizontal_line_value=5.0
    )

    # Plot utilization on grid layout.
    if plot_grid:
        fledge.plots.plot_grid_transformer_utilization(
            problem.electric_grid_model,
            electric_grid_graph,
            branch_power_vector_magnitude_per_unit * 100.0,
            results_path,
            vmin=20.0,
            vmax=120.0,
            value_unit='%',
            make_video=True
        )
    if plot_grid and plot_detailed_grid:
        fledge.plots.plot_grid_line_utilization(
            problem.electric_grid_model,
            electric_grid_graph,
            branch_power_vector_magnitude_per_unit * 100.0,
            results_path,
            vmin=20.0,
            vmax=120.0,
            value_unit='%',
            make_video=True
        )
        fledge.plots.plot_grid_node_utilization(
            problem.electric_grid_model,
            electric_grid_graph,
            (node_voltage_vector_magnitude_per_unit - 1.0) * -100.0,
            results_path,
            vmin=0.0,
            vmax=10.0,
            value_unit='%',
            suffix='drop',
            make_video=True
        )

    # More plots.
    histogram_bins = 100

    # Plot load timeseries.
    values = results['der_power_vector'].sum(axis='columns') / 1e6
    values.loc[:] = np.abs(np.real(values))
    values_1 = results_1['der_power_vector'].sum(axis='columns') / 1e6
    values_1.loc[:] = np.abs(np.real(values_1))
    values_2 = results_2['der_power_vector'].sum(axis='columns') / 1e6
    values_2.loc[:] = np.abs(np.real(values_2))
    title = 'Total demand'
    filename = 'demand_timeseries'
    y_label = 'Active power'
    value_unit = 'MW'

    plt.figure()
    plt.title(title)
    plt.fill_between(
        range(len(values.index)),
        values_2,
        label='Baseload',
        step='post'
    )
    plt.fill_between(
        range(len(values.index)),
        values_1,
        values_2,
        label='Baseload + private EV',
        step='post'
    )
    plt.fill_between(
        range(len(values.index)),
        values,
        values_1,
        label='Baseload + private EV + bus charging',
        step='post'
    )
    plt.xticks(
        range(len(values.index)),
        values.index.strftime('%H:%M:%S'),
        rotation=45,
        ha='right'
    )
    plt.ylabel(f'{y_label} [{value_unit}]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()

    # Plot line utilization timeseries.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
        :, branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'line'
        ].mean(axis='columns').drop('maximum')
    )
    title = 'Average line utilization'
    filename = 'line_utilization_timeseries'
    y_label = 'Utilization'
    value_unit = 'p.u.'

    plt.figure()
    plt.title(title)
    plt.bar(
        range(len(values.index)),
        values,
    )
    plt.xticks(
        range(len(values.index)),
        pd.to_datetime(values.index).strftime('%H:%M:%S'),
        rotation=45,
        ha='right'
    )
    plt.ylabel(f'{y_label} [{value_unit}]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()

    # Plot line utilization histogram.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
            :, branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_1 = (
        branch_power_vector_magnitude_per_unit_1.loc[
            :, branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_2 = (
        branch_power_vector_magnitude_per_unit_2.loc[
            :, branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values.loc[values > 1] = 1.0
    values_1.loc[values_1 > 1] = 1.0
    values_2.loc[values_2 > 1] = 1.0
    title = 'Lines'
    filename = 'line_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    plt.figure()
    plt.title(title)
    # plt.hist(values, histogram_bins, density=True)
    values_2 = np.histogram(values_2, bins=histogram_bins, range=(0.0, 1.0))
    plt.step(values_2[1], np.append(values_2[0], 0.0) / np.sum(values_2[0]), where='post', label='Baseload', linewidth=3.5)
    values_1 = np.histogram(values_1, bins=histogram_bins, range=(0.0, 1.0))
    plt.step(values_1[1], np.append(values_1[0], 0.0) / np.sum(values_1[0]), where='post', label='Baseload + private EV', linewidth=2.5)
    values = np.histogram(values, bins=histogram_bins, range=(0.0, 1.0))
    plt.step(values[1], np.append(values[0], 0.0) / np.sum(values[0]), where='post', label='Baseload + private EV + bus charging', linewidth=1.5)
    plt.ylim(0, 1.05 * np.max(values_2[0] / np.sum(values[0])))
    plt.ylabel('Frequency')
    plt.xlim([-0.01, 1.0])
    plt.xlabel(f'{y_label} [{value_unit}]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()

    # Plot line utilization cumulative.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
            :, branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_1 = (
        branch_power_vector_magnitude_per_unit_1.loc[
        :, branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_2 = (
        branch_power_vector_magnitude_per_unit_2.loc[
        :, branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values.loc[values > 1] = 1.0
    values_1.loc[values_1 > 1] = 1.0
    values_2.loc[values_2 > 1] = 1.0
    title = 'Lines'
    filename = 'line_utilization_cumulative'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    plt.figure()
    plt.title(title)
    plt.hist(values_2, histogram_bins, range=(0.0, 1.01), density=True, cumulative=True, histtype='step', label='Baseload')
    plt.hist(values_1, histogram_bins, range=(0.0, 1.01), density=True, cumulative=True, histtype='step', label='Baseload + private EV')
    plt.hist(values, histogram_bins, range=(0.0, 1.01), density=True, cumulative=True, histtype='step', label='Baseload + private EV + bus charging')
    plt.axhline(0.9, color='black', linewidth=1)
    plt.ylabel('Cumulative proportion')
    plt.xlim([-0.01, 1.0])
    plt.xlabel(f'{y_label} [{value_unit}]')
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()

    # Plot transformer utilization histogram.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
            :,
            (
                (branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'transformer')
                & branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_name').str.contains('22kV')
            )
        ].max()
    )
    values_1 = (
        branch_power_vector_magnitude_per_unit_1.loc[
            :,
            (
                (branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_type') == 'transformer')
                & branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_name').str.contains('22kV')
            )
        ].max()
    )
    values_2 = (
        branch_power_vector_magnitude_per_unit_2.loc[
            :,
            (
                (branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_type') == 'transformer')
                & branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_name').str.contains('22kV')
            )
        ].max()
    )
    values.loc[values > 1] = 1.0
    values_1.loc[values_1 > 1] = 1.0
    values_2.loc[values_2 > 1] = 1.0
    title = '1MVA Transformers'
    filename = 'transformer_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    plt.figure()
    plt.title(title)
    # plt.hist(values, histogram_bins, density=True)
    values_2 = np.histogram(values_2, bins=histogram_bins, range=(0.0, 1.0))
    plt.step(values_2[1], np.append(values_2[0], 0.0) / np.sum(values_2[0]), where='post', label='Baseload', linewidth=3.5)
    values_1 = np.histogram(values_1, bins=histogram_bins, range=(0.0, 1.0))
    plt.step(values_1[1], np.append(values_1[0], 0.0) / np.sum(values_1[0]), where='post', label='Baseload + private EV', linewidth=2.5)
    values = np.histogram(values, bins=histogram_bins, range=(0.0, 1.0))
    plt.step(values[1], np.append(values[0], 0.0) / np.sum(values[0]), where='post', label='Baseload + private EV + bus charging', linewidth=1.5)
    plt.ylim(0, 1.05 * np.max(values_2[0] / np.sum(values[0])))
    plt.ylabel('Frequency')
    plt.xlim([-0.01, 1.0])
    plt.xlabel(f'{y_label} [{value_unit}]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()

    # Plot transformer utilization cumulative.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
            :,
            (
                (branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'transformer')
                & branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_name').str.contains('22kV')
            )
        ].max()
    )
    values_1 = (
        branch_power_vector_magnitude_per_unit_1.loc[
            :,
            (
                (branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_type') == 'transformer')
                & branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_name').str.contains('22kV')
            )
        ].max()
    )
    values_2 = (
        branch_power_vector_magnitude_per_unit_2.loc[
            :,
            (
                (branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_type') == 'transformer')
                & branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_name').str.contains('22kV')
            )
        ].max()
    )
    values.loc[values > 1] = 1.0
    values_1.loc[values_1 > 1] = 1.0
    values_2.loc[values_2 > 1] = 1.0
    title = '1MVA Transformers'
    filename = 'transformer_utilization_cumulative'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    plt.figure()
    plt.title(title)
    plt.hist(values_2, histogram_bins, range=(0.0, 1.01), density=True, cumulative=True, histtype='step', label='Baseload')
    plt.hist(values_1, histogram_bins, range=(0.0, 1.01), density=True, cumulative=True, histtype='step', label='Baseload + private EV')
    plt.hist(values, histogram_bins, range=(0.0, 1.01), density=True, cumulative=True, histtype='step', label='Baseload + private EV + bus charging')
    plt.axhline(0.9, color='black', linewidth=1)
    plt.ylabel('Cumulative proportion')
    plt.xlim([-0.01, 1.0])
    plt.xlabel(f'{y_label} [{value_unit}]')
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
