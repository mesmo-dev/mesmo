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

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
