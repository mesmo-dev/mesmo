"""Project SITEM scenario evaluation script."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

import fledge.config
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
    histogram_bins = np.arange(0, 1.01, 0.01)

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

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='Baseload',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Baseload + private EV',
        fill='tonexty',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values.index,
        y=values.values,
        name='Baseload + private EV + bus charging',
        fill='tonexty',
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

    # Plot line utilization timeseries.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
            :, branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'line'
        ].mean(axis='columns').drop('maximum')
    )
    values_1 = (
        branch_power_vector_magnitude_per_unit_1.loc[
            :, branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_type') == 'line'
        ].mean(axis='columns').drop('maximum')
    )
    values_2 = (
        branch_power_vector_magnitude_per_unit_2.loc[
            :, branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_type') == 'line'
        ].mean(axis='columns').drop('maximum')
    )
    title = 'Average line utilization'
    filename = 'line_utilization_timeseries'
    y_label = 'Utilization'
    value_unit = 'p.u.'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='Baseload',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Baseload + private EV',
        fill='tonexty',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values.index,
        y=values.values,
        name='Baseload + private EV + bus charging',
        fill='tonexty',
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

    # Plot line utilization histogram.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
            :, branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values.loc[values > 1] = 1.0
    values = (
        pd.Series([*np.histogram(values, bins=histogram_bins)[0], 0], index=histogram_bins) / len(values)
    )
    values_1 = (
        branch_power_vector_magnitude_per_unit_1.loc[
            :, branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_1.loc[values_1 > 1] = 1.0
    values_1 = (
        pd.Series([*np.histogram(values_1, bins=histogram_bins)[0], 0], index=histogram_bins) / len(values_1)
    )
    values_2 = (
        branch_power_vector_magnitude_per_unit_2.loc[
            :, branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_2.loc[values_2 > 1] = 1.0
    values_2 = (
        pd.Series([*np.histogram(values_2, bins=histogram_bins)[0], 0], index=histogram_bins) / len(values_2)
    )
    values.loc[values > 1] = 1.0
    values_1.loc[values_1 > 1] = 1.0
    values_2.loc[values_2 > 1] = 1.0
    title = 'Lines'
    filename = 'line_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=values_2.index,
        y=values_2.values,
        name='Baseload'
    ))
    figure.add_trace(go.Bar(
        x=values_1.index,
        y=values_1.values,
        name='Baseload + private EV'
    ))
    figure.add_trace(go.Bar(
        x=values.index,
        y=values.values,
        name='Baseload + private EV + bus charging'
    ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Frequency',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

    # Plot line utilization cumulative.
    values = (
        branch_power_vector_magnitude_per_unit.loc[
            :, branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values.loc[values > 1] = 1.0
    values = (
        pd.Series([*np.histogram(values, bins=histogram_bins)[0], 0], index=histogram_bins).cumsum() / len(values)
    )
    values_1 = (
        branch_power_vector_magnitude_per_unit_1.loc[
            :, branch_power_vector_magnitude_per_unit_1.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_1.loc[values_1 > 1] = 1.0
    values_1 = (
        pd.Series([*np.histogram(values_1, bins=histogram_bins)[0], 0], index=histogram_bins).cumsum() / len(values_1)
    )
    values_2 = (
        branch_power_vector_magnitude_per_unit_2.loc[
            :, branch_power_vector_magnitude_per_unit_2.columns.get_level_values('branch_type') == 'line'
        ].max()
    )
    values_2.loc[values_2 > 1] = 1.0
    values_2 = (
        pd.Series([*np.histogram(values_2, bins=histogram_bins)[0], 0], index=histogram_bins).cumsum() / len(values_2)
    )
    title = 'Lines'
    filename = 'line_utilization_cumulative'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='Baseload',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Baseload + private EV',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values.index,
        y=values.values,
        name='Baseload + private EV + bus charging',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_shape(go.layout.Shape(
        x0=0,
        x1=1,
        xref='paper',
        y0=0.9,
        y1=0.9,
        yref='y',
        type='line',
        line=go.layout.shape.Line(width=2)
    ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Cumulative proportion',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

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
    values.loc[values > 1] = 1.0
    values = (
        pd.Series([*np.histogram(values, bins=histogram_bins)[0], 0], index=histogram_bins) / len(values)
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
    values_1.loc[values_1 > 1] = 1.0
    values_1 = (
        pd.Series([*np.histogram(values_1, bins=histogram_bins)[0], 0], index=histogram_bins) / len(values_1)
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
    values_2.loc[values_2 > 1] = 1.0
    values_2 = (
        pd.Series([*np.histogram(values_2, bins=histogram_bins)[0], 0], index=histogram_bins) / len(values_2)
    )
    title = '1MVA Transformers'
    filename = 'transformer_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=values_2.index,
        y=values_2.values,
        name='Baseload'
    ))
    figure.add_trace(go.Bar(
        x=values_1.index,
        y=values_1.values,
        name='Baseload + private EV'
    ))
    figure.add_trace(go.Bar(
        x=values.index,
        y=values.values,
        name='Baseload + private EV + bus charging'
    ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Frequency',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

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
    values.loc[values > 1] = 1.0
    values = (
        pd.Series([*np.histogram(values, bins=histogram_bins)[0], 0], index=histogram_bins).cumsum() / len(values)
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
    values_1.loc[values_1 > 1] = 1.0
    values_1 = (
        pd.Series([*np.histogram(values_1, bins=histogram_bins)[0], 0], index=histogram_bins).cumsum() / len(values_1)
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
    values_2.loc[values_2 > 1] = 1.0
    values_2 = (
        pd.Series([*np.histogram(values_2, bins=histogram_bins)[0], 0], index=histogram_bins).cumsum() / len(values_2)
    )
    title = '1MVA Transformers'
    filename = 'transformer_utilization_cumulative'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='Baseload',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Baseload + private EV',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values.index,
        y=values.values,
        name='Baseload + private EV + bus charging',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_shape(go.layout.Shape(
        x0=0,
        x1=1,
        xref='paper',
        y0=0.9,
        y1=0.9,
        yref='y',
        type='line',
        line=go.layout.shape.Line(width=2)
    ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Cumulative proportion',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + '.png'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
