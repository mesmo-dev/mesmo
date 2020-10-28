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

    # Plot demand timeseries.
    values_dict = {
        'Baseload + private EV + bus charging': results['der_power_vector'],
        'Baseload + private EV': results_1['der_power_vector'],
        'Baseload': results_2['der_power_vector']
    }

    # Pre-process values.
    for key in values_dict:
        values_dict[key] = values_dict[key].sum(axis='columns') / 1e6
        values_dict[key].loc[:] = np.abs(np.real(values_dict[key]))

    # Obtain plot title / labels / filename.
    title = 'Total demand'
    filename = 'total_demand_timeseries'
    y_label = 'Active power'
    value_unit = 'MW'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Scatter(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
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
    figure.write_image(os.path.join(results_path, filename + f".{fledge.config.config['plots']['file_format']}"))

    # Plot line utilization histogram.
    values_dict = {
        'Baseload': branch_power_vector_magnitude_per_unit_2,
        'Baseload + private EV': branch_power_vector_magnitude_per_unit_1,
        'Baseload + private EV + bus charging': branch_power_vector_magnitude_per_unit
    }

    # Pre-process values.
    for key in values_dict:
        # Obtain maximum utilization for all lines.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'line'].max()
        )
        # Set over-utilized lines to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > 1] = 1.0
        # Obtain histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins)
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = 'Lines'
    filename = 'line_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Bar(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key
        ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Frequency',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + f".{fledge.config.config['plots']['file_format']}"))

    # Plot line utilization cumulative.
    values_dict = {
        'Baseload': branch_power_vector_magnitude_per_unit_2,
        'Baseload + private EV': branch_power_vector_magnitude_per_unit_1,
        'Baseload + private EV + bus charging': branch_power_vector_magnitude_per_unit
    }

    # Pre-process values.
    for key in values_dict:
        # Obtain maximum utilization for all lines.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'line'].max()
        )
        # Set over-utilized lines to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > 1] = 1.0
        # Obtain cumulative histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins).cumsum()
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = 'Lines'
    filename = 'line_utilization_cumulative'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Scatter(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
            line=go.scatter.Line(shape='hv')
        ))
    # Add horizontal line at 90%.
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
    figure.write_image(os.path.join(results_path, filename + f".{fledge.config.config['plots']['file_format']}"))

    # Plot transformer utilization histogram.
    values_dict = {
        'Baseload': branch_power_vector_magnitude_per_unit_2,
        'Baseload + private EV': branch_power_vector_magnitude_per_unit_1,
        'Baseload + private EV + bus charging': branch_power_vector_magnitude_per_unit
    }
    selected_columns = (
        branch_power_vector_magnitude_per_unit.columns[
            branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_name').str.contains('22kV')
        ]
    )

    # Pre-process values.
    for key in values_dict:
        # Only use selected columns.
        values_dict[key] = values_dict[key].loc[:, selected_columns]
        # Obtain maximum utilization for all transformers.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'transformer'].max()
        )
        # Set over-utilized transformers to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > 1] = 1.0
        # Obtain histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins)
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = '1MVA Transformers'
    filename = 'transformer_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Bar(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key
        ))
    figure.update_layout(
        title=title,
        xaxis_title=f'{y_label} [{value_unit}]',
        yaxis_title='Frequency',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    )
    # figure.show()
    figure.write_image(os.path.join(results_path, filename + f".{fledge.config.config['plots']['file_format']}"))

    # Plot transformer utilization cumulative.
    values_dict = {
        'Baseload': branch_power_vector_magnitude_per_unit_2,
        'Baseload + private EV': branch_power_vector_magnitude_per_unit_1,
        'Baseload + private EV + bus charging': branch_power_vector_magnitude_per_unit
    }
    selected_columns = (
        branch_power_vector_magnitude_per_unit.columns[
            branch_power_vector_magnitude_per_unit.columns.get_level_values('branch_name').str.contains('22kV')
        ]
    )

    # Pre-process values.
    for key in values_dict:
        # Only use selected columns.
        values_dict[key] = values_dict[key].loc[:, selected_columns]
        # Obtain maximum utilization for all transformers.
        values_dict[key] = (
            values_dict[key].loc[:, values_dict[key].columns.get_level_values('branch_type') == 'transformer'].max()
        )
        # Set over-utilized transformers to 1 p.u. for better visualization.
        values_dict[key].loc[values_dict[key] > 1] = 1.0
        # Obtain histogram values.
        values_dict[key] = (
            pd.Series([*np.histogram(values_dict[key], bins=histogram_bins)[0], 0], index=histogram_bins).cumsum()
            / len(values_dict[key])
        )

    # Obtain plot title / labels / filename.
    title = '1MVA Transformers'
    filename = 'transformer_utilization_histogram'
    y_label = 'Peak utilization'
    value_unit = 'p.u.'

    # Create plot.
    figure = go.Figure()
    for key in values_dict:
        figure.add_trace(go.Scatter(
            x=values_dict[key].index,
            y=values_dict[key].values,
            name=key,
            line=go.scatter.Line(shape='hv')
        ))
    # Add horizontal line at 90%.
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
    figure.write_image(os.path.join(results_path, filename + f".{fledge.config.config['plots']['file_format']}"))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
