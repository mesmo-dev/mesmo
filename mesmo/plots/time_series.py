"""Timeseries-base plotting functions."""

import numpy as np
import plotly.graph_objects as go

from mesmo import data_models
from mesmo.plots import constants


def der_active_power_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.ACTIVE_POWER} per DER"
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.ACTIVE_POWER} [{constants.ValueUnitLabels.WATT}]"
    legend_title = constants.ValueLabels.DERS

    for der_type, der_name in results.der_model_set_index.ders:
        values = results.der_operation_results.der_active_power_vector.loc[:, (der_type, der_name)]
        figure.add_trace(go.Scatter(x=values.index, y=values.values, name=f"{der_name} ({der_type})"))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=go.layout.Legend(title=legend_title, x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
    )
    return figure


def der_reactive_power_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.REACTIVE_POWER} per DER"
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.REACTIVE_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE_REACTIVE}]"
    legend_title = constants.ValueLabels.DERS

    for der_type, der_name in results.der_model_set_index.ders:
        values = results.der_operation_results.der_reactive_power_vector.loc[:, (der_type, der_name)]
        figure.add_trace(go.Scatter(x=values.index, y=values.values, name=f"{der_name} ({der_type})"))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=go.layout.Legend(title=legend_title, x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
    )
    return figure


def der_apparent_power_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.APPARENT_POWER} per DER"
    filename = der_apparent_power_time_series.__name__
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.APPARENT_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE}]"
    legend_title = constants.ValueLabels.DERS

    for der_type, der_name in results.der_model_set_index.ders:
        # TODO: Add apparent power in result directly
        values = np.sqrt(
            results.der_operation_results.der_active_power_vector.loc[:, (der_type, der_name)] ** 2
            + results.der_operation_results.der_reactive_power_vector.loc[:, (der_type, der_name)] ** 2
        ) * np.sign(results.der_operation_results.der_active_power_vector.loc[:, (der_type, der_name)])
        figure.add_trace(go.Scatter(x=values.index, y=values.values, name=f"{der_name} ({der_type})"))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=go.layout.Legend(title=legend_title, x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
    )
    return figure


def der_aggregated_active_power_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.ACTIVE_POWER} aggregated for all DERs"
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.ACTIVE_POWER} [{constants.ValueUnitLabels.WATT}]"
    line_name = constants.ValueLabels.DERS

    values = results.der_operation_results.der_active_power_vector.sum(axis="columns")
    figure.add_trace(go.Scatter(x=values.index, y=values.values, name=line_name))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )
    return figure


def der_aggregated_reactive_power_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.REACTIVE_POWER} aggregated for all DERs"
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.REACTIVE_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE_REACTIVE}]"
    line_name = constants.ValueLabels.DERS

    values = results.der_operation_results.der_reactive_power_vector.sum(axis="columns")
    figure.add_trace(go.Scatter(x=values.index, y=values.values, name=line_name))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )
    return figure


def der_aggregated_apparent_power_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.APPARENT_POWER} aggregated for all DERs"
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.APPARENT_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE}]"
    line_name = constants.ValueLabels.DERS

    # TODO: Add apparent power in result directly
    values = np.sqrt(
        results.der_operation_results.der_active_power_vector.sum(axis="columns") ** 2
        + results.der_operation_results.der_reactive_power_vector.sum(axis="columns") ** 2
    ) * np.sign(results.der_operation_results.der_active_power_vector.sum(axis="columns"))
    figure.add_trace(go.Scatter(x=values.index, y=values.values, name=line_name))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )
    return figure


def node_voltage_per_unit_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.VOLTAGE} per Nodes"
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.VOLTAGE} [{constants.ValueUnitLabels.VOLT_PER_UNIT}]"
    legend_title = constants.ValueLabels.NODES

    for node_type, node_name, phase in results.electric_grid_model_index.nodes:
        values = results.electric_grid_operation_results.node_voltage_magnitude_vector_per_unit.loc[
            :, (slice(None), node_name, slice(None))
        ].mean(axis="columns")
        figure.add_trace(go.Scatter(x=values.index, y=values.values, name=f"{node_name} ({node_type})"))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=go.layout.Legend(title=legend_title, x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
    )
    return figure


def node_aggregated_voltage_per_unit_time_series(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    title = f"{constants.ValueLabels.VOLTAGE} aggregated for all Nodes"
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.VOLTAGE} [{constants.ValueUnitLabels.VOLT_PER_UNIT}]"

    for timestep in results.electric_grid_model_index.timesteps:
        values = results.electric_grid_operation_results.node_voltage_magnitude_vector_per_unit.loc[timestep, :]
        figure.add_trace(go.Box(name=timestep.isoformat(), y=values.T.values))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )
    return figure
