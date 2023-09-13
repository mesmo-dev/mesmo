"""Timeseries-base plotting functions."""

import numpy as np
import pathlib
import plotly.graph_objects as go

from mesmo import data_models
from mesmo.plots import plot_utils, constants


def der_active_power_time_series(results: data_models.RunResults, results_path: pathlib.Path):
    title = f"{constants.ValueLabels.ACTIVE_POWER} per DER"
    filename = der_active_power_time_series.__name__
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.ACTIVE_POWER} [{constants.ValueUnitLabels.WATT}]"
    legend_title = constants.ValueLabels.DERS

    figure = go.Figure()
    for der_type, der_name in results.der_model_set_index.ders:
        values = results.der_operation_results.der_active_power_vector.loc[:, (der_type, der_name)]
        figure.add_trace(go.Scatter(x=values.index, y=values.values, name=f"{der_name} ({der_type})"))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=go.layout.Legend(title=legend_title, x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
    )
    plot_utils.write_figure_plotly(figure, results_path / filename)


def der_reactive_power_time_series(results: data_models.RunResults, results_path: pathlib.Path):
    title = f"{constants.ValueLabels.REACTIVE_POWER} per DER"
    filename = der_reactive_power_time_series.__name__
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.REACTIVE_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE_REACTIVE}]"
    legend_title = constants.ValueLabels.DERS

    figure = go.Figure()
    for der_type, der_name in results.der_model_set_index.ders:
        values = results.der_operation_results.der_reactive_power_vector.loc[:, (der_type, der_name)]
        figure.add_trace(go.Scatter(x=values.index, y=values.values, name=f"{der_name} ({der_type})"))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=go.layout.Legend(title=legend_title, x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
    )
    plot_utils.write_figure_plotly(figure, results_path / filename)


def der_apparent_power_time_series(results: data_models.RunResults, results_path: pathlib.Path):
    title = f"{constants.ValueLabels.APPARENT_POWER} per DER"
    filename = der_apparent_power_time_series.__name__
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.APPARENT_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE}]"
    legend_title = constants.ValueLabels.DERS

    figure = go.Figure()
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
    plot_utils.write_figure_plotly(figure, results_path / filename)


def der_aggregated_active_power_time_series(results: data_models.RunResults, results_path: pathlib.Path):
    title = f"{constants.ValueLabels.ACTIVE_POWER} aggregated for all DERs"
    filename = der_active_power_time_series.__name__
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.ACTIVE_POWER} [{constants.ValueUnitLabels.WATT}]"
    line_name = constants.ValueLabels.DERS

    figure = go.Figure()
    values = results.der_operation_results.der_active_power_vector.sum(axis="columns")
    figure.add_trace(go.Scatter(x=values.index, y=values.values, name=line_name))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    plot_utils.write_figure_plotly(figure, results_path / filename)


def der_aggregated_reactive_power_time_series(results: data_models.RunResults, results_path: pathlib.Path):
    title = f"{constants.ValueLabels.REACTIVE_POWER} aggregated for all DERs"
    filename = der_reactive_power_time_series.__name__
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.REACTIVE_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE_REACTIVE}]"
    line_name = constants.ValueLabels.DERS

    figure = go.Figure()
    values = results.der_operation_results.der_reactive_power_vector.sum(axis="columns")
    figure.add_trace(go.Scatter(x=values.index, y=values.values, name=line_name))
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    plot_utils.write_figure_plotly(figure, results_path / filename)


def der_aggregated_apparent_power_time_series(results: data_models.RunResults, results_path: pathlib.Path):
    title = f"{constants.ValueLabels.APPARENT_POWER} aggregated for all DERs"
    filename = der_apparent_power_time_series.__name__
    x_label = constants.ValueLabels.TIME
    y_label = f"{constants.ValueLabels.APPARENT_POWER} [{constants.ValueUnitLabels.VOLT_AMPERE}]"
    line_name = constants.ValueLabels.DERS

    figure = go.Figure()
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
    )
    plot_utils.write_figure_plotly(figure, results_path / filename)
