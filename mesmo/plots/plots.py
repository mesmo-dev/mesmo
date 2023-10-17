"""Interfaces to plotting functions."""

import pathlib
from typing import Callable

import plotly.graph_objects as go

from mesmo import data_models
from mesmo.plots import plot_utils


def plot_to_figure(
    *plot_functions: Callable[[go.Figure, data_models.RunResults], go.Figure], results: data_models.RunResults
) -> go.Figure:
    """Generate new plotly figure and apply given plotting function(s) for given run results.

    Args:
        Callable: Plotting function
        data_models.RunResults: MESMO run results as input for the plotting function

    Returns:
        go.Figure: Plotly figure containing the generated plot
    """
    figure = go.Figure()
    for plot_function in plot_functions:
        figure = plot_function(figure, results)
    return figure


def plot_to_json(
    *plot_functions: Callable[[go.Figure, data_models.RunResults], go.Figure], results: data_models.RunResults
) -> str:
    """Generate new plotly figure and apply given plotting function(s) for given run results. Output the final figure
    as JSON string.

    Args:
        Callable: Plotting function
        data_models.RunResults: MESMO run results as input for the plotting function

    Returns:
        str: JSON string containing the generated plot
    """
    return plot_utils.get_plotly_figure_json(plot_to_figure(*plot_functions, results=results))


def plot_to_file(
    *plot_functions: Callable[[go.Figure, data_models.RunResults], go.Figure],
    results: data_models.RunResults,
    results_path: pathlib.Path,
):
    """Generate new plotly figure and apply given plotting function(s) for given run results. Out put the final figure
    to file.

    Args:
        Callable: Plotting function
        data_models.RunResults: MESMO run results as input for the plotting function
        pathlib.Path: Results file output path
    """
    filename = plot_functions[0].__name__
    plot_utils.write_plotly_figure_file(
        plot_to_figure(*plot_functions, results=results), results_path=results_path / filename
    )
