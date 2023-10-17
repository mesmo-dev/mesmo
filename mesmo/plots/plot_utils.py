"""Plotting utility functions."""

import json
import pathlib

import plotly.graph_objects as go
import plotly.io as pio

import mesmo.config
import mesmo.utils


def get_plotly_figure_json(figure: go.Figure) -> str:
    """Get JSON string representation of plotly figure.

    Args:
        figure (go.Figure): Figure for which the JSON representation is generated

    Returns:
        str: JSON representation of given figure
    """
    json_dict = json.loads(pio.to_json(figure))
    json_dict["layout"].pop("template")  # Exclude template information to minify JSON
    return json.dumps(json_dict)


def write_plotly_figure_file(
    figure: go.Figure,
    results_path: pathlib.Path,
    file_format=mesmo.config.config["plots"]["file_format"],
    width=mesmo.config.config["plots"]["plotly_figure_width"],
    height=mesmo.config.config["plots"]["plotly_figure_height"],
):
    """Utility function for writing / storing plotly figure to output file. File format can be given with
    `file_format` keyword argument, otherwise the default is obtained from config parameter `plots/file_format`.

    - `results_path` should be given as file name without file extension, because the file extension is appended
      automatically based on given `file_format`.
    - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'
    """

    if file_format in ["png", "jpg", "jpeg", "webp", "svg", "pdf"]:
        pio.write_image(
            figure,
            f"{results_path}.{file_format}",
            width=width,
            height=height,
        )
    elif file_format in ["html"]:
        pio.write_html(figure, f"{results_path}.{file_format}")
        mesmo.utils.launch(pathlib.Path(f"{results_path}.{file_format}"))
    elif file_format in ["json"]:
        json_dict = json.loads(pio.to_json(figure))
        json_dict["layout"].pop("template")  # Exclude template information to minify JSON
        with open(f"{results_path}.{file_format}", "w") as json_file:
            json.dump(json_dict, json_file)
    else:
        raise ValueError(
            f"Invalid `file_format` for `write_figure_plotly`: {file_format}"
            f" - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'"
        )

    # Additionally to the requested format, also output at plottable item in JSON format.
    if file_format != "json":
        write_plotly_figure_file(figure, results_path, "json", width, height)
