"""Utility functions module."""

import copy
import datetime
import functools
import itertools
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import ray
import re
import shutil
import subprocess
import sys
import time
import tqdm
import typing

import mesmo.config

logger = mesmo.config.get_logger(__name__)

# Instantiate dictionary for execution time logging.
log_times = dict()


class ObjectBase(object):
    """MESMO object base class, which extends the Python object base class.

    - Requires all attributes, i.e. parameters or object variables, to be defined with type declaration at the
      beginning of the class definition. Setting a value to an attribute which has not been defined will raise
      a warning. This is to ensure consistent definition structure of MESMO classes.
    - String representation of the object is the concatenation of the string representation of all its attributes.
      Thus, printing the object will print all its attributes.

    Example:

        Attributes should be defined in the beginning of the class definition as follows::

            class ExampleClass(ObjectBase):

                example_attribute1: str
                example_attribute2: pd.DataFrame

        In this case, ``example_attribute1`` and ``example_attribute2`` are valid attributes of the class.
    """

    def __setattr__(self, attribute_name, value):

        # Assert that attribute name is valid.
        # - Valid attributes are those which are defined as results class attributes with type declaration.
        if not (attribute_name in typing.get_type_hints(type(self))):
            logger.warning(
                f"Setting undefined attribute '{attribute_name}'. "
                f"Please ensure that the attribute has been defined by a type declaration "
                f"in the class definition of {type(self)}."
            )

        # Set attribute value.
        super().__setattr__(attribute_name, value)

    def __repr__(self) -> str:
        """Obtain string representation."""

        # Obtain attributes.
        attributes = vars(self)

        # Obtain representation string.
        repr_string = ""
        for attribute_name in attributes:
            repr_string += f"{attribute_name} = \n{attributes[attribute_name]}\n"

        return repr_string

    def copy(self):
        """Return a copy of this object. A new object will be created with a copy of the calling object’s attributes.
        Modifications to the attributes of the copy will not be reflected in the original object.
        """

        return copy.deepcopy(self)


class ResultsBase(ObjectBase):
    """Results object base class."""

    def __init__(self, **kwargs):

        # Set all keyword arguments as attributes.
        for attribute_name in kwargs:
            self.__setattr__(attribute_name, kwargs[attribute_name])

    def __getitem__(self, key):
        # Enable dict-like attribute getting.
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        # Enable dict-like attribute setting.
        self.__setattr__(key, value)

    def update(self, other_results):

        # Obtain attributes of other results object.
        attributes = vars(other_results)

        # Update attributes.
        # - Existing attributes are overwritten with values from the other results object.
        for attribute_name in attributes:
            if attributes[attribute_name] is not None:
                self.__setattr__(attribute_name, attributes[attribute_name])

    def save(self, results_path: pathlib.Path):
        """Store results to files at given results path.

        - Each results variable / attribute will be stored as separate file with the attribute name as file name.
        - Pandas Series / DataFrame are stored to CSV.
        - Other objects are stored to pickle binary file (PKL).
        """

        # Obtain results attributes.
        attributes = vars(self)

        # Store each attribute to a separate file.
        for attribute_name in attributes:
            if type(attributes[attribute_name]) in (pd.Series, pd.DataFrame):
                # Pandas Series / DataFrame are stored to CSV.
                attributes[attribute_name].to_csv((results_path / f"{attribute_name}.csv"))
            else:
                # Other objects are stored to pickle binary file (PKL).
                with open((results_path / f"{attribute_name}.pkl"), "wb") as output_file:
                    pickle.dump(attributes[attribute_name], output_file, pickle.HIGHEST_PROTOCOL)

    def load(self, results_path: pathlib.Path):
        """Load results from given path."""

        # Obtain all CSV and PKL files at results path.
        files = list(results_path.glob("*.csv")) + list(results_path.glob("*.pkl"))

        # Load all files which correspond to valid attributes.
        for file in files:

            # Obtain file extension / attribute name.
            file_extension = file.suffix
            attribute_name = file.stem

            # Load file and set attribute value.
            if attribute_name in typing.get_type_hints(type(self)):
                if file_extension.lower() == ".csv":
                    value = pd.read_csv(file)
                else:
                    with open(file, "rb") as input_file:
                        value = pickle.load(input_file)
                self.__setattr__(attribute_name, value)
            else:
                # Files which do not match any valid results attribute are not loaded.
                logger.debug(f"Skipping results file which does match any valid results attribute: {file}")

        return self


def starmap(
    function: typing.Callable, argument_sequence: typing.Iterable[tuple], keyword_arguments: dict = None
) -> list:
    """Utility function to execute a function for a sequence of arguments, effectively replacing a for-loop.
    Allows running repeated function calls in-parallel, based on Python's `multiprocessing` module.

    - If configuration parameter `run_parallel` is set to True, execution is passed to `starmap`
      of multiprocessing pool, hence running the function calls in parallel.
    - Otherwise, execution is passed to `itertools.starmap`, which is the non-parallel equivalent.
    """

    # Apply keyword arguments.
    if keyword_arguments is not None:
        function_partial = functools.partial(function, **keyword_arguments)
    else:
        function_partial = function

    # Ensure that argument sequence is list.
    argument_sequence = list(argument_sequence)

    if mesmo.config.config["multiprocessing"]["run_parallel"]:
        # TODO: Remove old parallel pool traces.
        # # If `run_parallel`, use starmap from multiprocessing pool for parallel execution.
        # if mesmo.config.parallel_pool is None:
        #     # Setup parallel pool on first execution.
        #     log_time('parallel pool setup')
        #     mesmo.config.parallel_pool = mesmo.config.get_parallel_pool()
        #     log_time('parallel pool setup')
        # results = mesmo.config.parallel_pool.starmap(function_partial, list(argument_sequence))

        # If `run_parallel`, use `ray_starmap` for parallel execution.
        if mesmo.config.parallel_pool is None:
            log_time("parallel pool setup")
            ray.init(num_cpus=max(int(mesmo.config.config["multiprocessing"]["cpu_share"] * os.cpu_count()), 1))
            mesmo.config.parallel_pool = True
            log_time("parallel pool setup")
        results = ray_starmap(function_partial, argument_sequence)
    else:
        # If not `run_parallel`, use for loop for sequential execution.
        results = [
            function_partial(*arguments)
            for arguments in tqdm.tqdm(
                argument_sequence,
                total=len(argument_sequence),
                disable=(mesmo.config.config["logs"]["level"] != "debug"),  # Progress bar only shown in debug mode.
            )
        ]

    return results


def chunk_dict(dict_in: dict, chunk_count: int = os.cpu_count()):
    """Divide dictionary into equally sized chunks."""

    chunk_size = int(np.ceil(len(dict_in) / chunk_count))
    dict_iter = iter(dict_in)

    return [
        {j: dict_in[j] for j in itertools.islice(dict_iter, chunk_size)} for i in range(0, len(dict_in), chunk_size)
    ]


def chunk_list(list_in: typing.Union[typing.Iterable, typing.Sized], chunk_count: int = os.cpu_count()):
    """Divide list into equally sized chunks."""

    chunk_size = int(np.ceil(len(list_in) / chunk_count))
    list_iter = iter(list_in)

    return [[j for j in itertools.islice(list_iter, chunk_size)] for i in range(0, len(list_in), chunk_size)]


def ray_iterator(objects: list):
    """Utility iterator for a list of parallelized ``ray`` objects.

    - This iterator enables progress reporting with ``tqdm`` in :func:`ray_get`.
    """

    while objects:
        done, objects = ray.wait(objects)
        yield ray.get(done[0])


def ray_get(objects: list):
    """Utility function for parallelized execution of a list of ``ray`` objects.

    - This function enables the parallelized execution with built-in progress reporting.
    """

    try:
        for _ in tqdm.tqdm(
            ray_iterator(objects),
            total=len(objects),
            disable=(mesmo.config.config["logs"]["level"] != "debug"),  # Progress bar only shown in debug mode.
        ):
            pass
    except TypeError:
        pass
    return ray.get(objects)


def ray_starmap(function_handle: typing.Callable, argument_sequence: list):
    """Utility function to provide an interface similar to ``itertools.starmap`` for ``ray``.

    - This replicates the ``starmap`` interface of the ``multiprocessing`` API, which ray also supports,
      but allows for additional modifications, e.g. progress reporting via :func:`ray_get`.
    """

    return ray_get(
        [ray.remote(lambda *args: function_handle(*args)).remote(*arguments) for arguments in argument_sequence]
    )


def log_time(label: str, log_level: str = "debug", logger_object: logging.Logger = logger):
    """Log start / end message and time duration for given label.

    - When called with given label for the first time, will log start message.
    - When called subsequently with the same / previously used label, will log end message and time duration since
      logging the start message.
    - The log level for start / end messages can be given as keyword argument, By default, messages are logged as
      debug messages.
    - The logger object can be given as keyword argument. By default, uses ``utils.logger`` as logger.
    - Start message: "Starting ``label``."
    - End message: "Completed ``label`` in ``duration`` seconds."

    Arguments:
        label (str): Label for the start / end message.

    Keyword Arguments:
        log_level (str): Log level to which the start / end messages are output. Choices: 'debug', 'info'.
            Default: 'debug'.
        logger_object (logging.logger.Logger): Logger object to which the start / end messages are output. Default:
            ``utils.logger``.
    """

    time_now = time.time()

    if log_level == "debug":
        logger_handle = lambda message: logger_object.debug(message)
    elif log_level == "info":
        logger_handle = lambda message: logger_object.info(message)
    else:
        raise ValueError(f"Invalid log level: '{log_level}'")

    if label in log_times.keys():
        logger_handle(f"Completed {label} in {(time_now - log_times.pop(label)):.6f} seconds.")
    else:
        log_times[label] = time_now
        logger_handle(f"Starting {label}.")


def get_index(index_set: typing.Union[pd.Index, pd.DataFrame], raise_empty_index_error: bool = True, **levels_values):
    """Utility function for obtaining the integer index array for given index set / level / value list combination.

    :syntax:
        - ``get_index(electric_grid_model.nodes, node_type='source', phase=1)``: Get index array for entries in
          index set `electric_grid_model.nodes` with given `node_type` and `phase`.

    Arguments:
        index_set (pd.Index): Index set, e.g., `electric_grid_model.nodes`.

    Keyword Arguments:
        raise_empty_index_error (bool): If true, raise an exception if obtained index array is empty. This is
            the default behavior, because it is usually caused by an invalid level / value combination.
        level (value): All other keyword arguments are interpreted as level / value combinations, where `level`
            must correspond to a level name of the index set.
    """

    # Define handle for get_level_values() depending on index set type.
    if isinstance(index_set, pd.Index):
        get_level_values = lambda level: index_set.get_level_values(level)
    elif isinstance(index_set, pd.DataFrame):
        # get_level_values = lambda level: index_set.get(level, pd.Series(index=index_set.index, name=level))
        get_level_values = lambda level: index_set.loc[:, level]
    else:
        raise TypeError(f"Invalid index set type: {type(index_set)}")

    # Obtain mask for each level / values combination keyword arguments.
    mask = np.ones(len(index_set), dtype=bool)
    for level, values in levels_values.items():

        # Ensure that values are passed as list.
        if isinstance(values, list):
            pass
        elif isinstance(values, tuple):
            # If values are passed as tuple, wrap in list, but only if index
            # level values are tuples. Otherwise, convert to list.
            if isinstance(get_level_values(level).dropna().values[0], tuple):
                values = [values]
            else:
                values = list(values)
        elif isinstance(values, range):
            # Convert range to list.
            values = list(values)
        elif isinstance(values, np.ndarray):
            # Convert numpy arrays to list.
            values = values.tolist()
            values = [values] if not isinstance(values, list) else values
        elif isinstance(values, pd.Index):
            # Convert pandas index to list.
            values = values.to_list()
        else:
            # Convert single values into list with one item.
            values = [values]

        # Obtain mask.
        mask &= get_level_values(level).isin(values)

    # Obtain integer index array.
    index = np.flatnonzero(mask)

    # Assert that index is not empty.
    if raise_empty_index_error:
        if not (len(index) > 0):
            raise ValueError(f"Empty index returned for: {levels_values}")

    return index


def get_element_phases_array(element: pd.Series):
    """Utility function for obtaining the list of connected phases for given element data."""

    # Obtain list of connected phases.
    phases_array = np.flatnonzero(
        [
            False,  # Ground / '0' phase connection is not considered.
            element.at["is_phase_1_connected"] == 1,
            element.at["is_phase_2_connected"] == 1,
            element.at["is_phase_3_connected"] == 1,
        ]
    )

    return phases_array


def get_element_phases_string(element: pd.Series):
    """Utility function for obtaining the OpenDSS phases string for given element data."""

    # Obtain string of connected phases.
    phases_string = ""
    if element.at["is_phase_1_connected"] == 1:
        phases_string += ".1"
    if element.at["is_phase_2_connected"] == 1:
        phases_string += ".2"
    if element.at["is_phase_3_connected"] == 1:
        phases_string += ".3"

    return phases_string


def get_inverse_with_zeros(array: np.ndarray) -> np.ndarray:
    """Obtain the inverse of an array, but do not take the inverse of zero values to avoid numerical errors.

    - Takes the inverse of an array and replaces the inverse of any zero values with zero,
      thus avoiding `inf` or `nan` values in the inverse.
    """

    # Take inverse.
    # - Suppress numpy runtime warning for divide by zero, because it is expected.
    # TODO: `invalid='ignore'` to be removed once https://github.com/conda-forge/numpy-feedstock/issues/229 is fixed.
    with np.errstate(divide="ignore", invalid="ignore"):
        array_inverse = array**-1

    # Replace inverse of zero values.
    array_inverse[array == 0.0] = array[array == 0.0]

    return array_inverse


def get_results_path(base_name: str, scenario_name: str = None) -> pathlib.Path:
    """Generate results path, which is a new subfolder in the results directory. The subfolder name is
    assembled of the given base name, scenario name and current timestamp. The new subfolder is
    created on disk along with this.

    - Non-alphanumeric characters are removed from `base_name` and `scenario_name`.
    - If is a script file path or `__file__` is passed as `base_name`, the base file name without extension
      will be taken as base name.
    """

    # Preprocess results path name components, including removing non-alphanumeric characters.
    base_name = re.sub(r"\W-+", "", pathlib.Path(base_name).stem) + "_"
    scenario_name = "" if scenario_name is None else re.sub(r"\W-+", "", scenario_name) + "_"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    # Obtain results path.
    results_path = mesmo.config.config["paths"]["results"] / f"{base_name}{scenario_name}{timestamp}"

    # Instantiate results directory.
    results_path.mkdir(parents=True)

    return results_path


def get_alphanumeric_string(string: str):
    """Create lowercase alphanumeric string from given string, replacing non-alphanumeric characters with underscore."""

    return re.sub(r"[^0-9a-zA-Z_]+", "_", string).strip("_").lower()


def get_plotly_mapbox_zoom_center(
    longitudes: tuple,
    latitudes: tuple,
    width_to_height: float = 2.0,
) -> (float, dict):
    """Get optimal zoom and centering for a plotly mapbox plot, given lists of longitude and latitude values.

    - Assumes that longitude and latitude are in Mercator projection.
    - Temporary solution awaiting official implementation, see: https://github.com/plotly/plotly.js/issues/3434
    - Source: https://github.com/richieVil/rv_packages/blob/master/rv_geojson.py

    Arguments:
        longitudes (tuple): Longitude component of each location.
        latitudes (tuple): Latitude component of each location.

    Keyword Arguments:
        width_to_height (float): Expected ratio of final graph's with to height, used to select the constrained axis.

    Returns:
        float: Zoom value from 1 to 20.
        dict: Center position with 'lon' and 'lat' keys.
    """

    # Get center.
    longitude_max, longitude_min = max(longitudes), min(longitudes)
    latitude_max, latitude_min = max(latitudes), min(latitudes)
    center = {"lon": round((longitude_max + longitude_min) / 2, 6), "lat": round((latitude_max + latitude_min) / 2, 6)}

    # Define longitudinal range by zoom level (20 to 1) in degrees, if centered at equator.
    longitude_zoom_range = np.array(
        [
            0.0007,
            0.0014,
            0.003,
            0.006,
            0.012,
            0.024,
            0.048,
            0.096,
            0.192,
            0.3712,
            0.768,
            1.536,
            3.072,
            6.144,
            11.8784,
            23.7568,
            47.5136,
            98.304,
            190.0544,
            360.0,
        ]
    )

    # Get zoom level.
    margin = 1.2
    height = (latitude_max - latitude_min) * margin * width_to_height
    width = (longitude_max - longitude_min) * margin
    longitude_zoom = np.interp(width, longitude_zoom_range, range(20, 0, -1))
    latitude_zoom = np.interp(height, longitude_zoom_range, range(20, 0, -1))
    zoom = round(min(longitude_zoom, latitude_zoom), 2)

    return zoom, center


def launch(path: pathlib.Path):
    """Launch the file at given path with its associated application. If path is a directory, open in file explorer."""

    if not path.exists():
        raise FileNotFoundError(f"Cannot launch file or directory that does not exist: {path}")

    if sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)], shell=True)
    else:
        subprocess.Popen(["xdg-open", str(path)], shell=True)


def write_figure_plotly(
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
    elif file_format in ["json"]:
        pio.write_json(figure, f"{results_path}.{file_format}")
    else:
        raise ValueError(
            f"Invalid `file_format` for `write_figure_plotly`: {file_format}"
            f" - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'"
        )


def OptimizationProblem() -> "mesmo.solutions.OptimizationProblem":
    """:class:`mesmo.utils.OptimizationProblem` is a placeholder for :class:`mesmo.solutions.OptimizationProblem` for backwards
    compatibility and will be removed in a future version of MESMO.
    """

    # Import `solutions` module only here, to avoid circular import issues.
    import mesmo.solutions

    # Issue warning when using this class.
    logger.warning(
        "`mesmo.utils.OptimizationProblem` is a placeholder for `mesmo.solutions.OptimizationProblem` for backwards"
        " compatibility and will be removed in a future version of MESMO."
    )

    return mesmo.solutions.OptimizationProblem()


def cleanup():
    """Clear contents of the results directory."""

    for content in mesmo.config.config.paths.results.iterdir():
        if content.name not in ["README.md"]:
            if content.is_dir():
                shutil.rmtree(content)
            else:
                content.unlink()
