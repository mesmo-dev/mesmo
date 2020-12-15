"""Utility functions module."""

import cvxpy as cp
import datetime
import dill
import functools
import glob
import itertools
import logging
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import re
import time
import typing
import subprocess
import sys

import cobmo.building_model
import fledge.config

logger = fledge.config.get_logger(__name__)

# Instantiate dictionary for execution time logging.
log_times = dict()


class ObjectBase(object):
    """FLEDGE object base class, which extends the Python object base class.

    - Requires all attributes, i.e. parameters or object variables, to be defined with type declaration at the
      beginning of the class definition. Setting a value to an attribute which has not been defined will raise an error.
      This is to ensure consistent definition structure of FLEDGE classes.
    - String representation of the object is the concatenation of the string representation of all its attributes.
      Thus, printing the object will print all its attributes.

    Example:

        Attributes should be defined in the beginning of the class definition as follows::

            class ExampleClass(ObjectBase):

                example_attribute1: str
                example_attribute2: pd.DataFrame

        In this case, ``example_attribute1`` and ``example_attribute2`` are valid attributes of the class.
    """

    def __setattr__(
            self,
            attribute_name,
            value
    ):

        # Assert that attribute name is valid.
        # - Valid attributes are those which are defined as results class attributes with type declaration.
        try:
            assert attribute_name in typing.get_type_hints(type(self))
        except AssertionError:
            logger.error(
                f"Cannot set invalid attribute '{attribute_name}'. "
                f"Please ensure that the attribute has been defined with type declaration in the class definition."
            )
            raise

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


class ResultsBase(ObjectBase):
    """Results object base class."""

    def __init__(
            self,
            **kwargs
    ):

        # Set all keyword arguments as attributes.
        for attribute_name in kwargs:
            self.__setattr__(attribute_name, kwargs[attribute_name])

    def update(
            self,
            other_results
    ):

        # Obtain attributes of other results object.
        attributes = vars(other_results)

        # Update attributes.
        # - Existing attributes are overwritten with values from the other results object.
        for attribute_name in attributes:
            if attributes[attribute_name] is not None:
                self.__setattr__(attribute_name, attributes[attribute_name])

    def save(
            self,
            results_path: str
    ):
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
                attributes[attribute_name].to_csv(os.path.join(results_path, f'{attribute_name}.csv'))
            else:
                # Other objects are stored to pickle binary file (PKL).
                with open(os.path.join(results_path, f'{attribute_name}.pkl'), 'wb') as output_file:
                    dill.dump(attributes[attribute_name], output_file, dill.HIGHEST_PROTOCOL)

    def load(
            self,
            results_path: str
    ):
        """Load results from given path."""

        # Obtain all CSV and PKL files at results path.
        files = glob.glob(os.path.join(results_path, '*.csv')) + glob.glob(os.path.join(results_path, '*.pkl'))

        # Load all files which correspond to valid attributes.
        for file in files:

            # Obtain file extension / attribute name.
            file_extension = os.path.splitext(file)[1]
            attribute_name = os.path.basename(os.path.splitext(file)[0])

            # Load file and set attribute value.
            if attribute_name in typing.get_type_hints(type(self)):
                if file_extension.lower() == '.csv':
                    value = pd.read_csv(file)
                else:
                    with open(file, 'rb') as input_file:
                        value = dill.load(input_file)
                self.__setattr__(attribute_name, value)
            else:
                # Files which do not match any valid results attribute are not loaded.
                logger.debug(f"Skipping results file which does match any valid results attribute: {file}")

        return self


class OptimizationProblem(object):
    """Optimization problem object for use with CVXPY."""

    constraints: list
    objective: cp.Expression
    cvxpy_problem: cp.Problem

    def __init__(self):

        self.constraints = []
        self.objective = cp.Constant(value=0.0)

    def solve(
            self,
            keep_problem=False
    ):

        # Instantiate CVXPY problem object.
        if hasattr(self, 'cvxpy_problem') and keep_problem:
            pass
        else:
            self.cvxpy_problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

        # Solve optimization problem.
        self.cvxpy_problem.solve(
            solver=(
                fledge.config.config['optimization']['solver_name'].upper()
                if fledge.config.config['optimization']['solver_name'] is not None
                else None
            ),
            verbose=fledge.config.config['optimization']['show_solver_output']
        )

        # Assert that solver exited with an optimal solution. If not, raise an error.
        try:
            assert self.cvxpy_problem.status == cp.OPTIMAL
        except AssertionError:
            logger.error(f"Solver termination status: {self.cvxpy_problem.status}")
            raise


def starmap(
        function: typing.Callable,
        argument_sequence: typing.Iterable[tuple],
        keyword_arguments: dict = None
) -> list:
    """Utility function to execute a function for a sequence of arguments, effectively replacing a for-loop.
    Allows running repeated function calls in-parallel, based on Python's `multiprocessing` module.

    - If configuration parameter `run_parallel` is set to True, execution is passed to `starmap`
      of `multiprocess.Pool`, hence running the function calls in parallel.
    - Otherwise, execution is passed to `itertools.starmap`, which is the non-parallel equivalent.
    """

    # Apply keyword arguments.
    if keyword_arguments is not None:
        function_partial = functools.partial(function, **keyword_arguments)
    else:
        function_partial = function

    if fledge.config.config['multiprocessing']['run_parallel']:
        # If `run_parallel`, use starmap from `multiprocess.Pool` for parallel execution.
        if fledge.config.parallel_pool is None:
            # Setup parallel pool on first execution.
            log_time('parallel pool setup')
            fledge.config.parallel_pool = fledge.config.get_parallel_pool()
            log_time('parallel pool setup')
        results = fledge.config.parallel_pool.starmap(function_partial, argument_sequence)
    else:
        # If not `run_parallel`, use `itertools.starmap` for non-parallel / sequential execution.
        results = list(itertools.starmap(function_partial, argument_sequence))

    return results


def log_timing_start(
        message: str,
        logger_object: logging.Logger = logger
) -> float:
    """Log start message and return start time. Should be used together with `log_timing_end`."""

    logger_object.debug(f"Start {message}.")

    return time.time()


def log_timing_end(
        start_time: float,
        message: str,
        logger_object: logging.Logger = logger
) -> float:
    """Log end message and execution time based on given start time. Should be used together with `log_timing_start`."""

    logger_object.debug(f"Completed {message} in {(time.time() - start_time):.6f} seconds.")

    return time.time()


def log_time(
        label: str,
        logger_object: logging.Logger = logger
):
    """Log start message and return start time. Should be used together with `log_timing_end`."""

    time_now = time.time()

    if label in log_times.keys():
        logger_object.debug(f"Completed {label} in {(time_now - log_times[label]):.6f} seconds.")
    else:
        log_times[label] = time_now
        logger_object.debug(f"Starting {label}.")


def get_index(
        index_set: pd.Index,
        raise_empty_index_error: bool = True,
        **levels_values
):
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

    # Obtain mask for each level / values combination keyword arguments.
    mask = np.ones(len(index_set), dtype=np.bool)
    for level, values in levels_values.items():

        # Ensure that values are passed as list.
        if isinstance(values, (list, tuple)):
            pass
        elif isinstance(values, np.ndarray):
            # Convert numpy arrays to list.
            values = values.tolist()
            values = [values] if not isinstance(values, list) else values
        else:
            # Convert single values into list with one item.
            values = [values]

        # Obtain mask.
        mask &= index_set.get_level_values(level).isin(values)

    # Obtain integer index array.
    index = np.flatnonzero(mask)

    # Assert that index is not empty.
    if raise_empty_index_error:
        try:
            assert len(index) > 0
        except AssertionError:
            logger.error(f"Empty index returned for: {levels_values}")
            raise

    return index


def get_element_phases_array(element: pd.Series):
    """Utility function for obtaining the list of connected phases for given element data."""

    # Obtain list of connected phases.
    phases_array = (
        np.flatnonzero([
            False,  # Ground / '0' phase connection is not considered.
            element.at['is_phase_1_connected'] == 1,
            element.at['is_phase_2_connected'] == 1,
            element.at['is_phase_3_connected'] == 1
        ])
    )

    return phases_array


def get_element_phases_string(element: pd.Series):
    """Utility function for obtaining the OpenDSS phases string for given element data."""

    # Obtain string of connected phases.
    phases_string = ""
    if element.at['is_phase_1_connected'] == 1:
        phases_string += ".1"
    if element.at['is_phase_2_connected'] == 1:
        phases_string += ".2"
    if element.at['is_phase_3_connected'] == 1:
        phases_string += ".3"

    return phases_string


def get_timestamp(
        time: datetime.datetime = None
) -> str:
    """Generate formatted timestamp string, e.g., for saving results with timestamp."""

    if time is None:
        time = datetime.datetime.now()

    return time.strftime('%Y-%m-%d_%H-%M-%S')


def get_results_path(
        base_name: str,
        scenario_name: str = None
) -> str:
    """Generate results path, which is a new subfolder in the results directory. The subfolder name is
    assembled of the given base name, scenario name and current timestamp. The new subfolder is
    created on disk along with this.

    - Non-alphanumeric characters are removed from `base_name` and `scenario_name`.
    - If is a script file path or `__file__` is passed as `base_name`, the base file name without extension
      will be taken as base name.
    """

    # Preprocess results path name components, including removing non-alphanumeric characters.
    base_name = re.sub(r'\W+', '', os.path.basename(os.path.splitext(base_name)[0])) + '_'
    scenario_name = '' if scenario_name is None else re.sub(r'\W+', '', scenario_name) + '_'
    timestamp = fledge.utils.get_timestamp()

    # Obtain results path.
    results_path = os.path.join(fledge.config.config['paths']['results'], f'{base_name}{scenario_name}{timestamp}')

    # Instantiate results directory.
    # TODO: Catch error if dir exists.
    os.mkdir(results_path)

    return results_path


def get_alphanumeric_string(
        string: str
):
    """Create lowercase alphanumeric string from given string, replacing non-alphanumeric characters with underscore."""

    return re.sub(r'\W+', '_', string).strip('_').lower()


def launch(path):
    """Launch the file at given path with its associated application. If path is a directory, open in file explorer."""

    try:
        assert os.path.exists(path)
    except AssertionError:
        logger.error(f'Cannot launch file or directory that does not exist: {path}')

    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        subprocess.Popen(['xdg-open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@fledge.config.memoize('get_building_model')
def get_building_model(*args, **kwargs):
    """Wrapper function for `cobmo.building_model.BuildingModel` with caching support for better performance."""

    return cobmo.building_model.BuildingModel(*args, **kwargs)


def write_figure_plotly(
        figure: go.Figure,
        results_path: str,
        file_format=fledge.config.config['plots']['file_format']
):
    """Utility function for writing / storing plotly figure to output file. File format can be given with
    `file_format` keyword argument, otherwise the default is obtained from config parameter `plots/file_format`.

    - `results_path` should be given as file name without file extension, because the file extension is appended
      automatically based on given `file_format`.
    - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'
    """

    if file_format in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        pio.write_image(figure, f"{results_path}.{file_format}")
    elif file_format in ['html']:
        pio.write_html(figure, f"{results_path}.{file_format}")
    elif file_format in ['json']:
        pio.write_json(figure, f"{results_path}.{file_format}")
    else:
        logger.error(
            f"Invalid `file_format` for `write_figure_plotly`: {file_format}"
            f" - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'"
        )
        raise ValueError
