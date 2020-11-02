"""Utility functions module."""

import datetime
import functools
import itertools
import logging
import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo
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


def solve_optimization(
        optimization_problem: pyo.ConcreteModel,
        enable_duals=False
):
    """Utility function for solving a Pyomo optimization problem. Automatically instantiates the solver as given in
    config. Raises error if no feasible solution is found.
    """

    # Enable duals.
    if enable_duals:
        optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Solve optimization problem.
    optimization_result = (
        fledge.config.optimization_solver.solve(
            optimization_problem,
            tee=fledge.config.config['optimization']['show_solver_output']
        )
    )

    # Assert that solver exited with any solution. If not, raise an error.
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        logger.error(f"Solver termination condition: {optimization_result.solver.termination_condition}")
        raise


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
        scenario_name: str
) -> str:
    """Generate results path, which is a new subfolder in the results directory. The subfolder name is
    assembled of the given base name, scenario name and current timestamp. The new subfolder is
    created on disk along with this.
    """

    # Obtain results path.
    results_path = (
        os.path.join(
            fledge.config.config['paths']['results'],
            # Remove non-alphanumeric characters, except `_`, then append timestamp string.
            re.sub(r'\W+', '', f'{base_name}_{scenario_name}_') + fledge.utils.get_timestamp()
        )
    )

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

    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.call(['open', path])
    else:
        subprocess.call(['xdg-open', path])


@fledge.config.memoize('get_building_model')
def get_building_model(*args, **kwargs):
    """Wrapper function for `cobmo.building_model.BuildingModel` with caching support for better performance."""

    return cobmo.building_model.BuildingModel(*args, **kwargs)
