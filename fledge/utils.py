"""Utility functions module."""

import datetime
import itertools
import numpy as np
import os
import pandas as pd
import re
import typing

import fledge.config

logger = fledge.config.get_logger(__name__)


def starmap(
        function: typing.Callable,
        argument_sequence: typing.List[tuple]
) -> list:
    """Utility function to execute a function for a sequence of arguments, effectively replacing a for-loop.
    Allows running repeated function calls in-parallel, based on Python's `multiprocessing` module.

    - If configuration parameter `run_parallel` is set to True, execution is passed to `starmap`
      of `multiprocessing.Pool`, hence running the function calls in parallel.
    - Otherwise, execution is passed to `itertools.starmap`, which is the non-parallel equivalent.
    """

    if fledge.config.config['multiprocessing']['run_parallel']:
        results = fledge.config.parallel_pool.starmap(function, argument_sequence)
    else:
        results = itertools.starmap(function, argument_sequence)

    return results


def get_index(
        index_set: pd.Index,
        **levels_values
):
    """Utility function for obtaining the integer index array for given index set / level / value list combination."""

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
