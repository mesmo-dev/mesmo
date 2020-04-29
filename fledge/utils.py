"""Utility functions module."""

import os
import numpy as np
import pandas as pd
import typing

import fledge.config

logger = fledge.config.get_logger(__name__)


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


class ResultsDict(typing.Dict[str, pd.DataFrame]):
    """Results dictionary object, i.e., a modified dictionary object with strings as keys and dataframes as values.

    - When printed or represented as string, all dataframes are printed in full.
    - Provides a method for storing all results dataframes to CSV files.
    """

    def __repr__(self) -> str:
        """Obtain string representation of results."""

        repr_string = ""
        for key in self:
            repr_string += f"{key} = \n{self[key]}\n"

        return repr_string

    def to_csv(self, path: str) -> None:
        """Store results to CSV files at given `path`."""

        for key in self:
            self[key].to_csv(os.path.join(path, f'{key}.csv'))
