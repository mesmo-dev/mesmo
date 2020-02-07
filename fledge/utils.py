"""Utility functions module."""

import numpy as np
import pandas as pd

import fledge.config

logger = fledge.config.get_logger(__name__)

def get_index(
        index_set: pd.Index,
        **levels_values
):
    """Utility function for obtaining the integer index array for given index set / level / value list combination."""

    # Obtain mask for each level / value list combination keyword arguments.
    mask = np.ones(len(index_set), dtype=np.bool)
    for level, values in levels_values.items():
        values = [values] if type(values) is not list else values
        mask &= index_set.get_level_values(level).isin(values)

    # Obtain integer index array.
    index = np.flatnonzero(mask)

    return index


def get_element_phases_list(element: pd.Series):
    """Utility function for obtaining the list of connected phases for given element data."""

    # Obtain list of connected phases.
    phases_list = (
        np.flatnonzero([
            element['is_phase_1_connected'] == 1,
            element['is_phase_2_connected'] == 1,
            element['is_phase_3_connected'] == 1
        ]).tolist()
    )

    return phases_list


def get_element_phases_string(element):
    """Utility function for obtaining the OpenDSS phases string for given element data."""

    # Obtain string of connected phases.
    phases_string = ""
    if element['is_phase_0_connected'] == 1:
        phases_string += ".0"
    if element['is_phase_1_connected'] == 1:
        phases_string += ".1"
    if element['is_phase_2_connected'] == 1:
        phases_string += ".2"
    if element['is_phase_3_connected'] == 1:
        phases_string += ".3"

    return phases_string
