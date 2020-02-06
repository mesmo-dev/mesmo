"""Utility functions module."""

import numpy as np
import pandas as pd


def get_index(
        index_set: pd.Index,
        **kwargs
):
    """Obtain integer index array for given index set / level / value list combination."""

    # Obtain mask for each level / value list combination keyword arguments.
    mask = np.ones(len(index_set), dtype=np.bool)
    for level, values in kwargs.items():
        values = [values] if type(values) is not list else values
        mask &= index_set.get_level_values(level).isin(values)

    # Obtain integer index array.
    index = np.flatnonzero(mask)

    return index
