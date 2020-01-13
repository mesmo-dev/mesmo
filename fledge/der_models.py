"""Distributed energy resource (DER) models."""

# TODO: Fix apparent power to active/reactive power ratio.

from multimethod import multimethod
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface

logger = fledge.config.get_logger(__name__)


class DERModel(object):
    """DER model object."""

    active_power_nominal_timeseries: pd.Series
    reactive_power_nominal_timeseries: pd.Series


class FixedLoadModel(DERModel):
    """Fixed load model object."""

    def __init__(
            self,
            fixed_load_data,
            load_name
    ):
        """Construct fixed load model object by fixed load data and load name."""

        # Get fixed load data by `load_name`.
        fixed_load = fixed_load_data.fixed_loads.loc[load_name, :]

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            fixed_load_data.fixed_load_timeseries_dict[fixed_load['timeseries_name']]['apparent_power_per_unit']
            * fixed_load['scaling_factor']
            * fixed_load['active_power']
            * -1.0  # Load / demand is negative.
        )
        self.reactive_power_nominal_timeseries = (
            fixed_load_data.fixed_load_timeseries_dict[fixed_load['timeseries_name']]['apparent_power_per_unit']
            * fixed_load['scaling_factor']
            * fixed_load['reactive_power']
            * -1.0  # Load / demand is negative.
        )
