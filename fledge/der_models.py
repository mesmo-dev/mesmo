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
            fixed_load_data: fledge.database_interface.FixedLoadData,
            load_name: str
    ):
        """Construct fixed load model object by `fixed_load_data` and `load_name`."""

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


class EVChargerModel(DERModel):
    """EV charger model object."""

    def __init__(
            self,
            ev_charger_data: fledge.database_interface.EVChargerData,
            load_name: str
    ):
        """Construct EV charger model object by `ev_charger_data` and `load_name`."""

        # Get fixed load data by `load_name`.
        ev_charger = ev_charger_data.ev_chargers.loc[load_name, :]

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            ev_charger_data.ev_charger_timeseries_dict[ev_charger['timeseries_name']]['apparent_power_per_unit']
            * ev_charger['scaling_factor']
            * ev_charger['active_power']
            * -1.0  # Load / demand is negative.
        )
        self.reactive_power_nominal_timeseries = (
            ev_charger_data.ev_charger_timeseries_dict[ev_charger['timeseries_name']]['apparent_power_per_unit']
            * ev_charger['scaling_factor']
            * ev_charger['reactive_power']
            * -1.0  # Load / demand is negative.
        )


class FlexibleLoadModel(DERModel):
    """Flexible load model object."""

    def __init__(
            self,
            flexible_load_data: fledge.database_interface.FlexibleLoadData,
            load_name: str
    ):
        """Construct flexible load model object by `flexible_load_data` and `load_name`."""

        # Get fixed load data by `load_name`.
        flexible_load = flexible_load_data.flexible_loads.loc[load_name, :]

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            flexible_load_data.flexible_load_timeseries_dict[flexible_load['timeseries_name']]['apparent_power_per_unit']
            * flexible_load['scaling_factor']
            * flexible_load['active_power']
            * -1.0  # Load / demand is negative.
        )
        self.reactive_power_nominal_timeseries = (
            flexible_load_data.flexible_load_timeseries_dict[flexible_load['timeseries_name']]['apparent_power_per_unit']
            * flexible_load['scaling_factor']
            * flexible_load['reactive_power']
            * -1.0  # Load / demand is negative.
        )
