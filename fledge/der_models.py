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
            der_name: str
    ):
        """Construct fixed load model object by `fixed_load_data` and `der_name`."""

        # Get fixed load data by `der_name`.
        fixed_load = fixed_load_data.fixed_loads.loc[der_name, :]

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            fixed_load_data.fixed_load_timeseries_dict[
                fixed_load['timeseries_name']
            ]['apparent_power_per_unit'].rename('active_power')
            * fixed_load['scaling_factor']
            * fixed_load['active_power']
            * -1.0  # Load / demand is negative.
        )
        self.reactive_power_nominal_timeseries = (
            fixed_load_data.fixed_load_timeseries_dict[
                fixed_load['timeseries_name']
            ]['apparent_power_per_unit'].rename('reactive_power')
            * fixed_load['scaling_factor']
            * fixed_load['reactive_power']
            * -1.0  # Load / demand is negative.
        )


class EVChargerModel(DERModel):
    """EV charger model object."""

    def __init__(
            self,
            ev_charger_data: fledge.database_interface.EVChargerData,
            der_name: str
    ):
        """Construct EV charger model object by `ev_charger_data` and `der_name`."""

        # Get fixed load data by `der_name`.
        ev_charger = ev_charger_data.ev_chargers.loc[der_name, :]

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            ev_charger_data.ev_charger_timeseries_dict[
                ev_charger['timeseries_name']
            ]['apparent_power_per_unit'].rename('active_power')
            * ev_charger['scaling_factor']
            * ev_charger['active_power']
            * -1.0  # Load / demand is negative.
        )
        self.reactive_power_nominal_timeseries = (
            ev_charger_data.ev_charger_timeseries_dict[
                ev_charger['timeseries_name']
            ]['apparent_power_per_unit'].rename('reactive_power')
            * ev_charger['scaling_factor']
            * ev_charger['reactive_power']
            * -1.0  # Load / demand is negative.
        )


class FlexibleDERModel(DERModel):
    """Flexible DER model, e.g., flexible load, object."""

    state_names: pd.Index
    control_names: pd.Index
    disturbance_names: pd.Index
    output_names: pd.Index
    state_matrix: pd.DataFrame
    control_matrix: pd.DataFrame
    disturbance_matrix: pd.DataFrame
    state_output_matrix: pd.DataFrame
    control_output_matrix: pd.DataFrame
    disturbance_output_matrix: pd.DataFrame
    disturbance_timeseries: pd.DataFrame
    output_maximum_timeseries: pd.DataFrame
    output_minimum_timeseries: pd.DataFrame


class FlexibleLoadModel(FlexibleDERModel):
    """Flexible load model object."""

    def __init__(
            self,
            flexible_load_data: fledge.database_interface.FlexibleLoadData,
            der_name: str
    ):
        """Construct flexible load model object by `flexible_load_data` and `der_name`."""

        # Get fixed load data by `der_name`.
        flexible_load = flexible_load_data.flexible_loads.loc[der_name, :]

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            flexible_load_data.flexible_load_timeseries_dict[
                flexible_load['timeseries_name']
            ]['apparent_power_per_unit'].rename('active_power')
            * flexible_load['scaling_factor']
            * flexible_load['active_power']
        )
        self.reactive_power_nominal_timeseries = (
            flexible_load_data.flexible_load_timeseries_dict[
                flexible_load['timeseries_name']
            ]['apparent_power_per_unit'].rename('reactive_power')
            * flexible_load['scaling_factor']
            * flexible_load['reactive_power']
        )

        # Calculate nominal accumulated energy timeseries.
        # TODO: Consider reactive power in accumulated energy.
        accumulated_energy_nominal_timeseries = (
            self.active_power_nominal_timeseries.cumsum().rename('accumulated_energy')
        )

        # Instantiate indexes.
        self.state_names = pd.Index(['accumulated_energy'])
        self.control_names = pd.Index(['active_power', 'reactive_power'])
        self.disturbance_names = pd.Index([])
        self.output_names = pd.Index(['accumulated_energy', 'active_power', 'reactive_power'])

        # Instantiate state space matrices.
        # TODO: Consolidate indexing approach with electric grid model.
        self.state_matrix = (
            pd.DataFrame(0.0, index=self.state_names, columns=self.state_names)
        )
        self.state_matrix.at['accumulated_energy', 'accumulated_energy'] = 1.0
        self.control_matrix = (
            pd.DataFrame(0.0, index=self.state_names, columns=self.control_names)
        )
        self.control_matrix.at['accumulated_energy', 'active_power'] = 1.0
        self.disturbance_matrix = (
            pd.DataFrame(0.0, index=self.state_names, columns=self.disturbance_names)
        )
        self.state_output_matrix = (
            pd.DataFrame(0.0, index=self.output_names, columns=self.state_names)
        )
        self.state_output_matrix.at['accumulated_energy', 'accumulated_energy'] = 1.0
        self.control_output_matrix = (
            pd.DataFrame(0.0, index=self.output_names, columns=self.control_names)
        )
        self.control_output_matrix.at['active_power', 'active_power'] = 1.0
        self.control_output_matrix.at['reactive_power', 'reactive_power'] = 1.0
        self.disturbance_output_matrix = (
            pd.DataFrame(0.0, index=self.output_names, columns=self.disturbance_names)
        )

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = (
            pd.DataFrame(0.0, index=self.active_power_nominal_timeseries.index, columns=self.disturbance_names)
        )

        # Construct output constraint timeseries
        # TODO: Fix offset of accumulated energy constraints.
        self.output_maximum_timeseries = (
            pd.concat([
                (
                    accumulated_energy_nominal_timeseries
                    - accumulated_energy_nominal_timeseries[int(flexible_load['time_period_power_shift_maximum'])]
                ),
                (
                    (1.0 - flexible_load['power_decrease_percentage_maximum'])
                    * self.active_power_nominal_timeseries
                ),
                (
                    (1.0 - flexible_load['power_decrease_percentage_maximum'])
                    * self.reactive_power_nominal_timeseries
                )
            ], axis='columns')
        )
        self.output_minimum_timeseries = (
            pd.concat([
                (
                    accumulated_energy_nominal_timeseries
                    + accumulated_energy_nominal_timeseries[int(flexible_load['time_period_power_shift_maximum'])]
                ),
                (
                    (1.0 + flexible_load['power_increase_percentage_maximum'])
                    * self.active_power_nominal_timeseries
                ),
                (
                    (1.0 + flexible_load['power_increase_percentage_maximum'])
                    * self.reactive_power_nominal_timeseries
                )
            ], axis='columns')
        )
