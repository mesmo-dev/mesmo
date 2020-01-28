"""Distributed energy resource (DER) models."""

from multimethod import multimethod
import numpy as np
import pandas as pd
import pyomo.core
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface

logger = fledge.config.get_logger(__name__)


class DERModel(object):
    """DER model object."""

    der_name: str
    active_power_nominal_timeseries: pd.Series
    reactive_power_nominal_timeseries: pd.Series

    def __init__(
            self,
            der_name: str
    ):

        # Store DER name.
        self.der_name = der_name


class FixedLoadModel(DERModel):
    """Fixed load model object."""

    def __init__(
            self,
            fixed_load_data: fledge.database_interface.FixedLoadData,
            der_name: str
    ):
        """Construct fixed load model object by `fixed_load_data` and `der_name`."""

        # Initialize inherited properties.
        super(FixedLoadModel, self).__init__(der_name)

        # Get fixed load data by `der_name`.
        fixed_load = fixed_load_data.fixed_loads.loc[self.der_name, :]

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

        # Initialize inherited properties.
        super(EVChargerModel, self).__init__(der_name)

        # Get fixed load data by `der_name`.
        ev_charger = ev_charger_data.ev_chargers.loc[self.der_name, :]

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

    timesteps: pd.Index
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

    def define_optimization_variables(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
    ):

        # Define variables.
        optimization_problem.state_vector = pyo.Var(self.timesteps, self.state_names)
        optimization_problem.control_vector = pyo.Var(self.timesteps, self.control_names)
        optimization_problem.output_vector = pyo.Var(self.timesteps, self.output_names)

    def define_optimization_constraints(
        self,
        optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel
    ):

        # Define shorthand for indexing 't+1'.
        # TODO: Is inferring timestep_interval from timesteps guaranteed to work?
        timestep_interval = self.timesteps[1] - self.timesteps[0]

        # Define constraints.
        optimization_problem.flexible_der_model_constraints = pyo.ConstraintList()

        # Initial state.
        # TODO: Define initial state in model.
        for state_name in self.state_names:
            optimization_problem.flexible_der_model_constraints.add(
                optimization_problem.state_vector[self.timesteps[0], state_name]
                ==
                0.0
            )

        for timestep in self.timesteps[:-1]:

            # State equation.
            for state_name in self.state_names:
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.state_vector[timestep + timestep_interval, state_name]
                    ==
                    pyo.quicksum(
                        self.state_matrix.at[state_name, state_name_other]
                        * optimization_problem.state_vector[timestep, state_name_other]
                        for state_name_other in self.state_names
                    )
                    + pyo.quicksum(
                        self.control_matrix.at[state_name, control_name]
                        * optimization_problem.control_vector[timestep, control_name]
                        for control_name in self.control_names
                    )
                    + pyo.quicksum(
                        self.disturbance_matrix.at[state_name, disturbance_name]
                        * self.disturbance_timeseries.at[timestep, disturbance_name]
                        for disturbance_name in self.disturbance_names
                    )
                )

        for timestep in self.timesteps:

            # Output equation.
            for output_name in self.output_names:
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.output_vector[timestep, output_name]
                    ==
                    pyo.quicksum(
                        self.state_output_matrix.at[output_name, state_name]
                        * optimization_problem.state_vector[timestep, state_name]
                        for state_name in self.state_names
                    )
                    + pyo.quicksum(
                        self.control_output_matrix.at[output_name, control_name]
                        * optimization_problem.control_vector[timestep, control_name]
                        for control_name in self.control_names
                    )
                    + pyo.quicksum(
                        self.disturbance_output_matrix.at[output_name, disturbance_name]
                        * self.disturbance_timeseries.at[timestep, disturbance_name]
                        for disturbance_name in self.disturbance_names
                    )
                )

            # Output limits.
            for output_name in self.output_names:
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.output_vector[timestep, output_name]
                    >=
                    self.output_minimum_timeseries.at[timestep, output_name]
                )
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.output_vector[timestep, output_name]
                    <=
                    self.output_maximum_timeseries.at[timestep, output_name]
                )

    def get_optimization_results(
            self,
            optimization_problem
    ):

        # Instantiate results variables.
        state_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.state_names)
        control_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.control_names)
        output_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.output_names)

        # Obtain results.
        for timestep in self.timesteps:
            for state_name in self.state_names:
                state_vector.at[timestep, state_name] = (
                    optimization_problem.state_vector[timestep, state_name].value
                )
            for control_name in self.control_names:
                control_vector.at[timestep, control_name] = (
                    optimization_problem.control_vector[timestep, control_name].value
                )
            for output_name in self.output_names:
                output_vector.at[timestep, output_name] = (
                    optimization_problem.output_vector[timestep, output_name].value
                )

        return (
            state_vector,
            control_vector,
            output_vector
        )


class FlexibleLoadModel(FlexibleDERModel):
    """Flexible load model object."""

    def __init__(
            self,
            flexible_load_data: fledge.database_interface.FlexibleLoadData,
            der_name: str
    ):
        """Construct flexible load model object by `flexible_load_data` and `der_name`."""

        # Initialize inherited properties.
        super(FlexibleDERModel, self).__init__(der_name)

        # Get flexible load data by `der_name`.
        flexible_load = flexible_load_data.flexible_loads.loc[der_name, :]

        # Store timesteps index.
        self.timesteps = flexible_load_data.flexible_load_timeseries_dict[flexible_load['timeseries_name']].index

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
        self.output_names = pd.Index(['accumulated_energy', 'active_power', 'reactive_power', 'power_factor_constant'])

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
        self.control_output_matrix.at['power_factor_constant', 'active_power'] = -1.0 / flexible_load['active_power']
        self.control_output_matrix.at['power_factor_constant', 'reactive_power'] = 1.0 / flexible_load['reactive_power']
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
                ),
                pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='power_factor_constant')
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
                ),
                pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='power_factor_constant')
            ], axis='columns')
        )
