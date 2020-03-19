"""Distributed energy resource (DER) models."""

from multimethod import multimethod
import numpy as np
import pandas as pd
import pyomo.core
import pyomo.environ as pyo
import typing

import fledge.config
import fledge.database_interface
import fledge.electric_grid_models
import fledge.thermal_grid_models
import fledge.utils

logger = fledge.config.get_logger(__name__)


class DERModel(object):
    """DER model object."""

    der_name: str
    timesteps: pd.Index
    active_power_nominal_timeseries: pd.Series
    reactive_power_nominal_timeseries: pd.Series


class FixedDERModel(DERModel):
    """Fixed DER model object."""

    def define_optimization_variables(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
    ):
        pass

    def define_optimization_constraints(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel
    ):
        pass

    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            power_flow_solution: fledge.electric_grid_models.PowerFlowSolution,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault
    ):

        # Obtain DER index.
        der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=self.der_name))
        der = electric_grid_model.ders[der_index]

        # Define connection constraints.
        if optimization_problem.find_component('der_connection_constraints') is None:
            optimization_problem.der_connection_constraints = pyo.ConstraintList()
        for timestep in self.timesteps:
            optimization_problem.der_connection_constraints.add(
                optimization_problem.der_active_power_vector_change[timestep, der]
                ==
                self.active_power_nominal_timeseries.at[timestep]
                - np.real(
                    power_flow_solution.der_power_vector[der_index]
                )
            )
            optimization_problem.der_connection_constraints.add(
                optimization_problem.der_reactive_power_vector_change[timestep, der]
                ==
                self.reactive_power_nominal_timeseries.at[timestep]
                - np.imag(
                    power_flow_solution.der_power_vector[der_index]
                )
            )

    def get_optimization_results(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel
    ):
        return None


class FixedLoadModel(FixedDERModel):
    """Fixed load model object."""

    def __init__(
            self,
            der_data: fledge.database_interface.DERData,
            der_name: str
    ):
        """Construct fixed load model object by `der_data` and `der_name`."""

        # Store DER name.
        self.der_name = der_name

        # Get fixed load data by `der_name`.
        fixed_load = der_data.fixed_loads.loc[self.der_name, :]

        # Store timesteps index.
        self.timesteps = der_data.fixed_load_timeseries_dict[fixed_load['timeseries_name']].index

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            der_data.fixed_load_timeseries_dict[
                fixed_load['timeseries_name']
            ]['apparent_power_per_unit'].rename('active_power')
            * fixed_load['scaling_factor']
            * fixed_load['active_power']
        )
        self.reactive_power_nominal_timeseries = (
            der_data.fixed_load_timeseries_dict[
                fixed_load['timeseries_name']
            ]['apparent_power_per_unit'].rename('reactive_power')
            * fixed_load['scaling_factor']
            * fixed_load['reactive_power']
        )


class EVChargerModel(FixedDERModel):
    """EV charger model object."""

    def __init__(
            self,
            der_data: fledge.database_interface.DERData,
            der_name: str
    ):
        """Construct EV charger model object by `der_data` and `der_name`."""

        # Store DER name.
        self.der_name = der_name

        # Get fixed load data by `der_name`.
        ev_charger = der_data.ev_chargers.loc[self.der_name, :]

        # Store timesteps index.
        self.timesteps = der_data.ev_charger_timeseries_dict[ev_charger['timeseries_name']].index

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            der_data.ev_charger_timeseries_dict[
                ev_charger['timeseries_name']
            ]['apparent_power_per_unit'].rename('active_power')
            * ev_charger['scaling_factor']
            * ev_charger['active_power']
        )
        self.reactive_power_nominal_timeseries = (
            der_data.ev_charger_timeseries_dict[
                ev_charger['timeseries_name']
            ]['apparent_power_per_unit'].rename('reactive_power')
            * ev_charger['scaling_factor']
            * ev_charger['reactive_power']
        )


class FlexibleDERModel(DERModel):
    """Flexible DER model, e.g., flexible load, object."""

    state_names: pd.Index
    control_names: pd.Index
    disturbance_names: pd.Index
    output_names: pd.Index
    state_vector_initial: pd.Series
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
        optimization_problem.state_vector = pyo.Var(self.timesteps, [self.der_name], self.state_names)
        optimization_problem.control_vector = pyo.Var(self.timesteps, [self.der_name], self.control_names)
        optimization_problem.output_vector = pyo.Var(self.timesteps, [self.der_name], self.output_names)

    def define_optimization_constraints(
        self,
        optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel
    ):

        # Define shorthand for indexing 't+1'.
        # TODO: Is inferring timestep_interval from timesteps guaranteed to work?
        timestep_interval = self.timesteps[1] - self.timesteps[0]

        # Define constraints.
        if optimization_problem.find_component('flexible_der_model_constraints') is None:
            optimization_problem.flexible_der_model_constraints = pyo.ConstraintList()

        # Initial state.
        for state_name in self.state_names:
            optimization_problem.flexible_der_model_constraints.add(
                optimization_problem.state_vector[self.timesteps[0], self.der_name, state_name]
                ==
                self.state_vector_initial.at[state_name]
            )

        for timestep in self.timesteps[:-1]:

            # State equation.
            for state_name in self.state_names:
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.state_vector[timestep + timestep_interval, self.der_name, state_name]
                    ==
                    sum(
                        self.state_matrix.at[state_name, state_name_other]
                        * optimization_problem.state_vector[timestep, self.der_name, state_name_other]
                        for state_name_other in self.state_names
                    )
                    + sum(
                        self.control_matrix.at[state_name, control_name]
                        * optimization_problem.control_vector[timestep, self.der_name, control_name]
                        for control_name in self.control_names
                    )
                    + sum(
                        self.disturbance_matrix.at[state_name, disturbance_name]
                        * self.disturbance_timeseries.at[timestep, disturbance_name]
                        for disturbance_name in self.disturbance_names
                    )
                )

        for timestep in self.timesteps:

            # Output equation.
            for output_name in self.output_names:
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.output_vector[timestep, self.der_name, output_name]
                    ==
                    sum(
                        self.state_output_matrix.at[output_name, state_name]
                        * optimization_problem.state_vector[timestep, self.der_name, state_name]
                        for state_name in self.state_names
                    )
                    + sum(
                        self.control_output_matrix.at[output_name, control_name]
                        * optimization_problem.control_vector[timestep, self.der_name, control_name]
                        for control_name in self.control_names
                    )
                    + sum(
                        self.disturbance_output_matrix.at[output_name, disturbance_name]
                        * self.disturbance_timeseries.at[timestep, disturbance_name]
                        for disturbance_name in self.disturbance_names
                    )
                )

            # Output limits.
            for output_name in self.output_names:
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.output_vector[timestep, self.der_name, output_name]
                    >=
                    self.output_minimum_timeseries.at[timestep, output_name]
                )
                optimization_problem.flexible_der_model_constraints.add(
                    optimization_problem.output_vector[timestep, self.der_name, output_name]
                    <=
                    self.output_maximum_timeseries.at[timestep, output_name]
                )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            power_flow_solution: fledge.electric_grid_models.PowerFlowSolution,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            thermal_power_flow_solution: fledge.thermal_grid_models.ThermalPowerFlowSolution,
            thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel,
    ):

        # Connect electric grid.
        self.define_optimization_connection_grid(
            optimization_problem,
            power_flow_solution,
            electric_grid_model,
            disconnect_thermal_grid=False
        )

        # Connect thermal grid.
        self.define_optimization_connection_grid(
            optimization_problem,
            thermal_power_flow_solution,
            thermal_grid_model,
            disconnect_electric_grid=False
        )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            power_flow_solution: fledge.electric_grid_models.PowerFlowSolution,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            disconnect_thermal_grid=True
    ):

        # Obtain DER index.
        der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=self.der_name))
        der = electric_grid_model.ders[der_index]

        # Define connection constraints.
        if optimization_problem.find_component('der_connection_constraints') is None:
            optimization_problem.der_connection_constraints = pyo.ConstraintList()
        for timestep in self.timesteps:
            optimization_problem.der_connection_constraints.add(
                optimization_problem.der_active_power_vector_change[timestep, der]
                ==
                optimization_problem.output_vector[timestep, self.der_name, 'active_power']
                - np.real(
                    power_flow_solution.der_power_vector[der_index]
                )
            )
            optimization_problem.der_connection_constraints.add(
                optimization_problem.der_reactive_power_vector_change[timestep, der]
                ==
                optimization_problem.output_vector[timestep, self.der_name, 'reactive_power']
                - np.imag(
                    power_flow_solution.der_power_vector[der_index]
                )
            )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            thermal_power_flow_solution: fledge.thermal_grid_models.ThermalPowerFlowSolution,
            thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel,
            disconnect_electric_grid=True
    ):
        pass

    def define_optimization_objective(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            price_timeseries: pd.DataFrame
    ):

        # Define objective.
        if optimization_problem.find_component('objective') is None:
            optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
        optimization_problem.objective.expr += (
            sum(
                -1.0
                * price_timeseries.at[timestep, 'price_value']
                * optimization_problem.output_vector[timestep, self.der_name, 'active_power']
                for timestep in self.timesteps
            )
        )

    def get_optimization_results(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel
    ):

        # Instantiate results variables.
        state_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.state_names)
        control_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.control_names)
        output_vector = pd.DataFrame(0.0, index=self.timesteps, columns=self.output_names)

        # Obtain results.
        for timestep in self.timesteps:
            for state_name in self.state_names:
                state_vector.at[timestep, state_name] = (
                    optimization_problem.state_vector[timestep, self.der_name, state_name].value
                )
            for control_name in self.control_names:
                control_vector.at[timestep, control_name] = (
                    optimization_problem.control_vector[timestep, self.der_name, control_name].value
                )
            for output_name in self.output_names:
                output_vector.at[timestep, output_name] = (
                    optimization_problem.output_vector[timestep, self.der_name, output_name].value
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
            der_data: fledge.database_interface.DERData,
            der_name: str
    ):
        """Construct flexible load model object by `der_data` and `der_name`."""

        # Store DER name.
        self.der_name = der_name

        # Get flexible load data by `der_name`.
        flexible_load = der_data.flexible_loads.loc[der_name, :]

        # Store timesteps index.
        self.timesteps = der_data.flexible_load_timeseries_dict[flexible_load['timeseries_name']].index

        # Construct active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            der_data.flexible_load_timeseries_dict[
                flexible_load['timeseries_name']
            ]['apparent_power_per_unit'].rename('active_power')
            * flexible_load['scaling_factor']
            * flexible_load['active_power']
        )
        self.reactive_power_nominal_timeseries = (
            der_data.flexible_load_timeseries_dict[
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

        # Instantiate initial state.
        self.state_vector_initial = (
            pd.Series(0.0, index=self.state_names)
        )

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


class FlexibleBuildingModel(FlexibleDERModel):
    """Flexible load model object."""

    power_factor_nominal: np.float

    def __init__(
            self,
            der_data: fledge.database_interface.DERData,
            der_name: str
    ):
        """Construct flexible building model object by `der_data` and `der_name`."""

        # Store DER name.
        self.der_name = der_name

        # Obtain shorthands for flexible building data and model by `der_name`.
        flexible_building = der_data.flexible_buildings.loc[der_name, :]
        flexible_building_model = der_data.flexible_building_model_dict[flexible_building['model_name']]

        # Store timesteps.
        self.timesteps = flexible_building_model.set_timesteps

        # Obtain nominal power factor.
        self.power_factor_nominal = (
            np.cos(np.arctan(
                flexible_building['reactive_power']
                / flexible_building['active_power']
            ))
        )

        # Construct nominal active and reactive power timeseries.
        self.active_power_nominal_timeseries = (
            pd.Series(
                1.0,
                index=self.timesteps
            )
            * flexible_building['active_power']
        )
        self.reactive_power_nominal_timeseries = (
            pd.Series(
                1.0,
                index=self.timesteps
            )
            * flexible_building['reactive_power']
        )

        # Obtain indexes.
        self.state_names = flexible_building_model.set_states
        self.control_names = flexible_building_model.set_controls
        self.disturbance_names = flexible_building_model.set_disturbances
        self.output_names = flexible_building_model.set_outputs

        # Obtain initial state.
        self.state_vector_initial = flexible_building_model.set_state_initial

        # Obtain state space matrices.
        self.state_matrix = flexible_building_model.state_matrix
        self.control_matrix = flexible_building_model.control_matrix
        self.disturbance_matrix = flexible_building_model.disturbance_matrix
        self.state_output_matrix = flexible_building_model.state_output_matrix
        self.control_output_matrix = flexible_building_model.control_output_matrix
        self.disturbance_output_matrix = flexible_building_model.disturbance_output_matrix

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = flexible_building_model.disturbance_timeseries

        # Obtain output constraint timeseries
        self.output_maximum_timeseries = flexible_building_model.output_constraint_timeseries_maximum
        self.output_minimum_timeseries = flexible_building_model.output_constraint_timeseries_minimum

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            power_flow_solution: fledge.electric_grid_models.PowerFlowSolution,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            thermal_power_flow_solution: fledge.thermal_grid_models.ThermalPowerFlowSolution,
            thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel,
    ):

        # Connect electric grid.
        self.define_optimization_connection_grid(
            optimization_problem,
            power_flow_solution,
            electric_grid_model,
            disconnect_thermal_grid=False
        )

        # Connect thermal grid.
        self.define_optimization_connection_grid(
            optimization_problem,
            thermal_power_flow_solution,
            thermal_grid_model,
            disconnect_electric_grid=False
        )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            power_flow_solution: fledge.electric_grid_models.PowerFlowSolution,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            disconnect_thermal_grid=True
    ):

        # Obtain DER index.
        der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=self.der_name))
        der = electric_grid_model.ders[der_index]

        # Define connection constraints.
        if optimization_problem.find_component('der_connection_constraints') is None:
            optimization_problem.der_connection_constraints = pyo.ConstraintList()
        for timestep in self.timesteps:
            optimization_problem.der_connection_constraints.add(
                optimization_problem.der_active_power_vector_change[timestep, der]
                ==
                -1.0 * optimization_problem.output_vector[timestep, self.der_name, 'grid_electric_power']
                - np.real(
                    power_flow_solution.der_power_vector[der_index]
                )
            )
            optimization_problem.der_connection_constraints.add(
                optimization_problem.der_reactive_power_vector_change[timestep, der]
                ==
                -1.0 * (
                    optimization_problem.output_vector[timestep, self.der_name, 'grid_electric_power']
                    * np.tan(np.arccos(self.power_factor_nominal))
                )
                - np.imag(
                    power_flow_solution.der_power_vector[der_index]
                )
            )

            # Disable thermal grid connection.
            if disconnect_thermal_grid:
                optimization_problem.der_connection_constraints.add(
                    0.0
                    ==
                    optimization_problem.output_vector[timestep, self.der_name, 'grid_thermal_power_cooling']
                )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            thermal_power_flow_solution: fledge.thermal_grid_models.ThermalPowerFlowSolution,
            thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel,
            disconnect_electric_grid=True
    ):

        # Obtain DER index.
        der_index = int(fledge.utils.get_index(thermal_grid_model.ders, der_name=self.der_name))
        der = thermal_grid_model.ders[der_index]

        # Define connection constraints.
        if optimization_problem.find_component('der_connection_constraints') is None:
            optimization_problem.der_connection_constraints = pyo.ConstraintList()
        for timestep in self.timesteps:
            optimization_problem.der_connection_constraints.add(
                optimization_problem.der_thermal_power_vector[timestep, der]
                ==
                -1.0 * optimization_problem.output_vector[timestep, self.der_name, 'grid_thermal_power_cooling']
            )

            # Disable electric grid connection.
            if disconnect_electric_grid:
                optimization_problem.der_connection_constraints.add(
                    0.0
                    ==
                    optimization_problem.output_vector[timestep, self.der_name, 'grid_electric_power']
                )

    def define_optimization_objective(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            price_timeseries: pd.DataFrame
    ):

        # Define objective.
        if optimization_problem.find_component('objective') is None:
            optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
        optimization_problem.objective.expr += (
            sum(
                price_timeseries.at[timestep, 'price_value']
                * optimization_problem.output_vector[timestep, self.der_name, 'grid_electric_power']
                for timestep in self.timesteps
            )
        )


class DERModelSet(object):
    """DER model set object."""

    timesteps: pd.Index
    der_names: pd.Index
    fixed_der_names: pd.Index
    flexible_der_names: pd.Index
    der_models: typing.Dict[str, DERModel]
    fixed_der_models: typing.Dict[str, FixedDERModel]
    flexible_der_models: typing.Dict[str, FlexibleDERModel]

    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain data.
        scenario_data = fledge.database_interface.ScenarioData(scenario_name)
        der_data = fledge.database_interface.DERData(scenario_name)

        # Obtain timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain DER names.
        self.der_names = (
            pd.Index(pd.concat([
                der_data.fixed_loads['der_name'],
                der_data.ev_chargers['der_name'],
                der_data.flexible_loads['der_name'],
                der_data.flexible_buildings['der_name']
            ]))
        )
        self.fixed_der_names = (
            pd.Index(pd.concat([
                der_data.fixed_loads['der_name'],
                der_data.ev_chargers['der_name'],
            ]))
        )
        self.flexible_der_names = (
            pd.Index(pd.concat([
                der_data.flexible_loads['der_name'],
                der_data.flexible_buildings['der_name']
            ]))
        )

        # Obtain models.
        self.der_models = dict.fromkeys(self.der_names)
        self.fixed_der_models = dict.fromkeys(self.fixed_der_names)
        self.flexible_der_models = dict.fromkeys(self.flexible_der_names)
        for der_name in self.der_names:
            if der_name in der_data.fixed_loads['der_name']:
                self.der_models[der_name] = self.fixed_der_models[der_name] = (
                    fledge.der_models.FixedLoadModel(
                        der_data,
                        der_name
                    )
                )
            elif der_name in der_data.ev_chargers['der_name']:
                self.der_models[der_name] = self.fixed_der_models[der_name] = (
                    fledge.der_models.EVChargerModel(
                        der_data,
                        der_name
                    )
                )
            elif der_name in der_data.flexible_loads['der_name']:
                self.der_models[der_name] = self.flexible_der_models[der_name] = (
                    fledge.der_models.FlexibleLoadModel(
                        der_data,
                        der_name
                    )
                )
            elif der_name in der_data.flexible_buildings['der_name']:
                self.der_models[der_name] = self.flexible_der_models[der_name] = (
                    fledge.der_models.FlexibleBuildingModel(
                        der_data,
                        der_name
                    )
                )
            else:
                logger.error(f"Cannot determine type of DER: {der_name}")
                raise ValueError

    def define_optimization_variables(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel
    ):

        # Define flexible DER variables.
        der_state_names = [
            (der_name, state_name)
            for der_name in self.flexible_der_names
            for state_name in self.flexible_der_models[der_name].state_names
        ]
        der_control_names = [
            (der_name, control_name)
            for der_name in self.flexible_der_names
            for control_name in self.flexible_der_models[der_name].control_names
        ]
        der_output_names = [
            (der_name, output_name)
            for der_name in self.flexible_der_names
            for output_name in self.flexible_der_models[der_name].output_names
        ]
        optimization_problem.state_vector = pyo.Var(self.timesteps, der_state_names)
        optimization_problem.control_vector = pyo.Var(self.timesteps, der_control_names)
        optimization_problem.output_vector = pyo.Var(self.timesteps, der_output_names)

    def define_optimization_constraints(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel
    ):

        # Define DER constraints, only for flexible DERs.
        for der_name in self.flexible_der_names:
            self.flexible_der_models[der_name].define_optimization_constraints(
                optimization_problem
            )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            power_flow_solution: fledge.electric_grid_models.PowerFlowSolution,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            thermal_power_flow_solution: fledge.thermal_grid_models.ThermalPowerFlowSolution,
            thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel,
    ):

        # Define constraints for the connection with the DER power vector of the grid.
        for der_name in self.der_names:
            self.der_models[der_name].define_optimization_connection_grid(
                optimization_problem,
                power_flow_solution,
                electric_grid_model,
                thermal_power_flow_solution,
                thermal_grid_model
            )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            power_flow_solution: fledge.electric_grid_models.PowerFlowSolution,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            **kwargs
    ):

        # Define constraints for the connection with the DER power vector of the grid.
        for der_name in self.der_names:
            self.der_models[der_name].define_optimization_connection_grid(
                optimization_problem,
                power_flow_solution,
                electric_grid_model,
                **kwargs
            )

    @multimethod
    def define_optimization_connection_grid(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            thermal_power_flow_solution: fledge.thermal_grid_models.ThermalPowerFlowSolution,
            thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel,
            **kwargs
    ):

        # Define constraints for the connection with the DER power vector of the grid.
        for der_name in self.der_names:
            self.der_models[der_name].define_optimization_connection_grid(
                optimization_problem,
                thermal_power_flow_solution,
                thermal_grid_model,
                **kwargs
            )

    def define_optimization_objective(
            self,
            optimization_problem: pyomo.core.base.PyomoModel.ConcreteModel,
            price_timeseries: pd.DataFrame
    ):

        # Define objective, only for flexible DERs.
        for der_name in self.flexible_der_names:
            self.flexible_der_models[der_name].define_optimization_objective(
                optimization_problem,
                price_timeseries
            )
