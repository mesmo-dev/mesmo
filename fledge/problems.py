"""Problems module for mathematical optimization and simulation problem type definitions."""

import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.thermal_grid_models
import fledge.utils

logger = fledge.config.get_logger(__name__)


class NominalOperationProblem(object):
    """Nominal operation problem object, consisting of the corresponding electric / thermal grid models,
    reference power flow solutions and DER model set for the given scenario.

    - The nominal operation problem (alias: simulation problem, power flow problem)
      represents the simulation problem of the DERs and grids considering the nominal operation schedule for all DERs.
    - The problem formulation is able to consider combined as well as individual operation of
      thermal and electric grids.
    """

    scenario_name: str
    timesteps: pd.Index
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: fledge.electric_grid_models.PowerFlowSolution = None
    thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None
    thermal_power_flow_solution_reference: fledge.thermal_grid_models.ThermalPowerFlowSolution = None
    der_model_set: fledge.der_models.DERModelSet
    results: fledge.data_interface.ResultsDict

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain data.
        scenario_data = fledge.data_interface.ScenarioData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['electric_grid_name']):
            start_time = fledge.utils.log_timing_start("electric grid model instantiation", logger)
            self.electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
            self.power_flow_solution_reference = (
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint(self.electric_grid_model)
            )
            fledge.utils.log_timing_end(start_time, "electric grid model instantiation", logger)

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['thermal_grid_name']):
            start_time = fledge.utils.log_timing_start("thermal grid model instantiation", logger)
            self.thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
            self.thermal_power_flow_solution_reference = (
                fledge.thermal_grid_models.ThermalPowerFlowSolution(self.thermal_grid_model)
            )
            fledge.utils.log_timing_end(start_time, "thermal grid model instantiation", logger)

        # Obtain DER model set.
        start_time = fledge.utils.log_timing_start("DER model instantiation", logger)
        self.der_model_set = fledge.der_models.DERModelSet(scenario_name)
        fledge.utils.log_timing_end(start_time, "DER model instantiation", logger)

    def solve(self):

        # Instantiate results variables.
        if self.electric_grid_model is not None:
            der_power_vector = (
                pd.DataFrame(columns=self.electric_grid_model.ders, index=self.timesteps, dtype=np.complex)
            )
            node_voltage_vector = (
                pd.DataFrame(columns=self.electric_grid_model.nodes, index=self.timesteps, dtype=np.complex)
            )
            branch_power_vector_1 = (
                pd.DataFrame(columns=self.electric_grid_model.branches, index=self.timesteps, dtype=np.complex)
            )
            branch_power_vector_2 = (
                pd.DataFrame(columns=self.electric_grid_model.branches, index=self.timesteps, dtype=np.complex)
            )
            loss = pd.DataFrame(columns=['total'], index=self.timesteps, dtype=np.complex)
        if self.thermal_grid_model is not None:
            der_thermal_power_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.timesteps, dtype=np.float)
            )
            der_flow_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.timesteps, dtype=np.float)
            )
            branch_flow_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.branches, index=self.timesteps, dtype=np.float)
            )
            node_head_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.timesteps, dtype=np.float)
            )

        # Obtain nominal DER power vector.
        if self.electric_grid_model is not None:
            for der in self.electric_grid_model.ders:
                # TODO: Use ders instead of der_names for der_models index.
                der_name = der[1]
                der_power_vector.loc[:, der] = (
                    self.der_model_set.der_models[der_name].active_power_nominal_timeseries
                    + (1.0j * self.der_model_set.der_models[der_name].reactive_power_nominal_timeseries)
                )
        if self.thermal_grid_model is not None:
            for der in self.electric_grid_model.ders:
                der_name = der[1]
                der_thermal_power_vector.loc[:, der] = (
                    self.der_model_set.der_models[der_name].thermal_power_nominal_timeseries
                )

        # Solve power flow.
        start_time = fledge.utils.log_timing_start("power flow solution", logger)
        if self.electric_grid_model is not None:
            power_flow_solutions = (
                fledge.utils.starmap(
                    fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
                    zip(
                        itertools.repeat(self.electric_grid_model),
                        der_power_vector.values
                    )
                )
            )
            power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))
        if self.thermal_grid_model is not None:
            thermal_power_flow_solutions = (
                fledge.utils.starmap(
                    fledge.thermal_grid_models.ThermalPowerFlowSolution,
                    [(self.thermal_grid_model, row) for row in der_thermal_power_vector.values]
                )
            )
            thermal_power_flow_solutions = dict(zip(self.timesteps, thermal_power_flow_solutions))
        fledge.utils.log_timing_end(start_time, "power flow solution", logger)

        # Obtain results.
        if self.electric_grid_model is not None:
            for timestep in self.timesteps:
                power_flow_solution = power_flow_solutions[timestep]
                # TODO: Flatten power flow solution arrays.
                node_voltage_vector.loc[timestep, :] = power_flow_solution.node_voltage_vector
                branch_power_vector_1.loc[timestep, :] = power_flow_solution.branch_power_vector_1
                branch_power_vector_2.loc[timestep, :] = power_flow_solution.branch_power_vector_2
                loss.loc[timestep, :] = power_flow_solution.loss
        if self.thermal_grid_model is not None:
            for timestep in self.timesteps:
                thermal_power_flow_solution = thermal_power_flow_solutions[timestep]
                der_flow_vector.loc[timestep, :] = thermal_power_flow_solution.der_flow_vector
                branch_flow_vector.loc[timestep, :] = thermal_power_flow_solution.branch_flow_vector
                node_head_vector.loc[timestep, :] = thermal_power_flow_solution.node_head_vector

        # Obtain magnitude values.
        if self.electric_grid_model is not None:
            der_power_magnitude = np.abs(der_power_vector)
            node_voltage_magnitude = np.abs(node_voltage_vector)
            branch_power_1_magnitude = np.abs(branch_power_vector_1)
            branch_power_2_magnitude = np.abs(branch_power_vector_2)
            loss_magnitude = np.abs(loss)

        # Obtain per-unit values.
        if self.electric_grid_model is not None:
            der_power_magnitude_per_unit = (
                der_power_magnitude
                / np.abs(self.electric_grid_model.der_power_vector_reference)
            )
            node_voltage_magnitude_per_unit = (
                node_voltage_magnitude
                / np.abs(self.electric_grid_model.node_voltage_vector_reference)
            )
            branch_power_1_magnitude_per_unit = (
                branch_power_1_magnitude
                / np.abs(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
            branch_power_2_magnitude_per_unit = (
                branch_power_2_magnitude
                / np.abs(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
        if self.thermal_grid_model is not None:
            pass
            # TODO: Define thermal grid reference properties.

        # Store results.
        self.results = fledge.data_interface.ResultsDict()
        if self.electric_grid_model is not None:
            self.results.update(
                fledge.data_interface.ResultsDict(
                    der_power_vector=der_power_vector,
                    node_voltage_vector=node_voltage_vector,
                    branch_power_vector_1=branch_power_vector_1,
                    branch_power_vector_2=branch_power_vector_2,
                    loss=loss,
                    der_power_magnitude=der_power_magnitude,
                    node_voltage_magnitude=node_voltage_magnitude,
                    branch_power_1_magnitude=branch_power_1_magnitude,
                    branch_power_2_magnitude=branch_power_2_magnitude,
                    loss_magnitude=loss_magnitude,
                    der_power_magnitude_per_unit=der_power_magnitude_per_unit,
                    node_voltage_magnitude_per_unit=node_voltage_magnitude_per_unit,
                    branch_power_1_magnitude_per_unit=branch_power_1_magnitude_per_unit,
                    branch_power_2_magnitude_per_unit=branch_power_2_magnitude_per_unit
                )
            )
        if self.thermal_grid_model is not None:
            self.results.update(
                fledge.data_interface.ResultsDict(
                    der_flow_vector=der_flow_vector,
                    branch_flow_vector=branch_flow_vector,
                    node_head_vector=node_head_vector,
                )
            )

    def get_results(self):

        return self.results


class OptimalOperationProblem(object):
    """Optimal operation problem object, consisting of an optimization problem as well as the corresponding
    electric / thermal grid models, reference power flow solutions, linear grid models and DER model set
    for the given scenario.

    - The optimal operation problem (alias: optimal dispatch problem, optimal power flow problem)
      formulates the optimization problem for minimizing the objective functions of DERs and grid operators
      subject to the model constraints of all DERs and grids.
    - The problem formulation is able to consider combined as well as individual operation of
      thermal and electric grids.
    """

    scenario_name: str
    timesteps: pd.Index
    price_data: fledge.data_interface.PriceData
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: fledge.electric_grid_models.PowerFlowSolution = None
    linear_electric_grid_model: fledge.electric_grid_models.LinearElectricGridModel = None
    thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None
    thermal_power_flow_solution_reference: fledge.thermal_grid_models.ThermalPowerFlowSolution = None
    linear_thermal_grid_model: fledge.thermal_grid_models.LinearThermalGridModel = None
    der_model_set: fledge.der_models.DERModelSet
    optimization_problem: fledge.utils.OptimizationProblem

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain data.
        scenario_data = fledge.data_interface.ScenarioData(scenario_name)
        self.price_data = fledge.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['electric_grid_name']):
            self.electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
            self.power_flow_solution_reference = (
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint(self.electric_grid_model)
            )
            self.linear_electric_grid_model = (
                fledge.electric_grid_models.LinearElectricGridModelGlobal(
                    self.electric_grid_model,
                    self.power_flow_solution_reference
                )
            )

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['thermal_grid_name']):
            self.thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
            self.thermal_grid_model.energy_transfer_station_head_loss = 0.0  # TODO: Remove modifications.
            self.thermal_grid_model.cooling_plant_efficiency = 10.0  # TODO: Remove modifications.
            self.thermal_power_flow_solution_reference = (
                fledge.thermal_grid_models.ThermalPowerFlowSolution(self.thermal_grid_model)
            )
            self.linear_thermal_grid_model = (
                fledge.thermal_grid_models.LinearThermalGridModel(
                    self.thermal_grid_model,
                    self.thermal_power_flow_solution_reference
                )
            )

        # Obtain DER model set.
        self.der_model_set = fledge.der_models.DERModelSet(scenario_name)

        # Instantiate optimization problem.
        self.optimization_problem = fledge.utils.OptimizationProblem()

        # Define linear electric grid model variables and constraints.
        if self.electric_grid_model is not None:
            self.linear_electric_grid_model.define_optimization_variables(
                self.optimization_problem,
                self.timesteps
            )
            node_voltage_magnitude_vector_minimum = (
                scenario_data.scenario['voltage_per_unit_minimum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                if pd.notnull(scenario_data.scenario['voltage_per_unit_minimum'])
                else None
            )
            node_voltage_magnitude_vector_maximum = (
                scenario_data.scenario['voltage_per_unit_maximum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                if pd.notnull(scenario_data.scenario['voltage_per_unit_maximum'])
                else None
            )
            branch_power_magnitude_vector_maximum = (
                scenario_data.scenario['branch_flow_per_unit_maximum']
                * self.electric_grid_model.branch_power_vector_magnitude_reference
                if pd.notnull(scenario_data.scenario['branch_flow_per_unit_maximum'])
                else None
            )
            self.linear_electric_grid_model.define_optimization_constraints(
                self.optimization_problem,
                self.timesteps,
                node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
                node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
                branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
            )

        # Define thermal grid model variables and constraints.
        if self.thermal_grid_model is not None:
            self.linear_thermal_grid_model.define_optimization_variables(
                self.optimization_problem,
                self.timesteps
            )
            node_head_vector_minimum = (
                scenario_data.scenario['node_head_per_unit_maximum']
                * self.thermal_power_flow_solution_reference.node_head_vector
                if pd.notnull(scenario_data.scenario['voltage_per_unit_maximum'])
                else None
            )
            branch_flow_vector_maximum = (
                scenario_data.scenario['pipe_flow_per_unit_maximum']
                * self.thermal_power_flow_solution_reference.branch_flow_vector
                if pd.notnull(scenario_data.scenario['pipe_flow_per_unit_maximum'])
                else None
            )
            self.linear_thermal_grid_model.define_optimization_constraints(
                self.optimization_problem,
                self.timesteps,
                node_head_vector_minimum=node_head_vector_minimum,
                branch_flow_vector_maximum=branch_flow_vector_maximum
            )

        # Define DER variables and constraints.
        self.der_model_set.define_optimization_variables(
            self.optimization_problem
        )
        self.der_model_set.define_optimization_constraints(
            self.optimization_problem,
            electric_grid_model=self.electric_grid_model,
            thermal_grid_model=self.thermal_grid_model
        )

        # Define objective.
        if self.thermal_grid_model is not None:
            self.linear_thermal_grid_model.define_optimization_objective(
                self.optimization_problem,
                self.price_data,
                self.timesteps
            )
        if self.electric_grid_model is not None:
            self.linear_electric_grid_model.define_optimization_objective(
                self.optimization_problem,
                self.price_data,
                self.timesteps
            )
        self.der_model_set.define_optimization_objective(
            self.optimization_problem,
            self.price_data,
            electric_grid_model=self.electric_grid_model,
            thermal_grid_model=self.thermal_grid_model
        )

    def solve(self):

        # Solve optimization problem.
        self.optimization_problem.solve()

    def get_results(
            self,
            in_per_unit=False,
            with_mean=False,
            get_dlmps=True
    ) -> fledge.data_interface.ResultsDict:

        # Instantiate results dictionary.
        results = fledge.data_interface.ResultsDict()

        # Obtain electric grid results.
        if self.electric_grid_model is not None:
            results.update(
                self.linear_electric_grid_model.get_optimization_results(
                    self.optimization_problem,
                    self.power_flow_solution_reference,
                    self.timesteps,
                    in_per_unit=in_per_unit,
                    with_mean=with_mean
                )
            )

        # Obtain thermal grid results.
        if self.thermal_grid_model is not None:
            results.update(
                self.linear_thermal_grid_model.get_optimization_results(
                    self.optimization_problem,
                    self.timesteps,
                    in_per_unit=in_per_unit,
                    with_mean=with_mean
                )
            )

        # Obtain DER results.
        results.update(
            self.der_model_set.get_optimization_results(
                self.optimization_problem
            )
        )

        if get_dlmps:

            # Obtain electric DLMPs.
            if self.electric_grid_model is not None:
                results.update(
                    self.linear_electric_grid_model.get_optimization_dlmps(
                        self.optimization_problem,
                        self.price_data,
                        self.timesteps
                    )
                )

            # Obtain thermal DLMPs.
            if self.thermal_grid_model is not None:
                results.update(
                    self.linear_thermal_grid_model.get_optimization_dlmps(
                        self.optimization_problem,
                        self.price_data,
                        self.timesteps
                    )
                )

        return results


class Results(object):
    """Results object."""

    # Models.
    electric_grid_model: fledge.electric_grid_models.ElectricGridModel = None
    thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None
    der_model_set: fledge.der_models.DERModelSet = None

    # Price input data.
    price_data: fledge.data_interface.PriceData = None

    # Electric grid result variables.
    der_power_vector: pd.DataFrame = None
    node_voltage_vector: pd.DataFrame = None
    branch_power_vector_1: pd.DataFrame = None
    branch_power_vector_2: pd.DataFrame = None
    loss: pd.DataFrame = None

    # Thermal grid result variables.
    der_thermal_power_vector: pd.DataFrame = None
    node_head_vector: pd.DataFrame = None
    branch_flow_vector: pd.DataFrame = None
    pump_power: pd.DataFrame = None

    # DER model result variables.
    state_vector: pd.DataFrame = None
    control_vector: pd.DataFrame = None
    output_vector: pd.DataFrame = None
    output_minimum_timeseries: pd.DataFrame = None
    output_maximum_timeseries: pd.DataFrame = None

    # Electric grid DLMPs.
    electric_grid_energy_dlmp_node_active_power: pd.DataFrame = None
    electric_grid_voltage_dlmp_node_active_power: pd.DataFrame = None
    electric_grid_congestion_dlmp_node_active_power: pd.DataFrame = None
    electric_grid_loss_dlmp_node_active_power: pd.DataFrame = None
    electric_grid_total_dlmp_node_active_power: pd.DataFrame = None
    electric_grid_voltage_dlmp_node_reactive_power: pd.DataFrame = None
    electric_grid_congestion_dlmp_node_reactive_power: pd.DataFrame = None
    electric_grid_loss_dlmp_node_reactive_power: pd.DataFrame = None
    electric_grid_energy_dlmp_node_reactive_power: pd.DataFrame = None
    electric_grid_total_dlmp_node_reactive_power: pd.DataFrame = None
    electric_grid_energy_dlmp_der_active_power: pd.DataFrame = None
    electric_grid_voltage_dlmp_der_active_power: pd.DataFrame = None
    electric_grid_congestion_dlmp_der_active_power: pd.DataFrame = None
    electric_grid_loss_dlmp_der_active_power: pd.DataFrame = None
    electric_grid_total_dlmp_der_active_power: pd.DataFrame = None
    electric_grid_voltage_dlmp_der_reactive_power: pd.DataFrame = None
    electric_grid_congestion_dlmp_der_reactive_power: pd.DataFrame = None
    electric_grid_loss_dlmp_der_reactive_power: pd.DataFrame = None
    electric_grid_energy_dlmp_der_reactive_power: pd.DataFrame = None
    electric_grid_total_dlmp_der_reactive_power: pd.DataFrame = None
    electric_grid_total_dlmp_price_timeseries: pd.DataFrame = None

    # Thermal grid DLMPs.
    thermal_grid_energy_dlmp_node_thermal_power: pd.DataFrame = None
    thermal_grid_head_dlmp_node_thermal_power: pd.DataFrame = None
    thermal_grid_congestion_dlmp_node_thermal_power: pd.DataFrame = None
    thermal_grid_pump_dlmp_node_thermal_power: pd.DataFrame = None
    thermal_grid_total_dlmp_node_thermal_power: pd.DataFrame = None
    thermal_grid_energy_dlmp_der_thermal_power: pd.DataFrame = None
    thermal_grid_head_dlmp_der_thermal_power: pd.DataFrame = None
    thermal_grid_congestion_dlmp_der_thermal_power: pd.DataFrame = None
    thermal_grid_pump_dlmp_der_thermal_power: pd.DataFrame = None
    thermal_grid_total_dlmp_der_thermal_power: pd.DataFrame = None
    thermal_grid_total_dlmp_price_timeseries: pd.DataFrame = None

    def __init__(
            self,
            **kwargs
    ):

        # Set all keyword arguments as attributes.
        for attribute_name in kwargs:
            self.__setattr__(attribute_name, kwargs[attribute_name])

    def __setattr__(self, attribute_name, value):

        # Assert that attribute name is valid.
        try:
            assert hasattr(Results, attribute_name)
        except AssertionError:
            logger.error(f"Cannot set invalid results object variable: {attribute_name}")
            raise

        # Set attribute value.
        super().__setattr__(attribute_name, value)

    def __getattribute__(self, attribute_name):

        # Obtain attribute value.
        value = super().__getattribute__(attribute_name)

        # Assert that attribute value has been set.
        try:
            assert value is not None
        except AssertionError:
            logger.error(f"Results variable '{attribute_name}' has no value / has not been set.")
            raise

        return value
