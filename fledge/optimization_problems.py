"""Optimization problems module."""

from multimethod import multimethod
import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.thermal_grid_models

logger = fledge.config.get_logger(__name__)


class OperationProblem(object):

    scenario_name: str
    timesteps: pd.Index
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault
    power_flow_solution_reference: fledge.electric_grid_models.PowerFlowSolution
    linear_electric_grid_model: fledge.electric_grid_models.LinearElectricGridModel
    thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel
    thermal_power_flow_solution_reference: fledge.thermal_grid_models.ThermalPowerFlowSolution
    linear_thermal_grid_model: fledge.thermal_grid_models.LinearThermalGridModel
    der_model_set: fledge.der_models.DERModelSet
    optimization_problem: pyo.ConcreteModel

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain data.
        scenario_data = fledge.database_interface.ScenarioData(scenario_name)
        price_data = fledge.database_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Store price timeseries.
        price_type = 'singapore_wholesale'
        self.price_timeseries = price_data.price_timeseries_dict[price_type]

        # Obtain models.
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
        self.thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
        self.thermal_grid_model.ets_head_loss = 0.0  # TODO: Remove modifications.
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
        self.der_model_set = fledge.der_models.DERModelSet(scenario_name)

        # Instantiate optimization problem.
        self.optimization_problem = pyo.ConcreteModel()

        # Define linear electric grid model variables.
        self.linear_electric_grid_model.define_optimization_variables(
            self.optimization_problem,
            self.timesteps
        )

        # Define linear electric grid model constraints.
        voltage_magnitude_vector_minimum = (
            scenario_data.scenario['voltage_per_unit_minimum']
            * np.abs(self.power_flow_solution_reference.node_voltage_vector)
            if pd.notnull(scenario_data.scenario['voltage_per_unit_minimum'])
            else None
        )
        voltage_magnitude_vector_maximum = (
            scenario_data.scenario['voltage_per_unit_maximum']
            * np.abs(self.power_flow_solution_reference.node_voltage_vector)
            if pd.notnull(scenario_data.scenario['voltage_per_unit_maximum'])
            else None
        )
        branch_power_vector_squared_maximum = (
            scenario_data.scenario['branch_flow_per_unit_maximum']
            * np.abs(self.power_flow_solution_reference.branch_power_vector_1 ** 2)
            if pd.notnull(scenario_data.scenario['branch_flow_per_unit_maximum'])
            else None
        )
        self.linear_electric_grid_model.define_optimization_constraints(
            self.optimization_problem,
            self.timesteps,
            voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
            voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
            branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
        )

        # Define thermal grid model variables.
        self.linear_thermal_grid_model.define_optimization_variables(
            self.optimization_problem,
            self.timesteps
        )

        # Define thermal grid model constraints.
        node_head_vector_minimum = 1.5 * self.thermal_power_flow_solution_reference.node_head_vector
        branch_flow_vector_maximum = 1.5 * self.thermal_power_flow_solution_reference.branch_flow_vector
        self.linear_thermal_grid_model.define_optimization_constraints(
            self.optimization_problem,
            self.timesteps,
            node_head_vector_minimum=node_head_vector_minimum,
            branch_flow_vector_maximum=branch_flow_vector_maximum
        )

        # Define DER variables.
        self.der_model_set.define_optimization_variables(
            self.optimization_problem
        )

        # Define DER constraints.
        self.der_model_set.define_optimization_constraints(
            self.optimization_problem
        )

        # Define constraints for the connection with the DER power vector of the electric and thermal grids.
        self.der_model_set.define_optimization_connection_grid(
            self.optimization_problem,
            self.power_flow_solution_reference,
            self.electric_grid_model,
            self.thermal_power_flow_solution_reference,
            self.thermal_grid_model
        )

        # Define objective.
        self.linear_thermal_grid_model.define_optimization_objective(
            self.optimization_problem,
            self.price_timeseries,
            self.timesteps
        )
        self.der_model_set.define_optimization_objective(
            self.optimization_problem,
            self.price_timeseries
        )

    def solve_optimization(self):

        # Solve optimization problem.
        self.optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
        optimization_result = (
            optimization_solver.solve(
                self.optimization_problem,
                tee=fledge.config.solver_output
            )
        )

        # Assert that solver exited with any solution. If not, raise an error.
        try:
            assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
        except AssertionError:
            logger.error(f"Solver termination condition: {optimization_result.solver.termination_condition}")
            raise

    def display_optimization_problem(self):

        self.optimization_problem.display()

    def get_optimization_results(
            self,
            in_per_unit=False,
            with_mean=False
    ):

        # Obtain results.
        (
            der_active_power_vector,
            der_reactive_power_vector,
            voltage_magnitude_vector,
            branch_power_vector_1_squared,
            branch_power_vector_2_squared,
            loss_active,
            loss_reactive
        ) = self.linear_electric_grid_model.get_optimization_results(
            self.optimization_problem,
            self.power_flow_solution_reference,
            self.timesteps,
            in_per_unit=in_per_unit,
            with_mean=with_mean
        )
        (
            der_thermal_power_vector,
            node_head_vector,
            branch_flow_vector,
            pump_power
        ) = self.linear_thermal_grid_model.get_optimization_results(
            self.optimization_problem,
            self.timesteps,
            in_per_unit=in_per_unit,
            with_mean=with_mean
        )

        return (
            der_active_power_vector,
            der_reactive_power_vector,
            voltage_magnitude_vector,
            branch_power_vector_1_squared,
            branch_power_vector_2_squared,
            loss_active,
            loss_reactive,
            der_thermal_power_vector,
            node_head_vector,
            branch_flow_vector,
            pump_power
        )

    def get_optimization_dlmps(self):

        # Obtain DLMPs.
        (
            voltage_magnitude_vector_minimum_dlmp,
            voltage_magnitude_vector_maximum_dlmp,
            branch_power_vector_1_squared_maximum_dlmp,
            branch_power_vector_2_squared_maximum_dlmp,
            loss_active_dlmp,
            loss_reactive_dlmp,
            electric_grid_energy_dlmp,
            electric_grid_voltage_dlmp,
            electric_grid_congestion_dlmp,
            electric_grid_loss_dlmp
        ) = self.linear_electric_grid_model.get_optimization_dlmps(
            self.optimization_problem,
            self.price_timeseries,
            self.timesteps
        )
        (
            node_head_vector_minimum_dlmp,
            branch_flow_vector_maximum_dlmp,
            pump_power_dlmp,
            thermal_grid_energy_dlmp,
            thermal_grid_head_dlmp,
            thermal_grid_congestion_dlmp,
            thermal_grid_pump_dlmp
        ) = self.linear_thermal_grid_model.get_optimization_dlmps(
            self.optimization_problem,
            self.price_timeseries,
            self.timesteps
        )

        return (
            voltage_magnitude_vector_minimum_dlmp,
            voltage_magnitude_vector_maximum_dlmp,
            branch_power_vector_1_squared_maximum_dlmp,
            branch_power_vector_2_squared_maximum_dlmp,
            loss_active_dlmp,
            loss_reactive_dlmp,
            electric_grid_energy_dlmp,
            electric_grid_voltage_dlmp,
            electric_grid_congestion_dlmp,
            electric_grid_loss_dlmp,
            node_head_vector_minimum_dlmp,
            branch_flow_vector_maximum_dlmp,
            pump_power_dlmp,
            thermal_grid_energy_dlmp,
            thermal_grid_head_dlmp,
            thermal_grid_congestion_dlmp,
            thermal_grid_pump_dlmp
        )
