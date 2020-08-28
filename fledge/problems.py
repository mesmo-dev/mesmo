"""Problems module for mathematical optimization and simulation problem type definitions."""

import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd
import pyomo.environ as pyo

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

        # Store results.
        self.results = fledge.data_interface.ResultsDict()
        if self.electric_grid_model is not None:
            self.results.update(
                fledge.data_interface.ResultsDict(
                    der_power_vector=der_power_vector,
                    node_voltage_vector=node_voltage_vector,
                    branch_power_vector_1=branch_power_vector_1,
                    branch_power_vector_2=branch_power_vector_2,
                    loss=loss
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
    price_timeseries: pd.DataFrame
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: fledge.electric_grid_models.PowerFlowSolution = None
    linear_electric_grid_model: fledge.electric_grid_models.LinearElectricGridModel = None
    thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None
    thermal_power_flow_solution_reference: fledge.thermal_grid_models.ThermalPowerFlowSolution = None
    linear_thermal_grid_model: fledge.thermal_grid_models.LinearThermalGridModel = None
    der_model_set: fledge.der_models.DERModelSet
    optimization_problem: pyo.ConcreteModel

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain data.
        scenario_data = fledge.data_interface.ScenarioData(scenario_name)
        price_data = fledge.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Store price timeseries.
        if pd.notnull(scenario_data.scenario.at['price_type']):
            self.price_timeseries = price_data.price_timeseries_dict[scenario_data.scenario.at['price_type']]
        else:
            self.price_timeseries = (
                pd.DataFrame(
                    1.0,
                    index=self.timesteps,
                    columns=['price_value']
                )
            )

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
        self.optimization_problem = pyo.ConcreteModel()

        # Define linear electric grid model variables and constraints.
        if self.electric_grid_model is not None:
            self.linear_electric_grid_model.define_optimization_variables(
                self.optimization_problem,
                self.timesteps
            )
            voltage_magnitude_vector_minimum = (
                scenario_data.scenario['voltage_per_unit_minimum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                if pd.notnull(scenario_data.scenario['voltage_per_unit_minimum'])
                else None
            )
            voltage_magnitude_vector_maximum = (
                scenario_data.scenario['voltage_per_unit_maximum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
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
            power_flow_solution=self.power_flow_solution_reference,
            thermal_grid_model=self.thermal_grid_model,
            thermal_power_flow_solution=self.thermal_power_flow_solution_reference
        )

        # Define objective.
        if self.thermal_grid_model is not None:
            self.linear_thermal_grid_model.define_optimization_objective(
                self.optimization_problem,
                self.price_timeseries,
                self.timesteps
            )
        self.der_model_set.define_optimization_objective(
            self.optimization_problem,
            self.price_timeseries
        )

    def solve(self):

        # Solve optimization problem.
        self.optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
        optimization_result = (
            optimization_solver.solve(
                self.optimization_problem,
                tee=fledge.config.config['optimization']['show_solver_output']
            )
        )

        # Assert that solver exited with any solution. If not, raise an error.
        try:
            assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
        except AssertionError:
            logger.error(f"Solver termination condition: {optimization_result.solver.termination_condition}")
            raise

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
                        self.price_timeseries,
                        self.timesteps
                    )
                )

            # Obtain thermal DLMPs.
            if self.thermal_grid_model is not None:
                results.update(
                    self.linear_thermal_grid_model.get_optimization_dlmps(
                        self.optimization_problem,
                        self.price_timeseries,
                        self.timesteps
                    )
                )

        return results
