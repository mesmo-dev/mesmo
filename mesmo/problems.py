"""Problems module for mathematical optimization and simulation problem type definitions."""

import datetime
import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd
import typing

import mesmo.config
import mesmo.data_interface
import mesmo.der_models
import mesmo.electric_grid_models
import mesmo.thermal_grid_models
import mesmo.utils

logger = mesmo.config.get_logger(__name__)


class Results(
    mesmo.electric_grid_models.ElectricGridOperationResults,
    mesmo.thermal_grid_models.ThermalGridOperationResults,
    mesmo.der_models.DERModelSetOperationResults,
    mesmo.electric_grid_models.ElectricGridDLMPResults,
    mesmo.thermal_grid_models.ThermalGridDLMPResults
):

    price_data: mesmo.data_interface.PriceData


class ResultsDict(typing.Dict[str, Results]):

    pass


class Problem(mesmo.utils.ObjectBase):

    pass


class ProblemDict(typing.Dict[str, Problem]):

    pass


class NominalOperationProblem(Problem):
    """Nominal operation problem object, consisting of the corresponding electric / thermal grid models,
    reference power flow solutions and DER model set for the given scenario.

    - The nominal operation problem (alias: simulation problem, power flow problem)
      represents the simulation problem of the DERs and grids considering the nominal operation schedule for all DERs.
    - The problem formulation is able to consider combined as well as individual operation of
      thermal and electric grids.
    """

    scenario_name: str
    timesteps: pd.Index
    price_data: mesmo.data_interface.PriceData
    electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None
    thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None
    der_model_set: mesmo.der_models.DERModelSet
    results: Results

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None,
            thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None,
            der_model_set: mesmo.der_models.DERModelSet = None
    ):

        # Obtain data.
        scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
        self.price_data = mesmo.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['electric_grid_name']):
            if electric_grid_model is not None:
                self.electric_grid_model = electric_grid_model
            else:
                mesmo.utils.log_time("electric grid model instantiation")
                self.electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
                mesmo.utils.log_time("electric grid model instantiation")

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['thermal_grid_name']):
            if thermal_grid_model is not None:
                self.thermal_grid_model = thermal_grid_model
            else:
                mesmo.utils.log_time("thermal grid model instantiation")
                self.thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
                mesmo.utils.log_time("thermal grid model instantiation")

        # Obtain DER model set.
        if der_model_set is not None:
            self.der_model_set = der_model_set
        else:
            mesmo.utils.log_time("DER model instantiation")
            self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)
            mesmo.utils.log_time("DER model instantiation")

    def solve(self):

        # Instantiate results variables.
        if self.electric_grid_model is not None:
            der_power_vector = (
                pd.DataFrame(columns=self.electric_grid_model.ders, index=self.timesteps, dtype=complex)
            )
            node_voltage_vector = (
                pd.DataFrame(columns=self.electric_grid_model.nodes, index=self.timesteps, dtype=complex)
            )
            branch_power_vector_1 = (
                pd.DataFrame(columns=self.electric_grid_model.branches, index=self.timesteps, dtype=complex)
            )
            branch_power_vector_2 = (
                pd.DataFrame(columns=self.electric_grid_model.branches, index=self.timesteps, dtype=complex)
            )
            loss = pd.DataFrame(columns=['total'], index=self.timesteps, dtype=complex)
        if self.thermal_grid_model is not None:
            der_thermal_power_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.timesteps, dtype=float)
            )
            node_head_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.timesteps, dtype=float)
            )
            branch_flow_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.branches, index=self.timesteps, dtype=float)
            )
            pump_power = pd.DataFrame(columns=['total'], index=self.timesteps, dtype=float)

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
        mesmo.utils.log_time("power flow solution")
        if self.electric_grid_model is not None:
            power_flow_solutions = (
                mesmo.utils.starmap(
                    mesmo.electric_grid_models.PowerFlowSolutionFixedPoint,
                    zip(
                        itertools.repeat(self.electric_grid_model),
                        der_power_vector.values
                    )
                )
            )
            power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))
        if self.thermal_grid_model is not None:
            thermal_power_flow_solutions = (
                mesmo.utils.starmap(
                    mesmo.thermal_grid_models.ThermalPowerFlowSolution,
                    [(self.thermal_grid_model, row) for row in der_thermal_power_vector.values]
                )
            )
            thermal_power_flow_solutions = dict(zip(self.timesteps, thermal_power_flow_solutions))
        mesmo.utils.log_time("power flow solution")

        # Obtain results.
        if self.electric_grid_model is not None:
            for timestep in self.timesteps:
                power_flow_solution = power_flow_solutions[timestep]
                # TODO: Flatten power flow solution arrays.
                node_voltage_vector.loc[timestep, :] = power_flow_solution.node_voltage_vector
                branch_power_vector_1.loc[timestep, :] = power_flow_solution.branch_power_vector_1
                branch_power_vector_2.loc[timestep, :] = power_flow_solution.branch_power_vector_2
                loss.loc[timestep, :] = power_flow_solution.loss
            der_active_power_vector = der_power_vector.apply(np.real)
            der_reactive_power_vector = der_power_vector.apply(np.imag)
            node_voltage_magnitude_vector = np.abs(node_voltage_vector)
            node_voltage_angle_vector = np.angle(node_voltage_vector)
            branch_power_magnitude_vector_1 = np.abs(branch_power_vector_1)
            branch_active_power_vector_1 = np.real(branch_power_vector_1)
            branch_reactive_power_vector_1 = np.imag(branch_power_vector_1)
            branch_power_magnitude_vector_2 = np.abs(branch_power_vector_2)
            branch_active_power_vector_2 = np.real(branch_power_vector_2)
            branch_reactive_power_vector_2 = np.imag(branch_power_vector_2)
            loss_active = loss.apply(np.real)
            loss_reactive = loss.apply(np.imag)
        if self.thermal_grid_model is not None:
            for timestep in self.timesteps:
                thermal_power_flow_solution = thermal_power_flow_solutions[timestep]
                node_head_vector.loc[timestep, :] = thermal_power_flow_solution.node_head_vector
                branch_flow_vector.loc[timestep, :] = thermal_power_flow_solution.branch_flow_vector
                pump_power.loc[timestep, :] = thermal_power_flow_solution.pump_power

        # Obtain per-unit values.
        if self.electric_grid_model is not None:
            der_active_power_vector_per_unit = (
                der_active_power_vector
                * mesmo.utils.get_inverse_with_zeros(np.real(self.electric_grid_model.der_power_vector_reference))
            )
            der_reactive_power_vector_per_unit = (
                der_reactive_power_vector
                * mesmo.utils.get_inverse_with_zeros(np.imag(self.electric_grid_model.der_power_vector_reference))
            )
            node_voltage_magnitude_vector_per_unit = (
                node_voltage_magnitude_vector
                * mesmo.utils.get_inverse_with_zeros(np.abs(self.electric_grid_model.node_voltage_vector_reference))
            )
            branch_power_magnitude_vector_1_per_unit = (
                branch_power_magnitude_vector_1
                * mesmo.utils.get_inverse_with_zeros(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
            branch_active_power_vector_1_per_unit = (
                branch_active_power_vector_1
                * mesmo.utils.get_inverse_with_zeros(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
            branch_reactive_power_vector_1_per_unit = (
                branch_reactive_power_vector_1
                * mesmo.utils.get_inverse_with_zeros(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
            branch_power_magnitude_vector_2_per_unit = (
                branch_power_magnitude_vector_2
                * mesmo.utils.get_inverse_with_zeros(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
            branch_active_power_vector_2_per_unit = (
                branch_active_power_vector_2
                * mesmo.utils.get_inverse_with_zeros(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
            branch_reactive_power_vector_2_per_unit = (
                branch_reactive_power_vector_2
                * mesmo.utils.get_inverse_with_zeros(self.electric_grid_model.branch_power_vector_magnitude_reference)
            )
        if self.thermal_grid_model is not None:
            der_thermal_power_vector_per_unit = (
                der_thermal_power_vector
                * mesmo.utils.get_inverse_with_zeros(self.thermal_grid_model.der_thermal_power_vector_reference)
            )
            node_head_vector_per_unit = (
                node_head_vector
                * mesmo.utils.get_inverse_with_zeros(self.thermal_grid_model.node_head_vector_reference)
            )
            branch_flow_vector_per_unit = (
                branch_flow_vector
                * mesmo.utils.get_inverse_with_zeros(self.thermal_grid_model.branch_flow_vector_reference)
            )

        # Store results.
        self.results = (
            Results(
                price_data=self.price_data,
                der_model_set=self.der_model_set
            )
        )
        if self.electric_grid_model is not None:
            self.results.update(
                Results(
                    electric_grid_model=self.electric_grid_model,
                    der_active_power_vector=der_active_power_vector,
                    der_active_power_vector_per_unit=der_active_power_vector_per_unit,
                    der_reactive_power_vector=der_reactive_power_vector,
                    der_reactive_power_vector_per_unit=der_reactive_power_vector_per_unit,
                    node_voltage_magnitude_vector=node_voltage_magnitude_vector,
                    node_voltage_magnitude_vector_per_unit=node_voltage_magnitude_vector_per_unit,
                    node_voltage_angle_vector=node_voltage_angle_vector,
                    branch_power_magnitude_vector_1=branch_power_magnitude_vector_1,
                    branch_power_magnitude_vector_1_per_unit=branch_power_magnitude_vector_1_per_unit,
                    branch_active_power_vector_1=branch_active_power_vector_1,
                    branch_active_power_vector_1_per_unit=branch_active_power_vector_1_per_unit,
                    branch_reactive_power_vector_1=branch_reactive_power_vector_1,
                    branch_reactive_power_vector_1_per_unit=branch_reactive_power_vector_1_per_unit,
                    branch_power_magnitude_vector_2=branch_power_magnitude_vector_2,
                    branch_power_magnitude_vector_2_per_unit=branch_power_magnitude_vector_2_per_unit,
                    branch_active_power_vector_2=branch_active_power_vector_2,
                    branch_active_power_vector_2_per_unit=branch_active_power_vector_2_per_unit,
                    branch_reactive_power_vector_2=branch_reactive_power_vector_2,
                    branch_reactive_power_vector_2_per_unit=branch_reactive_power_vector_2_per_unit,
                    loss_active=loss_active,
                    loss_reactive=loss_reactive
                )
            )
        if self.thermal_grid_model is not None:
            self.results.update(
                Results(
                    thermal_grid_model=self.thermal_grid_model,
                    der_thermal_power_vector=der_thermal_power_vector,
                    der_thermal_power_vector_per_unit=der_thermal_power_vector_per_unit,
                    node_head_vector=node_head_vector,
                    node_head_vector_per_unit=node_head_vector_per_unit,
                    branch_flow_vector=branch_flow_vector,
                    branch_flow_vector_per_unit=branch_flow_vector_per_unit,
                    pump_power=pump_power
                )
            )

    def get_results(self):

        return self.results


class OptimalOperationProblem(Problem):
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
    scenario_data: mesmo.data_interface.ScenarioData
    timesteps: pd.Index
    price_data: mesmo.data_interface.PriceData
    electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: mesmo.electric_grid_models.PowerFlowSolution = None
    linear_electric_grid_model_set: mesmo.electric_grid_models.LinearElectricGridModelSet = None
    thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None
    thermal_power_flow_solution_reference: mesmo.thermal_grid_models.ThermalPowerFlowSolution = None
    linear_thermal_grid_model_set: mesmo.thermal_grid_models.LinearThermalGridModelSet = None
    der_model_set: mesmo.der_models.DERModelSet
    optimization_problem: mesmo.utils.OptimizationProblem
    results: Results

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None,
            thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None,
            der_model_set: mesmo.der_models.DERModelSet = None
    ):

        # Obtain and store data.
        self.scenario_name = scenario_name
        self.scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
        self.timesteps = self.scenario_data.timesteps
        self.price_data = mesmo.data_interface.PriceData(scenario_name)

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(self.scenario_data.scenario.at['electric_grid_name']):
            mesmo.utils.log_time("electric grid model instantiation")
            if electric_grid_model is not None:
                self.electric_grid_model = electric_grid_model
            else:
                self.electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
            self.power_flow_solution_reference = (
                mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(self.electric_grid_model)
            )
            self.linear_electric_grid_model_set = (
                mesmo.electric_grid_models.LinearElectricGridModelSet(
                    self.electric_grid_model,
                    self.power_flow_solution_reference,
                    linear_electric_grid_model_method=mesmo.electric_grid_models.LinearElectricGridModelGlobal
                )
            )
            mesmo.utils.log_time("electric grid model instantiation")

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(self.scenario_data.scenario.at['thermal_grid_name']):
            mesmo.utils.log_time("thermal grid model instantiation")
            if thermal_grid_model is not None:
                self.thermal_grid_model = thermal_grid_model
            else:
                self.thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
            self.thermal_power_flow_solution_reference = (
                mesmo.thermal_grid_models.ThermalPowerFlowSolution(self.thermal_grid_model)
            )
            self.linear_thermal_grid_model_set = (
                mesmo.thermal_grid_models.LinearThermalGridModelSet(
                    self.thermal_grid_model,
                    self.thermal_power_flow_solution_reference
                )
            )
            mesmo.utils.log_time("thermal grid model instantiation")

        # Obtain DER model set.
        if der_model_set is not None:
            self.der_model_set = der_model_set
        else:
            mesmo.utils.log_time("DER model instantiation")
            self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)
            mesmo.utils.log_time("DER model instantiation")

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        # Define electric grid problem.
        if self.electric_grid_model is not None:
            self.linear_electric_grid_model_set.define_optimization_variables(self.optimization_problem)
            node_voltage_magnitude_vector_minimum = (
                self.scenario_data.scenario['voltage_per_unit_minimum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                if pd.notnull(self.scenario_data.scenario['voltage_per_unit_minimum'])
                else None
            )
            node_voltage_magnitude_vector_maximum = (
                self.scenario_data.scenario['voltage_per_unit_maximum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                if pd.notnull(self.scenario_data.scenario['voltage_per_unit_maximum'])
                else None
            )
            branch_power_magnitude_vector_maximum = (
                self.scenario_data.scenario['branch_flow_per_unit_maximum']
                * self.electric_grid_model.branch_power_vector_magnitude_reference
                if pd.notnull(self.scenario_data.scenario['branch_flow_per_unit_maximum'])
                else None
            )
            self.linear_electric_grid_model_set.define_optimization_parameters(
                self.optimization_problem,
                self.price_data,
                node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
                node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
                branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
            )
            self.linear_electric_grid_model_set.define_optimization_constraints(self.optimization_problem)
            self.linear_electric_grid_model_set.define_optimization_objective(self.optimization_problem)

        # Define thermal grid problem.
        if self.thermal_grid_model is not None:
            self.linear_thermal_grid_model_set.define_optimization_variables(self.optimization_problem)
            node_head_vector_minimum = (
                self.scenario_data.scenario['node_head_per_unit_maximum']
                * self.thermal_power_flow_solution_reference.node_head_vector
                if pd.notnull(self.scenario_data.scenario['voltage_per_unit_maximum'])
                else None
            )
            branch_flow_vector_maximum = (
                self.scenario_data.scenario['pipe_flow_per_unit_maximum']
                * self.thermal_power_flow_solution_reference.branch_flow_vector
                if pd.notnull(self.scenario_data.scenario['pipe_flow_per_unit_maximum'])
                else None
            )
            self.linear_thermal_grid_model_set.define_optimization_parameters(
                self.optimization_problem,
                self.price_data,
                node_head_vector_minimum=node_head_vector_minimum,
                branch_flow_vector_maximum=branch_flow_vector_maximum
            )
            self.linear_thermal_grid_model_set.define_optimization_constraints(self.optimization_problem)
            self.linear_thermal_grid_model_set.define_optimization_objective(self.optimization_problem)

        # Define DER problem.
        self.der_model_set.define_optimization_variables(self.optimization_problem)
        self.der_model_set.define_optimization_parameters(self.optimization_problem, self.price_data)
        self.der_model_set.define_optimization_constraints(self.optimization_problem)
        self.der_model_set.define_optimization_objective(self.optimization_problem)

    def solve(self, raise_runtime_error: bool = True) -> bool:

        # Solve optimization problem.
        # - Catch potential error so that simulation does not stop.
        # TODO: Only do this for trust-region solve.
        try:
            self.optimization_problem.solve()
        except RuntimeError:
            if raise_runtime_error:
                raise
            print('######### COULD NOT SOLVE OPTIMIZATION #########')
            # Return false if solution was not found.
            return False

        # Return true if solution could be found.
        return True

    def solve_trust_region(self):
        """
        Formulates and runs an optimal operation problem. If an electric_grid_model is passed, it runs an optmal power
        flow using the trust region algorithm.
        In general, trust-region based algorithm mitigates the approximation inaccuracy. The algorithm improve the approximate
        solution of the approximated quadratic program in an iterative manner.
        The algorithm is based on the works in:
        [1] Hanif et al. “Decomposition and Equilibrium Achieving Distribution Locational Marginal Prices using Trust-Region Method”
        IEEE Transactions on Smart Grid, pp. 1–1, 2018, doi: 10.1109/TSG.2018.2822766.
        Trust-Region parameters are based on the works in:
        [2] A. M. Giacomoni and B. F. Wollenberg, “Linear programming optimal power flow utilizing a trust region method”
        in North American Power Symposium 2010, Arlington, TX, USA, Sep. 2010, pp. 1–6, doi: 10.1109/NAPS.2010.5619970.
        [3] J. Nocedal and S. J. Wright, "Numerical Optimization", Chapter 4, 2nd ed., New York: Springer, 2006.
        """

        # Instantiate iteration variables.
        sigma = 0.0
        error_control = np.inf
        trust_region_iteration_count = 0
        trust_region_accepted_iteration_count = 0
        infeasible_count = 0
        power_flow_solutions_iter = []
        objective_power_flows_iter = []

        # Define trust-region parameters according to [2].
        # TODO: Move to configuration.
        delta = 1.0  # 3.0 / 0.5 / range: (0, delta_max] / If too big, no power flow solution.
        delta_max = 4.0  # 4.0 / 1.0
        gamma = 0.25  # 0.5 / range: (0, 1)
        eta = 0.1  # 0.1 / range: (0, 0.25]
        tau = 0.1  # 0.1 / range: (0, 0.25]
        epsilon = 1.0e-4  # 1e-3 / 1e-4
        trust_region_iteration_limit = 30
        infeasible_max = 3  # the maximum number of iterations to try to solve the optimization with different deltas

        print(f'Solving problem with trust-region algorithm.')
        # Ignore division by zero or nan warnings (this can happen with e.g. DERs with zero reactive power output)
        # np.seterr(divide='ignore', invalid='ignore')

        # ---------------------------------------------------------------------------------------------------------
        # Obtain the base case power flow, using the active_power_nominal values given as input as initial dispatch
        # quantities. This represents the initial solution candidate.
        print('Obtaining first best guess for optimal power flow...', end='\r')

        # Obtain DER pre-solve results.
        pre_solve_results = self.der_model_set.pre_solve(self.price_data)

        # Obtain power flow solution based on pre-solve from DERs.
        power_flow_solution_set = (
            mesmo.electric_grid_models.PowerFlowSolutionSet(
                self.electric_grid_model,
                pre_solve_results,
            )
        )
        self.linear_electric_grid_model_set = (
            mesmo.electric_grid_models.LinearElectricGridModelSet(
                self.electric_grid_model,
                power_flow_solution_set,
                mesmo.electric_grid_models.LinearElectricGridModelLocal
            )
        )

        # Formulate problem and solve
        feasible = self.solve(raise_runtime_error=False)
        if feasible:
            best_guess_results = self.get_results()
            print('Calculating power flow based on first best guess results')
            pre_solve_results = best_guess_results
        else:
            # If problem infeasible, use DER pre-solve results and start trust region algorithm.
            pass

        print('Obtaining initial power flow solution set candidate...', end='\r')
        # Obtain updated power flow solution set as initial candidate.
        power_flow_solution_set_candidate = (
            mesmo.electric_grid_models.PowerFlowSolutionSet(
                self.electric_grid_model,
                pre_solve_results,
            )
        )

        # ---------------------------------------------------------------------------------------------------------
        # Start trust-region iterations
        start_time = datetime.now()
        first_iter = True

        print('----------------------------------------------------------------------------------------')
        while (
                (trust_region_accepted_iteration_count < trust_region_iteration_limit)
                # and (error_control > epsilon)
        ):
            iter_start_time = datetime.now()
            # Print progress.
            print(f'Starting trust-region iteration #{trust_region_iteration_count}')
            print(f'Accepted iterations: {trust_region_accepted_iteration_count}')

            # Define / update delta parameter for trust region constraints.
            self.optimization_problem.define_parameter(
                name='delta_positive',
                value=delta
            )
            self.optimization_problem.define_parameter(
                name='delta_negative',
                value=(-1)*delta
            )

            # Check trust-region solution acceptance conditions.
            if first_iter or (sigma > tau):
                print('Setting new states.')
                # Check if a satisfactory solution was already found
                if error_control <= epsilon:
                    # If so, break and leave the trust region iterations
                    break

                # Accept der power vector and power flow solution candidate.
                # DER power vector is stored in the der_model_set for every DER and every timestep
                # der_model_set_reference = der_model_set_candidate
                power_flow_solution_set = power_flow_solution_set_candidate
                power_flow_solutions_iter.append(power_flow_solution_set)

                # Get the new reference power vector for DERs based on the accepted candidate. This vector is different
                # from the one of the electric grid model, which is not adapted every iteration
                power_flow_results = power_flow_solution_set.get_results()
                der_active_power_vector_reference = np.nan_to_num(
                    np.real(power_flow_solution_set.der_power_vector) /
                    np.real(self.electric_grid_model.der_power_vector_reference)
                )
                der_reactive_power_vector_reference = np.nan_to_num(
                    np.imag(power_flow_solution_set.der_power_vector) /
                    np.imag(self.electric_grid_model.der_power_vector_reference)
                )

                # Get the new reference values for voltage and branch flow which are used in the Trust-Region constraints
                node_voltage_vector_reference = power_flow_results.node_voltage_magnitude_vector_per_unit
                branch_power_magnitude_vector_1_reference = power_flow_results.branch_power_magnitude_vector_1_per_unit
                branch_power_magnitude_vector_2_reference = power_flow_results.branch_power_magnitude_vector_2_per_unit

                # Get linear electric grid model for all timesteps
                print('Obtaining linear electric grid model set...', end='\r')
                self.linear_electric_grid_model_set = (
                    mesmo.electric_grid_models.LinearElectricGridModelSet(
                        self.electric_grid_model,
                        power_flow_solution_set,
                        mesmo.electric_grid_models.LinearElectricGridModelLocal
                    )
                )

                # Update the parameters so that sensitivity matrices are updated
                # Check if ScenarioData contains information on limits, otherwise set to None (no limits).
                node_voltage_magnitude_vector_minimum = (
                    self.scenario_data.scenario['voltage_per_unit_minimum']
                    * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                    if pd.notnull(self.scenario_data.scenario['voltage_per_unit_minimum'])
                    else None
                )
                node_voltage_magnitude_vector_maximum = (
                    self.scenario_data.scenario['voltage_per_unit_maximum']
                    * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                    if pd.notnull(self.scenario_data.scenario['voltage_per_unit_maximum'])
                    else None
                )
                branch_power_magnitude_vector_maximum = (
                    self.scenario_data.scenario['branch_flow_per_unit_maximum']
                    * self.electric_grid_model.branch_power_vector_magnitude_reference
                    if pd.notnull(self.scenario_data.scenario['branch_flow_per_unit_maximum'])
                    else None
                )

                # Define electric grid constraints for every timestep
                self.linear_electric_grid_model_set.define_optimization_parameters(
                    optimization_problem=self.optimization_problem,
                    price_data=self.price_data,
                    node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
                    node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
                    branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
                )

                # Evaluate the objective function based on power flow results and store objective value.
                objective_power_flows_iter.append(
                    self.linear_electric_grid_model_set.evaluate_optimization_objective(
                        power_flow_results,
                        self.price_data
                    )
                )

                # Define / update parameters for trust region iteration
                self.optimization_problem.define_parameter(
                    name='node_voltage_vector_reference',
                    value=(-1) * node_voltage_vector_reference.to_numpy().ravel()
                )
                self.optimization_problem.define_parameter(
                    name='branch_power_magnitude_vector_1_reference',
                    value=(-1) * branch_power_magnitude_vector_1_reference.to_numpy().ravel()
                )
                self.optimization_problem.define_parameter(
                    name='branch_power_magnitude_vector_2_reference',
                    value=(-1) * branch_power_magnitude_vector_2_reference.to_numpy().ravel()
                )
                self.optimization_problem.define_parameter(
                    name='der_active_power_vector_reference',
                    value=(-1) * der_active_power_vector_reference.ravel()
                )
                self.optimization_problem.define_parameter(
                    name='der_reactive_power_vector_reference',
                    value=(-1) * der_reactive_power_vector_reference.ravel()
                )
                if first_iter:  # This is only done on this first iteration

                    # Define trust region constraints.
                    # The trust-region permissible value for variables to move is determined by radius delta, which is included
                    # in all inequality constraints [1].
                    # -> Branch flow and voltage limits
                    # -> DER power output limits
                    # We redefine the approximate state and dispatch quantities as the measure of change in their
                    # operating state at the current iteration.

                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='node_voltage_magnitude_vector', timestep=self.timesteps)),
                        ('constant', 'node_voltage_vector_reference', dict(timestep=self.timesteps)),
                        '<=',
                        ('constant', 'delta_positive', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='node_voltage_magnitude_vector', timestep=self.timesteps)),
                        ('constant', 'node_voltage_vector_reference', dict(timestep=self.timesteps)),
                        '>=',
                        ('constant', 'delta_negative', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='branch_power_magnitude_vector_1', timestep=self.timesteps)),
                        ('constant', 'branch_power_magnitude_vector_1_reference', dict(timestep=self.timesteps)),
                        '<=',
                        ('constant', 'delta_positive', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='branch_power_magnitude_vector_1', timestep=self.timesteps)),
                        ('constant', 'branch_power_magnitude_vector_1_reference', dict(timestep=self.timesteps)),
                        '>=',
                        ('constant', 'delta_negative', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='branch_power_magnitude_vector_2', timestep=self.timesteps)),
                        ('constant', 'branch_power_magnitude_vector_2_reference', dict(timestep=self.timesteps)),
                        '<=',
                        ('constant', 'delta_positive', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='branch_power_magnitude_vector_2', timestep=self.timesteps)),
                        ('constant', 'branch_power_magnitude_vector_2_reference', dict(timestep=self.timesteps)),
                        '>=',
                        ('constant', 'delta_negative', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='der_active_power_vector', timestep=self.timesteps)),
                        ('constant', 'der_active_power_vector_reference', dict(timestep=self.timesteps)),
                        '<=',
                        ('constant', 'delta_positive', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='der_active_power_vector', timestep=self.timesteps)),
                        ('constant', 'der_active_power_vector_reference', dict(timestep=self.timesteps)),
                        '>=',
                        ('constant', 'delta_negative', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='der_reactive_power_vector', timestep=self.timesteps)),
                        ('constant', 'der_reactive_power_vector_reference', dict(timestep=self.timesteps)),
                        '<=',
                        ('constant', 'delta_positive', dict(timestep=self.timesteps)),
                    )
                    self.optimization_problem.define_constraint(
                        ('variable', 1.0, dict(name='der_reactive_power_vector', timestep=self.timesteps)),
                        ('constant', 'der_reactive_power_vector_reference', dict(timestep=self.timesteps)),
                        '>=',
                        ('constant', 'delta_negative', dict(timestep=self.timesteps)),
                    )

                # After first iteration, set to False
                first_iter = False

            # Solve the optimization problem
            feasible = self.solve(raise_runtime_error=False)
            if not feasible:
                infeasible_count += 1
                if delta >= delta_max or infeasible_count > infeasible_max:
                    print(f'Optimization problem for scenario {self.scenario_name} infeasible')
                    return [None, None, self.linear_electric_grid_model_set]
                else:
                    print(f'Optimization problem infeasible, increasing delta to maximum')
                    print(f'Trying to solve again #{infeasible_count}')
                    # delta = min(2 * delta, delta_max)
                    print(f"new delta = {delta}")
                    delta = delta_max
                    continue

            # Obtain results.
            optimization_results = self.get_results()

            # ---------------------------------------------------------------------------------------------------------
            # Trust-region evaluation and update.
            print('Trust-region evaluation and update...', end='\r')
            # Obtain der power change value.
            der_active_power_vector_change_per_unit = (
                    optimization_results.der_active_power_vector_per_unit - der_active_power_vector_reference
            ).to_numpy()
            der_reactive_power_vector_change_per_unit = (
                    optimization_results.der_reactive_power_vector_per_unit - der_reactive_power_vector_reference
            ).to_numpy()

            der_power_vector_change_per_unit_max = (
                max(
                    np.max(abs(der_active_power_vector_change_per_unit)),
                    np.max(abs(der_reactive_power_vector_change_per_unit))
                )
            )
            error_control = der_power_vector_change_per_unit_max

            # ---------------------------------------------------------------------------------------------------------
            # Check trust-region conditions and obtain DER power vector / power flow solution candidates for next iteration.
            # - Only if termination condition is not met, otherwise risk of division by zero.
            # if der_power_vector_change_per_unit_max > epsilon:x
            # Get new power flow solution candidate
            print('Obtaining power flow solution set candidate...', end='\r')
            power_flow_solution_set_candidate = (
                mesmo.electric_grid_models.PowerFlowSolutionSet(
                    self.electric_grid_model,
                    optimization_results
                )
            )

            # Obtain objective values.
            objective_linear_model = (
                self.linear_electric_grid_model_set.evaluate_optimization_objective(
                    optimization_results,
                    self.price_data
                )
            )

            # Get power flow solutions for candidate
            power_flow_results_candidate = power_flow_solution_set_candidate.get_results()

            # Evaluate the optimization objective
            objective_power_flow = (
                self.linear_electric_grid_model_set.evaluate_optimization_objective(
                    power_flow_results_candidate,
                    self.price_data
                )
            )

            # Calculate objective function error
            error_obj_function = np.abs(objective_power_flow - objective_linear_model)

            # Check if power flow of candidate is violating line limits, to then increase the radius delta
            # This is only to save some iterations
            # pf_violation_flag_1 = (
            #         np.abs(branch_power_magnitude_vector_maximum
            #                - power_flow_results_candidate.branch_power_magnitude_vector_1) < 0).any(axis=None)
            # pf_violation_flag_2 = (
            #         np.abs(branch_power_magnitude_vector_maximum
            #                - power_flow_results_candidate.branch_power_magnitude_vector_2) < 0).any(axis=None)

            # ---------------------------------------------------------------------------------------------------------
            # Evaluate solution progress.
            # sigma represents the ratio between the cost improvement of approximated system to the actual one. A
            # smaller value of sigma shows that the current approximation does not represent the actual system and hence
            # the the optimization region must be reduced. For a considerably higher value of sigma, the linear
            # approximation is accurate and the system can move to a new operating point. [1]
            try:
                sigma = float(
                    (objective_power_flows_iter[-1] - objective_power_flow)
                    / (objective_power_flows_iter[-1] - objective_linear_model)
                )
                # TODO: there are cases when sigma repeats itself every second iteration causing and endless loop until
                #  the max number of iterations is reached. This should probably be checked and if true, what then?
            except ZeroDivisionError:
                print('ZeroDivisionError in calculating sigma value.')
                sigma_numerator = (objective_power_flows_iter[-1] - objective_power_flow)
                if sigma_numerator == 0:  # TODO: does this case really exist? should it evaluate to zero or 1?
                    sigma = 0  # this means, no progress has been done, so something should happen --> decrease delta
                elif sigma_numerator < 0:
                    sigma = (-1) * np.inf
                else:
                    sigma = np.inf
                print(f'Evaluated numerator, falling back to sigma = {sigma}')

            # Print trust-region progress
            print(f'objective_power_flow = {objective_power_flow}')
            print(f'objective_linear_model = {objective_linear_model}')
            print(f'control error = {error_control}')
            print(f'objective error = {error_obj_function}')
            print(f'sigma = {sigma}')

            # if pf_violation_flag_1 or pf_violation_flag_2:  # first check if there are any line flow violations
            #     print('Found line flow violation, decreasing delta.')
            #     delta *= gamma
            # If the new objective value is greater than the current value ([-1]), the step must be rejected, see [4]
            # elif (objective_power_flows_iter[-1] - objective_linear_model) <= 0:
            #     print('New objective larger than objective[-1]')
            #     delta *= gamma
            if sigma < eta:
                print('sigma < eta, linearized model is a bad approximation of the nonlinear model, decreasing delta')
                delta *= gamma
                print(f'new delta = {delta}')
            elif sigma > (1.0 - eta) and np.abs(der_power_vector_change_per_unit_max - delta) <= epsilon:
                # elif sigma > (1.0 - eta) and np.abs(voltage_magnitude_change - delta) <= epsilon:
                print('sigma > (1.0 - eta), linearized model is a good approximation of the nonlinear model, increasing delta')
                delta = min(2 * delta, delta_max)
                print(f'new delta = {delta}')
            else:
                # If the step stays strictly inside the region, we infer that the current value of delta is not
                # interfering with the progress of the algorithm, so we leave its value unchanged for the next iteration
                # see [3]
                print('linearized model is a satisfactory approximation of the nonlinear model, delta remains unchanged.')
                print(f'delta = {delta}')

            if sigma > tau:
                print('sigma > tau -> the solution to the current iteration makes satisfactory progress toward the '
                      'optimal solution')
                print('Accepting iteration.')
                # Increase counter
                trust_region_accepted_iteration_count += 1

            else:
                print('sigma <= tau -> Rejecting iteration. Repeating iteration using the modified region (delta).')

            trust_region_iteration_count += 1
            print(f'Time elapsed for iteration: {datetime.now()-iter_start_time}')
            print('----------------------------------------------------------------------------------------')

        # ---------------------------------------------------------------------------------------------------------
        end_time = datetime.now()
        print('Found solution, exiting the trust region iterations.')
        print('----------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------')
        print(f'Trust region iterations: {trust_region_iteration_count}')
        print(f'Total time elapsed for trust region algorithm: {(end_time - start_time)}')

    def get_results(self) -> Results:

        # Instantiate results.
        self.results = Results(price_data=self.price_data)

        # Obtain electric grid results.
        if self.electric_grid_model is not None:
            self.results.update(self.linear_electric_grid_model_set.get_optimization_results(self.optimization_problem))

        # Obtain thermal grid results.
        if self.thermal_grid_model is not None:
            self.results.update(self.linear_thermal_grid_model_set.get_optimization_results(self.optimization_problem))

        # Obtain DER results.
        self.results.update(self.der_model_set.get_optimization_results(self.optimization_problem))

        # Obtain electric DLMPs.
        if self.electric_grid_model is not None:
            self.results.update(
                self.linear_electric_grid_model_set.get_optimization_dlmps(
                    self.optimization_problem,
                    self.price_data
                )
            )

        # Obtain thermal DLMPs.
        if self.thermal_grid_model is not None:
            self.results.update(
                self.linear_thermal_grid_model_set.get_optimization_dlmps(
                    self.optimization_problem,
                    self.price_data
                )
            )

        return self.results
