"""
This module controls the simulation steps
"""

import numpy as np
import pandas as pd
import itertools
from multimethod import multimethod

import fledge.config
import fledge.data_interface
import fledge.utils
import fledge.der_models
import fledge.electric_grid_models
import fledge.problems

logger = fledge.config.get_logger(__name__)


class SolutionEngine(object):

    @multimethod
    def run_presolve_with_der_models(
            self,
            scenario_name: str,
    ) -> fledge.problems.Results:
        return self.__run_presolve(scenario_name=scenario_name, der_model_set=None, price_data=None)

    @multimethod
    def run_presolve_with_der_models(
            self,
            der_model_set: fledge.der_models.DERModelSet,
            price_data: fledge.data_interface.PriceData
    ) -> fledge.problems.Results:
        return self.__run_presolve(scenario_name=None, der_model_set=der_model_set, price_data=price_data)

    def __run_presolve(
            self,
            scenario_name: str = None,
            der_model_set: fledge.der_models.DERModelSet = None,
            price_data: fledge.data_interface.PriceData = None
    ) -> fledge.problems.Results:
        # Pre-solve optimal operation problem without underlying electric grid to get realistic initial values
        # Obtain all DERs
        print('Running pre-solve for der models only...', end='\r')

        # Turn off solver output for pre-solve
        # Store original solver output value
        show_solver_output_original = fledge.config.config['optimization']['show_solver_output']
        fledge.config.config['optimization']['show_solver_output'] = False

        if scenario_name is not None:
            _, results = self.run_optimal_operation_problem(
                scenario_name=scenario_name,
                problem_type='decentral'
            )
        elif (der_model_set is not None) and (price_data is not None):
            problem = DEROptimalOperationProblem(
                der_model_set=der_model_set,
                price_data=price_data
            )
            problem.formulate_optimization_problem()
            problem.solve()
            # Obtain results.
            results = problem.get_results()

        # Set back to original value
        fledge.config.config['optimization']['show_solver_output'] = show_solver_output_original
        return results

    def run_optimal_operation_problem(
        self,
        scenario_name: str,
        problem_type: str,
        results_path: str = None,
    ) -> [float, fledge.problems.Results]:
        if problem_type is 'decentral':
            problem = DEROptimalOperationProblem(scenario_name)
        elif problem_type is 'central':
            # Instantiate problem object
            problem = ElectricGridOptimalOperationProblem(scenario_name)
            # run pre-solve and change initial DER setpoints for more accurate linearization
            der_model_set = fledge.der_models.DERModelSet(scenario_name)
            presolve_results = self.__run_presolve(scenario_name=scenario_name)
            problem.der_model_set = ScenarioHandler.change_der_set_points_based_on_results(
                der_model_set, presolve_results
            )
        else:
            raise ValueError(f'Unknown problem type: {problem_type}.')

        print(f'Formulating problem of type {problem_type}')

        # TODO: there should be a way to add custom constraints! --> something like notify in MATLAB?
        problem.formulate_optimization_problem()
        feasible = problem.solve()
        if not feasible:
            logger.warning(f"Optimization problem for scenario {scenario_name} infeasible")
            return [None, None]

        # Obtain results.
        results = problem.get_results()
        # Print results.
        # print(results)

        # Store results to CSV.
        if results_path is not None:
            results.save(results_path)

        # Print results path.
        # fledge.utils.launch(results_path)
        print(f"Results are stored in: {results_path}")

        return [problem.optimization_problem.objective.value, results]

    @staticmethod
    def run_nominal_operation(
            scenario_name: str,
            der_model_set: fledge.der_models.DERModelSet,
            results_path: str = None
    ) -> fledge.problems.Results:
        # run nominal operation problem with the set points from the decentralized problems
        # Formulate nominal operation problem
        problem_type = 'nominal'
        print(f'Formulating problem of type: {problem_type}.')
        problem = fledge.problems.NominalOperationProblem(scenario_name)
        # Update the der model set (with new set points)
        problem.der_model_set = der_model_set
        problem.solve()
        results = problem.get_results()
        # Print results.
        print(results)
        # Store results as CSV.
        if results_path is not None:
            results.save(results_path)

        print(f"Results are stored in: {results_path}")

        return results

    @staticmethod
    def get_der_power_vector(
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            der_models_set: fledge.der_models.DERModelSet,
            timesteps: pd.Index
    ) -> pd.DataFrame:
        der_power_vector = (
            pd.DataFrame(columns=electric_grid_model.ders, index=timesteps, dtype=np.complex)
        )
        for der in electric_grid_model.ders:
            der_name = der[1]
            der_power_vector.loc[:, der] = (
                    der_models_set.der_models[der_name].active_power_nominal_timeseries
                    + (1.0j * der_models_set.der_models[der_name].reactive_power_nominal_timeseries)
            )
        return der_power_vector

    def get_power_flow_solutions_per_timestep(
            self,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            der_model_set_new_setpoints: fledge.der_models.DERModelSet,
            timesteps: pd.Index
    ):
        der_power_vector = self.get_der_power_vector(electric_grid_model, der_model_set_new_setpoints, timesteps)
        # use DER power vector to calculate power flow per timestep
        power_flow_solutions = (
            fledge.utils.starmap(
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
                zip(
                    itertools.repeat(electric_grid_model),
                    der_power_vector.values
                )
            )
        )
        return dict(zip(timesteps, power_flow_solutions))

    @staticmethod
    def get_linear_electric_grid_models_per_timestep(
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            power_flow_solutions: dict,
            timesteps
    ) -> dict:
        print('Obtaining linear electric grid model for all timesteps...', end='\r')
        # TODO: adapt to Local Approx: LinearElectricGridModelLocal
        linear_electric_grid_models = (
            fledge.utils.starmap(
                fledge.electric_grid_models.LinearElectricGridModelGlobal,
                zip(
                    itertools.repeat(electric_grid_model),
                    list(power_flow_solutions.values())
                )
            )
        )
        linear_electric_grid_models = dict(zip(timesteps, linear_electric_grid_models))
        # Assign corresponding timestep to the linear electric grid model attribute
        for timestep in timesteps:
            linear_electric_grid_models[timestep].timestep = timestep

        return linear_electric_grid_models

    @staticmethod
    def evaluate_optimization_objective_based_on_power_flow(
            optimization_problem: fledge.utils.OptimizationProblem,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            power_flow_solutions: dict,
            timesteps
    ) -> float:
        # set objective function variables to zero (power vector change is zero, as the solution is already correct)
        loss_active = np.zeros([len(timesteps), 1], dtype=float)
        loss_reactive = np.zeros([len(timesteps), 1], dtype=float)
        active_power = np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=float)
        reactive_power = np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=float)
        time_index = 0
        for timestep in timesteps:
            power_flow_solution = power_flow_solutions[timestep]
            loss_active[time_index][0] = np.real(power_flow_solution.loss)
            loss_reactive[time_index][0] = np.imag(power_flow_solution.loss)
            for der_index, der in enumerate(electric_grid_model.ders):
                active_power[time_index][der_index] = np.real(power_flow_solution.der_power_vector[der_index])
                reactive_power[time_index][der_index] = np.imag(power_flow_solution.der_power_vector[der_index])
            time_index += 1

        optimization_problem.loss_active.value = loss_active
        optimization_problem.loss_reactive.value = loss_reactive
        optimization_problem.der_active_power_vector.value = active_power
        optimization_problem.der_reactive_power_vector.value = reactive_power

        return float(optimization_problem.objective.value)


class ScenarioHandler(object):
    default_power_factor = 0.8

    def change_der_set_points_based_on_results(
            self,
            der_model_set: fledge.der_models.DERModelSet,
            results: fledge.problems.Results
    ) -> fledge.der_models.DERModelSet:
        attributes = dir(results)
        if 'der_active_power_vector' in attributes:
            for der_name in der_model_set.der_names:
                der_model = der_model_set.der_models[der_name]
                der_type = der_model.der_type
                der_model.active_power_nominal_timeseries = (
                    results.der_active_power_vector[der_type, der_name]
                )
                der_model.reactive_power_nominal_timeseries = (
                    results.der_reactive_power_vector[der_type, der_name]
                )
        # If there was no electric grid model in the optimization, get the results based on the output vector
        elif 'output_vector' in attributes:
            for der_name in der_model_set.der_names:
                der_model = der_model_set.der_models[der_name]
                if issubclass(type(der_model), fledge.der_models.FlexibleDERModel):
                    if 'active_power' in results.output_vector[der_name].columns:
                        der_model.active_power_nominal_timeseries = (
                            results.output_vector[(der_name, 'active_power')]
                        )
                        der_model.reactive_power_nominal_timeseries = (
                            results.output_vector[(der_name, 'reactive_power')]
                        )
                    elif 'grid_electric_power' in results.output_vector[der_name].columns:
                        logger.warning(
                            f'FlexibleBuildingModel detected, using default power factor {self.default_power_factor}'
                        )
                        der_model.active_power_nominal_timeseries = (
                            results.output_vector[(der_name, 'grid_electric_power')]
                        ) * (-1)
                        der_model.reactive_power_nominal_timeseries = (
                            results.output_vector[(der_name, 'grid_electric_power')]
                            * np.tan(np.arccos(self.default_power_factor))
                        ) * (-1)
        else:
            print('Results object does not contain any data on active power output. ')
            raise ValueError

        return der_model_set


class OptimalOperationProblem(object):
    """"Custom optimal operation problem object based on fledge.problems.OptimalOperationProblem, consisting of an
    optimization problem as well as the corresponding electric reference power flow solutions, linear grid models and
    DER model set for the given scenario.

    The main difference lies in the separation of the problem formulation from the init function and the possibility
    to linearize the electric grid model based on a power flow solution for each timestep
    """
    # TODO: this should be changed to accept power_flow_solution and linear_electric_grid_model per timestep
    scenario_name: str
    timesteps: pd.Index
    price_data: fledge.data_interface.PriceData
    scenario_data: fledge.data_interface.ScenarioData
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solutions_per_timestep: dict = None
    linear_electric_grid_models_per_timestep: dict = None
    linear_electric_grid_model: fledge.electric_grid_models.LinearElectricGridModelGlobal
    der_model_set: fledge.der_models.DERModelSet
    optimization_problem: fledge.utils.OptimizationProblem
    results: fledge.problems.Results

    def __init__(
            self,
            scenario_name: str,
            central: bool = True
    ):

        # Obtain data.
        self.scenario_data = fledge.data_interface.ScenarioData(scenario_name)
        self.price_data = fledge.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = self.scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if central and pd.notnull(self.scenario_data.scenario.at['electric_grid_name']):
            self.electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

        # Obtain DER model set.
        self.der_model_set = fledge.der_models.DERModelSet(scenario_name)

    def formulate_optimization_problem(self):
        # ---------------------------------------------------------------------------------------------------------
        # OPTIMAL POWER FLOW
        print('Formulating optimization problem...', end='\r')
        # Instantiate optimization problem.
        self.optimization_problem = fledge.utils.OptimizationProblem()

        if self.electric_grid_model is not None:
            # ---------------------------------------------------------------------------------------------------------
            # POWER FLOW AND LINEAR ELECTRIC GRID MODEL
            # Obtain the base power flow, using the values from the presolved optmization given as input as initial
            # dispatch quantities.
            solution_engine = SolutionEngine()
            self.power_flow_solutions_per_timestep = solution_engine.get_power_flow_solutions_per_timestep(
                electric_grid_model=self.electric_grid_model,
                der_model_set_new_setpoints=self.der_model_set,
                timesteps=self.timesteps
            )
            # Get linear electric grid model for all timesteps
            linear_electric_grid_models_per_timestep = solution_engine.get_linear_electric_grid_models_per_timestep(
                electric_grid_model=self.electric_grid_model,
                power_flow_solutions=self.power_flow_solutions_per_timestep,
                timesteps=self.timesteps)
            # Get the first linear electric grid model for the next function calls
            self.linear_electric_grid_model = linear_electric_grid_models_per_timestep[self.timesteps[0]]

            # Define linear electric grid model variables and constraints.
            self.linear_electric_grid_model.define_optimization_variables(
                self.optimization_problem,
                self.timesteps
            )
            # TODO: should I just pass custom key paramter argument?
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
            self.linear_electric_grid_model.define_optimization_constraints(
                self.optimization_problem,
                self.timesteps,
                node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
                node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
                branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
            )

        # Define DER variables and constraints.
        self.der_model_set.define_optimization_variables(
            self.optimization_problem
        )

        self.der_model_set.define_optimization_constraints(
            self.optimization_problem,
            electric_grid_model=self.electric_grid_model
        )

        # Additional constraint for flexible buildings
        for der_name in self.der_model_set.der_models.keys():
            der_model = self.der_model_set.der_models[der_name]
            if type(der_model) is fledge.der_models.FlexibleBuildingModel:
                # Limit loads to their nominal power consumption
                der_model.output_maximum_timeseries['grid_electric_power'] = \
                    (-1) * der_model.active_power_nominal_timeseries
                # Put a constraint on cooling power (= 0) to effectively disable cooling in the HVAC system
                der_model.output_maximum_timeseries['zone_generic_cool_thermal_power_cooling'] = 0

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
        )

    def solve(
            self
    ) -> bool:
        print('Solving optimal power flow...', end='\r')
        # Catch potential error so that simulation does not stop
        # Return if solution was found or not (true / false)
        try:
            self.optimization_problem.solve()
        except AssertionError:
            print('######### COULD NOT SOLVE OPTIMIZATION #########')
            print(f"Solver termination status: {self.optimization_problem.cvxpy_problem.status}")
            # return that opimization was infeasible
            # TODO: check status for infeasible status
            return False

        # return that solution could be found
        return True

    def get_results(self) -> fledge.problems.Results:
        # Instantiate results.
        self.results = (
            fledge.problems.Results(
                price_data=self.price_data
            )
        )

        # Obtain electric grid results.
        if self.electric_grid_model is not None:
            self.results.update(
                self.linear_electric_grid_model.get_optimization_results(
                    self.optimization_problem,
                    None,
                    self.timesteps
                )
            )

        # Obtain DER results.
        self.results.update(
            self.der_model_set.get_optimization_results(
                self.optimization_problem
            )
        )

        # Obtain electric DLMPs.
        if self.electric_grid_model is not None:
            self.results.update(
                self.linear_electric_grid_model.get_optimization_dlmps(
                    self.optimization_problem,
                    self.price_data,
                    self.timesteps
                )
            )

        return self.results


class DEROptimalOperationProblem(OptimalOperationProblem):
    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):
        super().__init__(
            scenario_name=scenario_name,
            central=False
        )

    @multimethod
    def __init__(
            self,
            der_model_set: fledge.der_models.DERModelSet,
            price_data: fledge.data_interface.PriceData
    ):
        self.der_model_set = der_model_set
        self.price_data = price_data


class ElectricGridOptimalOperationProblem(OptimalOperationProblem):
    def __init__(
            self,
            scenario_name: str
    ):
        super().__init__(
            scenario_name=scenario_name,
            central=True
        )


class Controller(object):
    results_dict: dict
    simulation_scenarios: dict
    der_penetration_scenario_data: dict
    solution_engine: SolutionEngine

    def __init__(
            self,
            simulation_scenarios: dict,
    ):
        self.results_dict = {}
        self.simulation_scenarios = simulation_scenarios
        pass

    def run(
            self
    ) -> dict:

        optimization_results_dict = {}
        # Run decentralized problem based on wholesale market price (no granularity)
        # TODO: make this more general like in the prototype
        granularity_level = 'no_granularity'
        problem_type = 'decentral'
        results_path = fledge.utils.get_results_path(
            'optimal_operation', granularity_level + '_' + problem_type)
        scenario_name = self.simulation_scenarios[granularity_level]
        self.solution_engine = SolutionEngine()
        opt_objective, opt_results = self.solution_engine.run_optimal_operation_problem(
                scenario_name=scenario_name,
                results_path=results_path,
                problem_type=problem_type
            )
        optimization_results_dict['opt_objective_' + granularity_level + '_' + problem_type] = opt_objective
        optimization_results_dict['results_' + granularity_level + '_' + problem_type] = opt_results

        power_flow_results_dict = {}
        # Run nominal operation of electric grid and check for violations
        # Get the set points from decentralized problems and calculate nominal power flow
        # now using the entire grid again
        granularity_level = 'high_granularity'
        scenario_name = self.simulation_scenarios[granularity_level]
        for key in optimization_results_dict:
            if not(('decentral' in key) and ('results' in key)):
                # only use solution from decentral optimization, central optimization will yield the same
                continue
            results_path = fledge.utils.get_results_path('electric_grid_nominal_operation', 'wholesale')
            # change the set points of the DERs
            try:
                der_model_set = fledge.der_models.DERModelSet(scenario_name)
                der_model_set_new_setpoints = ScenarioHandler.change_der_set_points_based_on_results(
                    der_model_set,
                    optimization_results_dict[key])
            except KeyError:
                raise
            # run the nominal power flow and store results in dictionary
            power_flow_results_dict[scenario_name] = self.solution_engine.run_nominal_operation(
                scenario_name,
                der_model_set_new_setpoints,
                results_path)

        self.results_dict = {
            'optimization_results': optimization_results_dict,
            'power_flow_results': power_flow_results_dict
        }
        return self.results_dict

    def get_results(
            self
    ) -> dict:
        return self.results_dict
