"""
This module controls the simulation steps
"""

import numpy as np
import pandas as pd
import itertools

import fledge.config
import fledge.data_interface
import fledge.utils
import fledge.der_models
import fledge.electric_grid_models
import fledge.problems

logger = fledge.config.get_logger(__name__)


class Controller(object):
    results_dict: dict
    simulation_scenarios: dict
    der_penetration_scenario_data: dict

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
        opt_objective, opt_results = SolutionEngine.run_optimal_operation_problem(
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
                der_model_set_new_setpoints = ScenarioHandler.change_der_set_points_based_on_results(
                    scenario_name,
                    optimization_results_dict[key])
            except KeyError:
                raise
            # run the nominal power flow and store results in dictionary
            power_flow_results_dict[scenario_name] = SolutionEngine.run_nominal_operation(
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


class SolutionEngine(object):

    @staticmethod
    def run_optimal_operation_problem(
        scenario_name: str,
        results_path: str,
        problem_type: str
    ) -> [float, fledge.problems.Results]:
        if problem_type is 'central':
            problem = DEROptimalOperationProblem(scenario_name)
        elif problem_type is 'decentral':
            problem = ElectricGridOptimalOperationProblem(scenario_name)
        else:
            raise ValueError(f'Unknown problem type: {problem_type}.')

        logger.info(f'Formulating problem of type {problem_type}')

        # TODO: there should be a way to add custom constraints! --> something like notify in MATLAB?
        problem.formulate_optimization_problem()
        feasible = problem.solve()
        if not feasible:
            logger.warning(f"Optimization problem for scenario {scenario_name} infeasible")
            return [None, None]

        # Obtain results.
        results = problem.get_results()
        # Print results.
        print(results)

        # Store results to CSV.
        results.save(results_path)

        # Print results path.
        # fledge.utils.launch(results_path)
        logger.info(f"Results are stored in: {results_path}")

        return [problem.optimization_problem.objective.value[0], results]

    @staticmethod
    def run_nominal_operation(
            scenario_name: str,
            der_model_set: fledge.der_models.DERModelSet,
            results_path: str
    ) -> fledge.problems.Results:
        # run nominal operation problem with the set points from the decentralized problems
        # Formulate nominal operation problem
        problem_type = 'nominal'
        logger.info(f'Formulating problem of type: {problem_type}.')
        problem = fledge.problems.NominalOperationProblem(scenario_name)
        # Update the der model set (with new set points)
        problem.der_model_set = der_model_set
        problem.solve()
        results = problem.get_results()
        # Print results.
        print(results)
        # Store results as CSV.
        results.save(results_path)
        logger.info(f"Results are stored in: {results_path}")

        return results

    @staticmethod
    def get_power_flow_solutions_for_timesteps(
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            der_model_set_new_setpoints: fledge.der_models.DERModelSet,
            timesteps
    ):
        der_power_vector = (
            pd.DataFrame(columns=electric_grid_model.ders, index=timesteps, dtype=np.complex)
        )
        # Obtain nominal DER power vector based on set points from the optimal power flows
        for der in electric_grid_model.ders:
            der_name = der[1]
            der_power_vector.loc[:, der] = (
                    der_model_set_new_setpoints.der_models[der_name].active_power_nominal_timeseries
                    + (1.0j * der_model_set_new_setpoints.der_models[der_name].reactive_power_nominal_timeseries)
            )
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
    def set_electric_grid_optimization_variables_based_on_power_flow(
            optimization_problem: fledge.utils.OptimizationProblem,
            electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
            power_flow_solution_reference,
            power_flow_solutions,
            timesteps
    ):
        # TODO: this must be adapted to absolute values (not "change")
        # set objective function variables to zero (power vector change is zero, as the solution is already correct)
        for timestep in timesteps:
            power_flow_solution = power_flow_solutions[timestep]
            loss_active_change = np.real(power_flow_solution.loss) - np.real(power_flow_solution_reference.loss)
            loss_reactive_change = np.imag(power_flow_solution.loss) - np.imag(power_flow_solution_reference.loss)
            optimization_problem.loss_active_change[timestep] = loss_active_change
            optimization_problem.loss_reactive_change[timestep] = loss_reactive_change
            for der_index, der in enumerate(electric_grid_model.ders):
                active_power_change = np.real(power_flow_solution.der_power_vector[der_index]) - \
                                      np.real(power_flow_solution_reference.der_power_vector[der_index])
                reactive_power_change = np.imag(power_flow_solution.der_power_vector[der_index]) - \
                                        np.imag(power_flow_solution_reference.der_power_vector[der_index])
                optimization_problem.der_active_power_vector_change[timestep, der].value = active_power_change
                optimization_problem.der_reactive_power_vector_change[timestep, der].value = reactive_power_change


class ScenarioHandler(object):
    @staticmethod
    def change_der_set_points_based_on_results(
            scenario_name: str,
            results: fledge.problems.Results
    ) -> fledge.der_models.DERModelSet:
        # Requirements: for every DER model it should return the active and reactive power output based on the results
        grid_data = fledge.data_interface.ElectricGridData(scenario_name)
        # Obtain DER model set
        der_model_set = fledge.der_models.DERModelSet(scenario_name)

        for der_name in der_model_set.der_names:
            der_model = der_model_set.der_models[der_name]
            der_type = der_model.der_type
            der_model.active_power_nominal_timeseries = \
                results.der_active_power_vector[der_type, der_name]
            der_model.reactive_power_nominal_timeseries = \
                results.der_reactive_power_vector[der_type, der_name]

        return der_model_set


class OptimalOperationProblem(object):
    """"Custom optimal operation problem object based on fledge.problems.OptimalOperationProblem, consisting of an optimization problem as well as the corresponding
    electric reference power flow solutions, linear grid models and DER model set
    for the given scenario.

    The main difference lies in the separation of the problem formulation from the init function
    """
    scenario_name: str
    timesteps: pd.Index
    price_data: fledge.data_interface.PriceData
    scenario_data: fledge.data_interface.ScenarioData
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: fledge.electric_grid_models.PowerFlowSolution = None
    linear_electric_grid_model: fledge.electric_grid_models.LinearElectricGridModel = None
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
            self.power_flow_solution_reference = (
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint(self.electric_grid_model)
            )
            self.linear_electric_grid_model = (
                fledge.electric_grid_models.LinearElectricGridModelGlobal(
                    self.electric_grid_model,
                    self.power_flow_solution_reference
                )
            )

        # Obtain DER model set.
        self.der_model_set = fledge.der_models.DERModelSet(scenario_name)

    def formulate_optimization_problem(self):
        # Instantiate optimization problem.
        self.optimization_problem = fledge.utils.OptimizationProblem()

        if self.electric_grid_model is not None:
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
            self,
            keep_problem=False
    ) -> bool:
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
                    self.power_flow_solution_reference,
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
    def __init__(
            self,
            scenario_name: str
    ):
        super().__init__(
            scenario_name=scenario_name,
            central=False
        )


class ElectricGridOptimalOperationProblem(OptimalOperationProblem):
    def __init__(
            self,
            scenario_name: str
    ):
        super().__init__(
            scenario_name=scenario_name,
            central=True
        )
