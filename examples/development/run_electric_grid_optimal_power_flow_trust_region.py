"""Example script for setting up and solving an optimal power flow problem with trust-region algorithm."""

import numpy as np
import pandas as pd
from datetime import datetime

import mesmo


def main():

    # Settings.
    scenario_name = "singapore_6node"
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    print("Loading data...", end="\r")
    mesmo.data_interface.recreate_database()

    # Obtain data / models.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name)
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModel(scenario_name)
    linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)
    presolve_results = der_model_set.pre_solve(price_data)

    [objective, results, linear_electric_grid_model_set] = run_optimal_operation_problem_trust_region(
        scenario_data,
        price_data,
        der_model_set,
        electric_grid_model,
        linear_electric_grid_model_set,
        presolve_results,
        results_path,
    )

    print("done")


def run_optimal_operation_problem_trust_region(
    scenario_data: mesmo.data_interface.ScenarioData,
    price_data: mesmo.data_interface.PriceData,
    der_model_set: mesmo.der_models.DERModelSet,
    electric_grid_model: mesmo.electric_grid_models.ElectricGridModel,
    linear_electric_grid_model_set: mesmo.electric_grid_models.LinearElectricGridModelSet = None,
    presolve_results: mesmo.der_models.DERModelSetOperationResults = None,
    results_path: str = None,
) -> [float, mesmo.problems.Results, dict]:
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
    Args:
        scenario_data:
        price_data:
        der_model_set:
        electric_grid_model: optional, if passed, an OPF problem is formulated
        presolve_results: presolve results from another iteration, otherwise it is calculated again
        linear_electric_grid_model_set: linearization from previous iteration, otherwise re-calculated
        results_path: ptional, when set, then results are stored as CSVs including PKL-files

    Returns:
        problem.optimization_problem.objective.value: objective value of optimal operation problem
        results: Results object
        problem.linear_electric_grid_model_set: dictionary linear electric grid model for all timesteps
    """
    # TODO: these should become attributes of the class
    # Instantiate iteration variables.
    sigma = 0.0
    error_control = np.inf
    trust_region_iteration_count = 0
    trust_region_accepted_iteration_count = 0
    infeasible_count = 0
    power_flow_solutions_iter = []
    objective_power_flows_iter = []

    # Define trust-region parameters according to [2].
    delta = 1.0  # 3.0 / 0.5 / range: (0, delta_max] / If too big, no power flow solution.
    delta_max = 4.0  # 4.0 / 1.0
    gamma = 0.25  # 0.5 / range: (0, 1)
    eta = 0.1  # 0.1 / range: (0, 0.25]
    tau = 0.1  # 0.1 / range: (0, 0.25]
    epsilon = 1.0e-4  # 1e-3 / 1e-4
    trust_region_iteration_limit = 30
    infeasible_iteration_limit = (
        3  # the maximum number of iterations to try to solve the optimization with different deltas
    )

    print(f"Solving problem with trust-region algorithm.")
    # Ignore division by zero or nan warnings (this can happen with e.g. DERs with zero reactive power output)
    np.seterr(divide="ignore", invalid="ignore")

    scenario_name = scenario_data.scenario["scenario_name"]

    if presolve_results is None:
        presolve_results = der_model_set.pre_solve(price_data)

    # ---------------------------------------------------------------------------------------------------------
    # Instantiate problem object (only once, the parameters will only be updated)
    problem = ElectricGridOptimalOperationProblem(scenario_data, price_data, der_model_set, electric_grid_model)

    # ---------------------------------------------------------------------------------------------------------
    # Obtain the base case power flow, using the active_power_nominal values given as input as initial dispatch
    # quantities. This represents the initial solution candidate.
    print("Obtaining first best guess for optimal power flow...", end="\r")
    if linear_electric_grid_model_set is None:
        # Obtain power flow solution based on presolve from DERs.
        power_flow_solution_set = mesmo.electric_grid_models.PowerFlowSolutionSet(
            electric_grid_model,
            presolve_results,
        )
        linear_electric_grid_model_set = get_linear_electric_grid_model_set(
            electric_grid_model,
            power_flow_solution_set,
        )

    # Formulate problem and solve
    problem.linear_electric_grid_model_set = linear_electric_grid_model_set
    problem.formulate_optimization_problem()
    feasible = problem.solve()
    if feasible:
        best_guess_results = problem.get_results()
        print("Calculating power flow based on first best guess results")
        # If problem infeasible, use presolve results and start trust region algorithm
        presolve_results = best_guess_results

    print("Obtaining initial power flow solution set candidate...", end="\r")
    # Obtain updated power flow solution set as initial candidate
    power_flow_solution_set_candidate = mesmo.electric_grid_models.PowerFlowSolutionSet(
        electric_grid_model,
        presolve_results,
    )

    # ---------------------------------------------------------------------------------------------------------
    # Start trust-region iterations
    start_time = datetime.now()
    first_iter = True

    print("----------------------------------------------------------------------------------------")
    while (
        trust_region_accepted_iteration_count
        < trust_region_iteration_limit
        # and (error_control > epsilon)
    ):
        iter_start_time = datetime.now()
        # Print progress.
        print(f"Starting trust-region iteration #{trust_region_iteration_count}")
        print(f"Accepted iterations: {trust_region_accepted_iteration_count}")

        # Define / update delta parameter for trust region constraints.
        define_trust_region_delta_parameter(
            optimization_problem=problem.optimization_problem,
            delta=delta,
        )

        # Check trust-region solution acceptance conditions.
        if first_iter or (sigma > tau):
            print("Setting new states.")
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
                np.real(power_flow_solution_set.der_power_vector)
                / np.real(electric_grid_model.der_power_vector_reference)
            )
            der_reactive_power_vector_reference = np.nan_to_num(
                np.imag(power_flow_solution_set.der_power_vector)
                / np.imag(electric_grid_model.der_power_vector_reference)
            )

            # Get the new reference values for voltage and branch flow which are used in the Trust-Region constraints
            node_voltage_vector_reference = power_flow_results.node_voltage_magnitude_vector_per_unit
            branch_power_magnitude_vector_1_reference = power_flow_results.branch_power_magnitude_vector_1_per_unit
            branch_power_magnitude_vector_2_reference = power_flow_results.branch_power_magnitude_vector_2_per_unit

            # Get linear electric grid model for all timesteps
            print("Obtaining linear electric grid model set...", end="\r")
            linear_electric_grid_model_set = get_linear_electric_grid_model_set(
                electric_grid_model,
                power_flow_solution_set,
            )

            problem.linear_electric_grid_model_set = linear_electric_grid_model_set
            # Update the parameters so that sensitivity matrices are updated
            problem.update_linear_electric_grid_model_parameters()

            # Evaluate the objective function based on power flow results and store objective value.
            objective_power_flows_iter.append(
                linear_electric_grid_model_set.evaluate_optimization_objective(power_flow_results, price_data)
            )

            # Define / update parameters for trust region iteration
            define_trust_region_electric_grid_parameters(
                optimization_problem=problem.optimization_problem,
                node_voltage_vector_reference=node_voltage_vector_reference,
                branch_power_magnitude_vector_1_reference=branch_power_magnitude_vector_1_reference,
                branch_power_magnitude_vector_2_reference=branch_power_magnitude_vector_2_reference,
                der_active_power_vector_reference=der_active_power_vector_reference,
                der_reactive_power_vector_reference=der_reactive_power_vector_reference,
            )
            if first_iter:  # This is only done on this first iteration
                # Define trust region constraints
                define_trust_region_constraints(
                    optimization_problem=problem.optimization_problem, timesteps=problem.timesteps
                )

            # After first iteration, set to False
            first_iter = False

        # Solve the optimization problem
        feasible = problem.solve()
        if not feasible:
            infeasible_count += 1
            if delta >= delta_max or infeasible_count > infeasible_iteration_limit:
                print(f"Optimization problem for scenario {scenario_name} infeasible")
                return [None, None, problem.linear_electric_grid_model_set]
            else:
                print(f"Optimization problem infeasible, increasing delta to maximum")
                print(f"Trying to solve again #{infeasible_count}")
                # delta = min(2 * delta, delta_max)
                print(f"new delta = {delta}")
                delta = delta_max
                continue

        # Obtain results.
        optimization_results = problem.get_results()

        # ---------------------------------------------------------------------------------------------------------
        # Trust-region evaluation and update.
        print("Trust-region evaluation and update...", end="\r")
        # Obtain der power change value.
        der_active_power_vector_change_per_unit = (
            optimization_results.der_active_power_vector_per_unit - der_active_power_vector_reference
        ).to_numpy()
        der_reactive_power_vector_change_per_unit = (
            optimization_results.der_reactive_power_vector_per_unit - der_reactive_power_vector_reference
        ).to_numpy()

        der_power_vector_change_per_unit_max = max(
            np.max(abs(der_active_power_vector_change_per_unit)), np.max(abs(der_reactive_power_vector_change_per_unit))
        )
        error_control = der_power_vector_change_per_unit_max

        # ---------------------------------------------------------------------------------------------------------
        # Check trust-region conditions and obtain DER power vector / power flow solution candidates for next iteration.
        # - Only if termination condition is not met, otherwise risk of division by zero.
        # if der_power_vector_change_per_unit_max > epsilon:x
        # Get new power flow solution candidate
        print("Obtaining power flow solution set candidate...", end="\r")
        power_flow_solution_set_candidate = mesmo.electric_grid_models.PowerFlowSolutionSet(
            electric_grid_model, optimization_results
        )

        # Obtain objective values.
        objective_linear_model = problem.optimization_problem.evaluate_objective(problem.optimization_problem.x_vector)
        objective_linear_model = linear_electric_grid_model_set.evaluate_optimization_objective(
            optimization_results, price_data
        )

        # Get power flow solutions for candidate
        power_flow_results_candidate = power_flow_solution_set_candidate.get_results()

        # Evaluate the optimization objective
        objective_power_flow = linear_electric_grid_model_set.evaluate_optimization_objective(
            power_flow_results_candidate, price_data
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
            print("ZeroDivisionError in calculating sigma value.")
            sigma_numerator = objective_power_flows_iter[-1] - objective_power_flow
            if sigma_numerator == 0:  # TODO: does this case really exist? should it evaluate to zero or 1?
                sigma = 0  # this means, no progress has been done, so something should happen --> decrease delta
            elif sigma_numerator < 0:
                sigma = (-1) * np.inf
            else:
                sigma = np.inf
            print(f"Evaluated numerator, falling back to sigma = {sigma}")

        # Print trust-region progress
        print(f"objective_power_flow = {objective_power_flow}")
        print(f"objective_linear_model = {objective_linear_model}")
        print(f"control error = {error_control}")
        print(f"objective error = {error_obj_function}")
        print(f"sigma = {sigma}")

        # if pf_violation_flag_1 or pf_violation_flag_2:  # first check if there are any line flow violations
        #     print('Found line flow violation, decreasing delta.')
        #     delta *= gamma
        # If the new objective value is greater than the current value ([-1]), the step must be rejected, see [4]
        # elif (objective_power_flows_iter[-1] - objective_linear_model) <= 0:
        #     print('New objective larger than objective[-1]')
        #     delta *= gamma
        if sigma < eta:
            print("sigma < eta, linearized model is a bad approximation of the nonlinear model, decreasing delta")
            delta *= gamma
            print(f"new delta = {delta}")
        elif sigma > (1.0 - eta) and np.abs(der_power_vector_change_per_unit_max - delta) <= epsilon:
            # elif sigma > (1.0 - eta) and np.abs(voltage_magnitude_change - delta) <= epsilon:
            print(
                "sigma > (1.0 - eta), linearized model is a good approximation of the nonlinear model, increasing delta"
            )
            delta = min(2 * delta, delta_max)
            print(f"new delta = {delta}")
        else:
            # If the step stays strictly inside the region, we infer that the current value of delta is not
            # interfering with the progress of the algorithm, so we leave its value unchanged for the next iteration
            # see [3]
            print("linearized model is a satisfactory approximation of the nonlinear model, delta remains unchanged.")
            print(f"delta = {delta}")

        if sigma > tau:
            print(
                "sigma > tau -> the solution to the current iteration makes satisfactory progress toward the "
                "optimal solution"
            )
            print("Accepting iteration.")
            # Increase counter
            trust_region_accepted_iteration_count += 1

        else:
            print("sigma <= tau -> Rejecting iteration. Repeating iteration using the modified region (delta).")

        trust_region_iteration_count += 1
        print(f"Time elapsed for iteration: {datetime.now()-iter_start_time}")
        print("----------------------------------------------------------------------------------------")

    # ---------------------------------------------------------------------------------------------------------
    end_time = datetime.now()
    print("Found solution, exiting the trust region iterations.")
    print("----------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------")
    # TODO:
    print(f"Trust region iterations: {trust_region_iteration_count}")
    print(f"Total time elapsed for trust region algorithm: {(end_time - start_time)}")

    # ---------------------------------------------------------------------------------------------------------
    # Obtain results.
    # NOTE: which lin electric grid model and power flow solution is the correct one to pass to the results?
    # --> the result method of lin electric grid model is independent of the actual model, so it is irrelevant
    results = problem.get_results()

    # Store results to CSV.
    if results_path is not None:
        results.save(results_path)
        # Print results path.
        # fledge.utils.launch(results_path)
        print(f"Results are stored in: {results_path}")

    objective = problem.optimization_problem.evaluate_objective(problem.optimization_problem.x_vector)

    return [objective, results, problem.linear_electric_grid_model_set]


def get_linear_electric_grid_model_set(
    electric_grid_model: mesmo.electric_grid_models.ElectricGridModel,
    power_flow_solution_set: mesmo.electric_grid_models.PowerFlowSolutionSet,
):
    try:
        linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model, power_flow_solution_set, mesmo.electric_grid_models.LinearElectricGridModelLocal
        )
    except RuntimeError:
        print("*********** COULD NOT PERFORM LOCAL APPROXIMATION ***********")
        print("Defaulting to global approximation.")
        linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model, power_flow_solution_set, mesmo.electric_grid_models.LinearElectricGridModelGlobal
        )
    return linear_electric_grid_model_set


def define_trust_region_delta_parameter(
    optimization_problem: mesmo.solutions.OptimizationProblem,
    delta: float,
):
    optimization_problem.define_parameter(name="delta_positive", value=delta)
    optimization_problem.define_parameter(name="delta_negative", value=(-1) * delta)


def define_trust_region_electric_grid_parameters(
    optimization_problem: mesmo.solutions.OptimizationProblem,
    node_voltage_vector_reference,
    branch_power_magnitude_vector_1_reference,
    branch_power_magnitude_vector_2_reference,
    der_active_power_vector_reference,
    der_reactive_power_vector_reference,
):

    optimization_problem.define_parameter(
        name="node_voltage_vector_reference", value=(-1) * node_voltage_vector_reference.to_numpy().ravel()
    )
    optimization_problem.define_parameter(
        name="branch_power_magnitude_vector_1_reference",
        value=(-1) * branch_power_magnitude_vector_1_reference.to_numpy().ravel(),
    )
    optimization_problem.define_parameter(
        name="branch_power_magnitude_vector_2_reference",
        value=(-1) * branch_power_magnitude_vector_2_reference.to_numpy().ravel(),
    )
    optimization_problem.define_parameter(
        name="der_active_power_vector_reference", value=(-1) * der_active_power_vector_reference.ravel()
    )
    optimization_problem.define_parameter(
        name="der_reactive_power_vector_reference", value=(-1) * der_reactive_power_vector_reference.ravel()
    )


def define_trust_region_constraints(optimization_problem: mesmo.solutions.OptimizationProblem, timesteps: pd.Index):
    # Define trust region constraints.
    # The trust-region permissible value for variables to move is determined by radius delta, which is included
    # in all inequality constraints [1].
    # -> Branch flow and voltage limits
    # -> DER power output limits
    # We redefine the approximate state and dispatch quantities as the measure of change in their
    # operating state at the current iteration.

    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="node_voltage_magnitude_vector", timestep=timesteps)),
        ("constant", "node_voltage_vector_reference", dict(timestep=timesteps)),
        "<=",
        ("constant", "delta_positive", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="node_voltage_magnitude_vector", timestep=timesteps)),
        ("constant", "node_voltage_vector_reference", dict(timestep=timesteps)),
        ">=",
        ("constant", "delta_negative", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="branch_power_magnitude_vector_1", timestep=timesteps)),
        ("constant", "branch_power_magnitude_vector_1_reference", dict(timestep=timesteps)),
        "<=",
        ("constant", "delta_positive", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="branch_power_magnitude_vector_1", timestep=timesteps)),
        ("constant", "branch_power_magnitude_vector_1_reference", dict(timestep=timesteps)),
        ">=",
        ("constant", "delta_negative", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="branch_power_magnitude_vector_2", timestep=timesteps)),
        ("constant", "branch_power_magnitude_vector_2_reference", dict(timestep=timesteps)),
        "<=",
        ("constant", "delta_positive", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="branch_power_magnitude_vector_2", timestep=timesteps)),
        ("constant", "branch_power_magnitude_vector_2_reference", dict(timestep=timesteps)),
        ">=",
        ("constant", "delta_negative", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="der_active_power_vector", timestep=timesteps)),
        ("constant", "der_active_power_vector_reference", dict(timestep=timesteps)),
        "<=",
        ("constant", "delta_positive", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="der_active_power_vector", timestep=timesteps)),
        ("constant", "der_active_power_vector_reference", dict(timestep=timesteps)),
        ">=",
        ("constant", "delta_negative", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="der_reactive_power_vector", timestep=timesteps)),
        ("constant", "der_reactive_power_vector_reference", dict(timestep=timesteps)),
        "<=",
        ("constant", "delta_positive", dict(timestep=timesteps)),
    )
    optimization_problem.define_constraint(
        ("variable", 1.0, dict(name="der_reactive_power_vector", timestep=timesteps)),
        ("constant", "der_reactive_power_vector_reference", dict(timestep=timesteps)),
        ">=",
        ("constant", "delta_negative", dict(timestep=timesteps)),
    )


class OptimalOperationProblem(object):
    """ "Custom optimal operation problem object based on fledge.problems.OptimalOperationProblem, consisting of an
    optimization problem as well as the corresponding electric reference power flow solutions, linear grid models and
    DER model set for the given scenario.

    The main difference lies in the separation of the problem formulation from the init function and the possibility
    to linearize the electric grid model based on a power flow solution for each timestep
    """

    scenario_name: str
    timesteps: pd.Index
    price_data: mesmo.data_interface.PriceData
    scenario_data: mesmo.data_interface.ScenarioData
    electric_grid_model: mesmo.electric_grid_models.ElectricGridModel = None
    power_flow_solution_set: mesmo.electric_grid_models.PowerFlowSolutionSet = None
    linear_electric_grid_model_set: mesmo.electric_grid_models.LinearElectricGridModelSet = None
    der_model_set: mesmo.der_models.DERModelSet
    presolve_results: mesmo.der_models.DERModelSetOperationResults = None
    optimization_problem: mesmo.solutions.OptimizationProblem
    results: mesmo.problems.Results

    def __init__(
        self,
        scenario_data: mesmo.data_interface.ScenarioData,
        price_data: mesmo.data_interface.PriceData,
        der_model_set: mesmo.der_models.DERModelSet,
        electric_grid_model: mesmo.electric_grid_models.ElectricGridModel = None,
    ):
        # Obtain data.
        self.scenario_data = scenario_data
        self.price_data = price_data

        # Store timesteps.
        self.timesteps = self.scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        self.electric_grid_model = electric_grid_model

        # Obtain DER model set.
        self.der_model_set = der_model_set

    def formulate_optimization_problem(
        self, linear_electric_grid_model_method=mesmo.electric_grid_models.LinearElectricGridModelGlobal
    ):
        """
        Function formulates the optimization problem by defining variables and constraints
        Args:
            linear_electric_grid_model_method:
            scenario_handler: object handling custom constraints of the current scenario
        """
        print("Formulating optimization problem...", end="\r")
        # Instantiate optimization problem.
        self.optimization_problem = mesmo.solutions.OptimizationProblem()

        if self.electric_grid_model is not None:
            # ---------------------------------------------------------------------------------------------------------
            # POWER FLOW AND LINEAR ELECTRIC GRID MODEL
            # Obtain the base power flow, using the values from the presolved optmization given as input as initial
            # dispatch quantities.
            if self.linear_electric_grid_model_set is None:
                if linear_electric_grid_model_method is mesmo.electric_grid_models.LinearElectricGridModelLocal:
                    if self.presolve_results is None:
                        self.presolve_results = self.der_model_set.pre_solve(self.price_data)
                    self.power_flow_solution_set = mesmo.electric_grid_models.PowerFlowSolutionSet(
                        self.electric_grid_model, self.presolve_results
                    )
                    # Get linear electric grid model for all timesteps
                    self.linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(
                        self.electric_grid_model, self.power_flow_solution_set, linear_electric_grid_model_method
                    )
                else:
                    # If global approximation, use the reference power flow of the electric grid model
                    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(
                        self.electric_grid_model
                    )
                    self.linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(
                        self.electric_grid_model,
                        power_flow_solution,
                        linear_electric_grid_model_method=linear_electric_grid_model_method,
                    )

            # Define linear electric grid model variables and constraints.
            self.linear_electric_grid_model_set.define_optimization_problem(self.optimization_problem, self.price_data)

        # Define DER variables and constraints.
        self.der_model_set.define_optimization_problem(self.optimization_problem, self.price_data)

    def update_linear_electric_grid_model_parameters(self):

        # Check if ScenarioData contains information on limits, otherwise set to None (no limits).
        node_voltage_magnitude_vector_minimum = (
            self.scenario_data.scenario["voltage_per_unit_minimum"]
            * np.abs(self.electric_grid_model.node_voltage_vector_reference)
            if pd.notnull(self.scenario_data.scenario["voltage_per_unit_minimum"])
            else None
        )
        node_voltage_magnitude_vector_maximum = (
            self.scenario_data.scenario["voltage_per_unit_maximum"]
            * np.abs(self.electric_grid_model.node_voltage_vector_reference)
            if pd.notnull(self.scenario_data.scenario["voltage_per_unit_maximum"])
            else None
        )
        branch_power_magnitude_vector_maximum = (
            self.scenario_data.scenario["branch_flow_per_unit_maximum"]
            * self.electric_grid_model.branch_power_vector_magnitude_reference
            if pd.notnull(self.scenario_data.scenario["branch_flow_per_unit_maximum"])
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

    def solve(self) -> bool:
        # print('Solving optimization problem...', end='\r')
        # Catch potential error so that simulation does not stop
        # Return if solution was found or not (True / False)
        try:
            self.optimization_problem.solve()
        except RuntimeError:
            print("######### COULD NOT SOLVE OPTIMIZATION #########")
            # return that optimization was infeasible
            return False

        # return that solution could be found
        return True

    def get_results(self) -> mesmo.problems.Results:
        # Instantiate results.
        self.results = mesmo.problems.Results(price_data=self.price_data)

        # Obtain electric grid results.
        if self.electric_grid_model is not None:
            self.results.update(self.linear_electric_grid_model_set.get_optimization_results(self.optimization_problem))

        # Obtain DER results.
        self.results.update(self.der_model_set.get_optimization_results(self.optimization_problem))

        # Obtain electric DLMPs.
        if self.electric_grid_model is not None:
            # Get one version of the dlmp results which is then updated
            self.results.update(
                self.linear_electric_grid_model_set.get_optimization_dlmps(self.optimization_problem, self.price_data)
            )

        return self.results


class DEROptimalOperationProblem(OptimalOperationProblem):
    def __init__(
        self,
        scenario_data: mesmo.data_interface.ScenarioData,
        price_data: mesmo.data_interface.PriceData,
        der_model_set: mesmo.der_models.DERModelSet,
    ):
        # Inherits all functionality from its super class OptimalOperationProblem
        # Passes electric_grid_model as None, so the problem is defined as decentral

        # for der_name in der_model_set.der_names:
        #     der_model_set.der_models[der_name].is_electric_grid_connected = False

        super().__init__(scenario_data, price_data, der_model_set, electric_grid_model=None)


class ElectricGridOptimalOperationProblem(OptimalOperationProblem):
    def __init__(
        self,
        scenario_data: mesmo.data_interface.ScenarioData,
        price_data: mesmo.data_interface.PriceData,
        der_model_set: mesmo.der_models.DERModelSet,
        electric_grid_model: mesmo.electric_grid_models.ElectricGridModel,
    ):
        # for der_name in der_model_set.der_names:
        #     der_model_set.der_models[der_name].is_electric_grid_connected = True

        # Inherits all functionality from its super class OptimalOperationProblem
        super().__init__(scenario_data, price_data, der_model_set, electric_grid_model=electric_grid_model)


if __name__ == "__main__":
    main()
