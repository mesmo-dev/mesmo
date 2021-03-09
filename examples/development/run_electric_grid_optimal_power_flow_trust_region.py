"""
Example script for setting up and solving an optimal power flow problem with trust-region algorithm.
In general, trust-region based algorithm mitigates the approximation inaccuracy. The algorithm improve the approximate
solution of the approximated quadratic program in an iterative manner.
The algorithm is based on the works in:
[1] Hanif et al. “Decomposition and Equilibrium Achieving Distribution Locational Marginal Prices using Trust-Region Method,”
IEEE Transactions on Smart Grid, pp. 1–1, 2018, doi: 10.1109/TSG.2018.2822766.
Trust-Region parameters are based on the works in:
[2] A. M. Giacomoni and B. F. Wollenberg, “Linear programming optimal power flow utilizing a trust region method,”
in North American Power Symposium 2010, Arlington, TX, USA, Sep. 2010, pp. 1–6, doi: 10.1109/NAPS.2010.5619970.
"""

import numpy as np
from datetime import datetime

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models
import fledge.utils

# Ignore division by zero or nan warnings (this can happen with e.g. DERs with zero reactive power output)
np.seterr(divide='ignore', invalid='ignore')
fledge.config.config['optimization']['show_solver_output'] = False


def main():

    # Settings.
    scenario_name = 'singapore_6node'

    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    print('Loading data...', end='\r')
    # fledge.data_interface.recreate_database()

    # Instantiate iteration variables.
    sigma = 0.0
    der_power_vector_change_per_unit_max = np.inf
    trust_region_iteration_count = 0
    power_flow_solutions_iter = []
    objective_power_flows_iter = []
    optimization_problems_iter = []

    # Define trust-region parameters according to [2].
    delta = 0.8  # 3.0 / 0.5 / range: (0, delta_max] / If too big, no power flow solution.
    delta_max = 2.0  # 4.0 / 1.0
    gamma = 0.5  # 0.5 / range: (0, 1)
    eta = 0.1  # 0.1 / range: (0, 0.5]
    tau = 0.1  # 0.1 / range: [0, 0.25)
    epsilon = 1.0e-3  # 1e-3 / 1e-4
    trust_region_iteration_limit = 100

    # Obtain models.
    electric_grid_model = (
        fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    )
    # Get scenario data and timesteps
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)
    timesteps = scenario_data.timesteps

    # ---------------------------------------------------------------------------------------------------------
    # Pre-solve optimal operation problem without underlying electric grid to get realistic initial values
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    pre_solve_der_results = der_model_set.pre_solve(price_data)

    # ---------------------------------------------------------------------------------------------------------
    # Obtain the base case power flow, using the active_power_nominal values given as input as initial dispatch
    # quantities. This represents the initial solution candidate.
    power_flow_solution_set_candidate = (
        fledge.electric_grid_models.PowerFlowSolutionSet(
            electric_grid_model,
            pre_solve_der_results
        )
    )

    # ---------------------------------------------------------------------------------------------------------
    # Start trust-region iterations
    start_time = datetime.now()
    while (
            (der_power_vector_change_per_unit_max > epsilon)
            and (trust_region_iteration_count < trust_region_iteration_limit)
    ):

        # Print progress.
        print('------------------------------------------------------------------')
        print(f"Starting trust-region iteration #{trust_region_iteration_count}")

        # Check trust-region solution acceptance conditions.
        if (trust_region_iteration_count == 0) or (sigma > tau):
            if trust_region_iteration_count != 0:
                print('sigma > tau -> Accepting iteration, setting new states.')

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
                    np.real(electric_grid_model.der_power_vector_reference)
            )
            der_reactive_power_vector_reference = np.nan_to_num(
                    np.imag(power_flow_solution_set.der_power_vector) /
                    np.imag(electric_grid_model.der_power_vector_reference)
            )

            # Get the new reference values for voltage and branch flow which are used in the Trust-Region constraints
            node_voltage_vector_reference = power_flow_results.node_voltage_magnitude_vector_per_unit
            branch_power_magnitude_vector_1_reference = power_flow_results.branch_power_magnitude_vector_1_per_unit
            branch_power_magnitude_vector_2_reference = power_flow_results.branch_power_magnitude_vector_2_per_unit

            # Get linear electric grid model for all timesteps
            linear_electric_grid_model_set = (
                fledge.electric_grid_models.LinearElectricGridModelSet(
                    electric_grid_model,
                    power_flow_solution_set
                )
            )

            # Evaluate the objective function based on power flow results and store objective value.
            objective_power_flows_iter.append(
                linear_electric_grid_model_set.evaluate_optimization_objective(
                    power_flow_results,
                    price_data
                )
            )
        else:
            print('sigma <= tau -> Rejecting iteration. Repeating iteration using the modified region (delta).')

        # ---------------------------------------------------------------------------------------------------------
        print('Formulating optimization problem...', end='\r')
        # Instantiate / reset optimization problem.
        optimization_problem = fledge.utils.OptimizationProblem()

        # Define linear electric grid model variables.
        # The variables of the linearized electric grid model are independent of the linearization,
        #  --> no need to define them for every timestep
        linear_electric_grid_model_set.define_optimization_variables(
            optimization_problem=optimization_problem
        )

        # Define DER model variables
        der_model_set.define_optimization_variables(
            optimization_problem=optimization_problem
        )

        # Define linear electric grid model constraints.
        # TODO: adapt to what we actually want as limits
        node_voltage_magnitude_vector_minimum = 0.9 * np.abs(electric_grid_model.node_voltage_vector_reference)
        node_voltage_magnitude_vector_maximum = 1.1 * np.abs(electric_grid_model.node_voltage_vector_reference)
        branch_power_magnitude_vector_maximum = 2.0 * electric_grid_model.branch_power_vector_magnitude_reference
        # The linear electric grid model is different for every timestep
        linear_electric_grid_model_set.define_optimization_constraints(
            optimization_problem=optimization_problem,
            node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
            node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
            branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
        )

        # set_custom_der_constraints(
        #     der_model_set=der_model_set_reference
        # )

        # Define DER constraints.
        der_model_set.define_optimization_constraints(
            optimization_problem=optimization_problem,
            electric_grid_model=electric_grid_model
        )

        # ---------------------------------------------------------------------------------------------------------
        # Define trust region constraints.
        # The trust-region permissible value for variables to move is determined by radius delta, which is included
        # in all inequality constraints [1].
        # -> Branch flow and voltage limits
        # -> DER power output limits
        # TODO: actually over the entire output vector?
        # We redefine the approximate state and dispatch quantities as the measure of change in their
        # operating state at the current iteration.
        # Voltage.
        optimization_problem.constraints.append(
            optimization_problem.node_voltage_magnitude_vector
            - np.abs(node_voltage_vector_reference).to_numpy()
            >=
            -delta
        )
        optimization_problem.constraints.append(
            optimization_problem.node_voltage_magnitude_vector
            - np.abs(node_voltage_vector_reference).to_numpy()
            <=
            delta
        )

        # Branch flows.
        optimization_problem.constraints.append(
            optimization_problem.branch_power_magnitude_vector_1
            - np.abs(branch_power_magnitude_vector_1_reference).to_numpy()
            >=
            -delta
        )
        optimization_problem.constraints.append(
            optimization_problem.branch_power_magnitude_vector_1
            - np.abs(branch_power_magnitude_vector_1_reference).to_numpy()
            <=
            delta
        )
        optimization_problem.constraints.append(
            optimization_problem.branch_power_magnitude_vector_2
            - np.abs(branch_power_magnitude_vector_2_reference).to_numpy()
            >=
            -delta
        )
        optimization_problem.constraints.append(
            optimization_problem.branch_power_magnitude_vector_2
            - np.abs(branch_power_magnitude_vector_2_reference).to_numpy()
            <=
            delta
        )

        # DERs.
        optimization_problem.constraints.append(
            optimization_problem.der_active_power_vector
            - der_active_power_vector_reference
            >=
            -delta
        )
        optimization_problem.constraints.append(
            optimization_problem.der_active_power_vector
            - der_active_power_vector_reference
            <=
            delta
        )
        optimization_problem.constraints.append(
            optimization_problem.der_reactive_power_vector
            - der_reactive_power_vector_reference
            <=
            delta
        )
        optimization_problem.constraints.append(
            optimization_problem.der_reactive_power_vector
            - der_reactive_power_vector_reference
            >=
            -delta
        )

        # ---------------------------------------------------------------------------------------------------------
        # Define objective and solve.
        # NOTE: the objective is independent of the linearized power flow, hence we can use any of the models
        linear_electric_grid_model_set.define_optimization_objective(
            optimization_problem,
            price_data,
        )

        # Solve optimization problem.
        print('Solving optimal power flow...', end='\r')
        optimization_problem.solve()
        optimization_problems_iter.append(optimization_problem)  # TODO: really needed?
        optimization_results = fledge.problems.Results()
        optimization_results.update(
            linear_electric_grid_model_set.get_optimization_results(
                optimization_problem
            )
        )
        optimization_results.update(
            der_model_set.get_optimization_results(
                optimization_problem
            )
        )

        # ---------------------------------------------------------------------------------------------------------
        # Trust-region evaluation and update
        print('Trust-region evaluation and update...', end='\r')
        # Obtain der power change value.
        der_active_power_vector_change_per_unit = (
            np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=np.float)
        )
        der_reactive_power_vector_change_per_unit = (
            np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=np.float)
        )

        for der_index, der in enumerate(electric_grid_model.ders):
            der_active_power_vector_change_per_unit[:, der_index] = (
                optimization_results.der_active_power_vector_per_unit.loc[:, der]
                - der_active_power_vector_reference[:, der_index]
            )
            der_reactive_power_vector_change_per_unit[:, der_index] = (
                optimization_results.der_reactive_power_vector_per_unit.loc[:, der]
                - der_reactive_power_vector_reference[:, der_index]
            )
        der_power_vector_change_per_unit_max = (
            max(
                np.max(abs(der_active_power_vector_change_per_unit)),
                np.max(abs(der_reactive_power_vector_change_per_unit))
            )
        )

        # Print change variable.
        print(f"der_power_vector_change_per_unit_max = {der_power_vector_change_per_unit_max}")

        # ---------------------------------------------------------------------------------------------------------
        # Check trust-region conditions and obtain DER power vector / power flow solution candidates for next iteration.
        # - Only if termination condition is not met, otherwise risk of division by zero.
        if der_power_vector_change_per_unit_max > epsilon:
            # Get new power flow solution candidate
            power_flow_solution_set_candidate = (
                fledge.electric_grid_models.PowerFlowSolutionSet(
                    electric_grid_model,
                    optimization_results
                )
            )

            # Obtain objective values.
            objective_linear_model = float(
                optimization_problem.objective.value
            )

            power_flow_results_candidate = power_flow_solution_set_candidate.get_results()

            objective_power_flow = (
                linear_electric_grid_model_set.evaluate_optimization_objective(
                    power_flow_results_candidate,
                    price_data
                )
            )

            # Check if power flow of candidate is violating line limits, to then increase the radius delta
            # This is only to save some iterations
            pf_violation_flag_1 = (
                    np.abs(branch_power_magnitude_vector_maximum
                           - power_flow_results_candidate.branch_power_magnitude_vector_1) < 0).any(axis=None)
            pf_violation_flag_2 = (
                    np.abs(branch_power_magnitude_vector_maximum
                           - power_flow_results_candidate.branch_power_magnitude_vector_2) < 0).any(axis=None)

            # ---------------------------------------------------------------------------------------------------------
            # Evaluate solution progress.
            # sigma represents the ratio between the cost improvement of approximated system to the actual one. A
            # smaller value of sigma shows that the current approximation does not represent the actual system and hence
            # the the optimization region must be reduced. For a considerably higher value of sigma, the linear
            # approximation is accurate and the system can move to a new operating point. [1]
            sigma = float(
                (objective_power_flows_iter[-1] - objective_power_flow)
                / (objective_power_flows_iter[-1] - objective_linear_model)
            )

            if pf_violation_flag_1 or pf_violation_flag_2:  # first check if there are any line flow violations
                delta *= gamma
            elif (objective_power_flows_iter[-1] - objective_linear_model) <= 0:  # see code Hanif
                delta *= gamma
            elif sigma <= eta:
                delta *= gamma
            elif sigma > (1.0 - eta):
                delta = min(2 * delta, delta_max)

            # Print trust-region parameters.
            print(f"objective_power_flow = {objective_power_flow}")
            print(f"objective_linear_model = {objective_linear_model}")
            print(f"objective_power_flows_iter[-1] = {objective_power_flows_iter[-1]}")
            print(f"sigma = {sigma}")
            print(f"delta = {delta}")

        # Iterate counter.
        trust_region_iteration_count += 1

    # ---------------------------------------------------------------------------------------------------------
    print('----------------------------------------------------------')
    print('Found solution, exiting the trust region iterations.')
    end_time = datetime.now()

    # ---------------------------------------------------------------------------------------------------------
    # Obtain results.
    # NOTE: which lin electric grid model and power flow solution is the correct one to pass to the results?
    # --> the result method of lin electric grid model is independent of the actual model, so it is irrelevant
    results = fledge.problems.Results()
    results.update(
        linear_electric_grid_model_set.get_optimization_results(
            optimization_problem
        )
    )
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    # print(results)

    # Store results to CSV.
    results.save(results_path)

    # Obtain DLMPs.
    dlmps = (
        linear_electric_grid_model_set.get_optimization_dlmps(
            optimization_problem,
            price_data
        )
    )

    # Print DLMPs.
    # print(dlmps)

    # Store DLMPs to CSV.
    dlmps.save(results_path)

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")
    print(f'Time elapsed for trust region: {(end_time - start_time)}')
    print(f'Trust region iterations: {trust_region_iteration_count}')


def set_custom_der_constraints(
        der_model_set: fledge.der_models.DERModelSet,
        factor: float = 1.0
):
    """
    Function that must be called BEFORE defining the DER constraints. Here, any custom DER constraints on the output
    can be defined.
    Args:
        der_model_set: DERModelSet to constrain
        factor: optional, maximum output is multiplied by this, defaults to 1
    """
    # TODO: define these with input parameters?
    # Additional constraint for flexible buildings
    for der_name in der_model_set.der_models.keys():
        der_model = der_model_set.der_models[der_name]
        if type(der_model) is fledge.der_models.FlexibleBuildingModel:
            # Limit loads to their nominal power consumption
            der_model.output_maximum_timeseries['grid_electric_power'] = (
                    float(der_model.active_power_nominal
                          / der_model.mapping_active_power_by_output['grid_electric_power'])
                    * np.ones([len(der_model.output_maximum_timeseries), 1])
                    * factor
            )
            # Put a constraint on cooling power (= 0) to effectively disable cooling in the HVAC system
            if 'zone_generic_cool_thermal_power' in der_model.output_maximum_timeseries.columns:
                der_model.output_maximum_timeseries['zone_generic_cool_thermal_power'] = (
                    np.zeros([len(der_model.output_maximum_timeseries), 1])
                )


if __name__ == '__main__':
    main()
