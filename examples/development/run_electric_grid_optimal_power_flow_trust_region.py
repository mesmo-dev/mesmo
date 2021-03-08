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
import pandas as pd
import itertools
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
    fledge.data_interface.recreate_database()

    # Instantiate iteration variables.
    sigma = 0.0
    der_power_vector_change_per_unit_max = np.inf
    trust_region_iteration_count = 0
    power_flow_solutions_iter = []
    objective_power_flows_iter = []
    optimization_problems_iter = []

    # Define trust-region parameters according to [2].
    delta = 0.3  # 3.0 / 0.5 / range: (0, delta_max] / If too big, no power flow solution.
    delta_max = 1.0  # 4.0 / 1.0
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
    presolve_results = run_presolve_with_der_models(der_model_set, price_data)
    der_model_set_candidate = change_der_set_points_based_on_results(der_model_set, presolve_results)
    # der_model_set_candidate = fledge.der_models.DERModelSet(scenario_name)

    # ---------------------------------------------------------------------------------------------------------
    # Obtain the base case power flow, using the active_power_nominal values given as input as initial dispatch
    # quantities. This represents the initial solution candidate.
    # For the base power flow, we use the nominal power flow results
    power_flow_solutions_per_timestep_candidate = get_power_flow_solutions_per_timestep(
        electric_grid_model=electric_grid_model,
        der_model_set_new_setpoints=der_model_set_candidate,
        timesteps=timesteps
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
            der_model_set_reference = der_model_set_candidate
            power_flow_solutions_per_timestep = power_flow_solutions_per_timestep_candidate
            power_flow_solutions_iter.append(power_flow_solutions_per_timestep)

            # Get the new reference power vector for DERs based on the accepted candidate. This vector is different
            # from the one of the electric grid model, which is not adapted every iteration
            der_power_vector_reference = get_der_power_vector(electric_grid_model, der_model_set_reference, timesteps)

            # Get the node voltage and branch flow reference vector based on the accepted candidate
            node_voltage_vector_reference = get_node_voltage_vector_per_timestep(
                electric_grid_model,
                power_flow_solutions_per_timestep,
                timesteps
            )
            if trust_region_iteration_count != 0:
                # The branch power flow has already been calculated below, use existing instead of re-calculating
                branch_power_magnitude_vector_1_reference = branch_power_magnitude_vector_1_candidate
                branch_power_magnitude_vector_2_reference = branch_power_magnitude_vector_2_candidate
            else:
                branch_power_magnitude_vector_1_reference = get_branch_power_vector_per_timestep(
                    electric_grid_model,
                    power_flow_solutions_per_timestep,
                    timesteps,
                    1
                )
                branch_power_magnitude_vector_2_reference = get_branch_power_vector_per_timestep(
                    electric_grid_model,
                    power_flow_solutions_per_timestep,
                    timesteps,
                    2
                )

            # Get linear electric grid model for all timesteps
            linear_electric_grid_models_per_timestep = get_linear_electric_grid_models_per_timestep(
                electric_grid_model,
                power_flow_solutions_per_timestep,
                timesteps)

            # Instantiate optimization problem to evaluate the objective based on power flow solution setpoints
            # The problem will not be solved, it is just used to evaluate the objective
            optimization_problem = fledge.utils.OptimizationProblem()

            # Define optimization variables.
            # The variables of the linearized electric grid model are independent of the linearization
            #  --> no need to define them for every timestep
            linear_electric_grid_models_per_timestep[timesteps[0]].define_optimization_variables(
                optimization_problem=optimization_problem,
                timesteps=timesteps
            )
            der_model_set_reference.define_optimization_variables(
                optimization_problem
            )
            linear_electric_grid_models_per_timestep[timesteps[0]].define_optimization_objective(
                optimization_problem=optimization_problem,
                price_data=price_data,
                timesteps=timesteps
            )
            # Store objective value.
            objective_power_flows_iter.append(evaluate_optimization_objective_based_on_power_flow(
                optimization_problem=optimization_problem,
                electric_grid_model=electric_grid_model,
                power_flow_solutions=power_flow_solutions_per_timestep,
                timesteps=timesteps
            ))
        else:
            print('sigma <= tau -> Rejecting iteration. Repeating iteration using the modified region (delta).')

        # ---------------------------------------------------------------------------------------------------------
        print('Formulating optimization problem...', end='\r')
        # Instantiate / reset optimization problem.
        optimization_problem = fledge.utils.OptimizationProblem()

        # Define linear electric grid model variables.
        # The variables of the linearized electric grid model are independent of the linearization,
        #  --> no need to define them for every timestep
        linear_electric_grid_models_per_timestep[timesteps[0]].define_optimization_variables(
            optimization_problem=optimization_problem,
            timesteps=timesteps
        )
        # Define DER model variables
        der_model_set_reference.define_optimization_variables(
            optimization_problem
        )

        # Define linear electric grid model constraints.
        # TODO: adapt to what we actually want as limits
        node_voltage_magnitude_vector_minimum = 0.8 * np.abs(electric_grid_model.node_voltage_vector_reference)
        node_voltage_magnitude_vector_maximum = 1.2 * np.abs(electric_grid_model.node_voltage_vector_reference)
        branch_power_magnitude_vector_maximum = 1.0 * electric_grid_model.branch_power_vector_magnitude_reference
        # The linear electric grid model is different for every timestep
        for timestep in timesteps:
            linear_electric_grid_models_per_timestep[timestep].define_optimization_constraints(
                optimization_problem=optimization_problem,
                timesteps=timesteps,
                node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
                node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
                branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
            )

        # Define DER constraints.
        der_model_set_reference.define_optimization_constraints(
            optimization_problem,
            electric_grid_model
        )

        # TODO: must be removed in the end, this should only be part of a script
        # Add constraint on electricity use of flex building
        for der_name in der_model_set_reference.der_models.keys():
            der_model = der_model_set_reference.der_models[der_name]
            if type(der_model) is fledge.der_models.FlexibleBuildingModel:
                der_model.output_maximum_timeseries['grid_electric_power'] = (
                    (-1) * der_model.active_power_nominal_timeseries
                )
                # # Put a constraint on cooling power (= 0) to effectively disable cooling in the HVAC system
                # der_model.output_maximum_timeseries['zone_generic_cool_thermal_power_cooling'] = 0

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
        for node_index, node in enumerate(electric_grid_model.nodes):
            optimization_problem.constraints.append(
                optimization_problem.node_voltage_magnitude_vector[:, node_index]
                - np.abs(node_voltage_vector_reference.loc[:, node])
                >=
                -delta * np.abs(node_voltage_vector_reference.loc[:, node])
            )
            optimization_problem.constraints.append(
                optimization_problem.node_voltage_magnitude_vector[:, node_index]
                - np.abs(node_voltage_vector_reference.loc[:, node])
                <=
                delta * np.abs(node_voltage_vector_reference.loc[:, node])
            )

        # Branch flows.
        # TODO: needed or is voltage and DER active power constraint for trust region enough?
        for branch_index, branch in enumerate(electric_grid_model.branches):
            optimization_problem.constraints.append(
                optimization_problem.branch_power_magnitude_vector_1[:, branch_index]
                - np.abs(branch_power_magnitude_vector_1_reference.loc[:, branch])
                >=
                -delta * np.abs(branch_power_magnitude_vector_1_reference.loc[:, branch])
            )
            optimization_problem.constraints.append(
                optimization_problem.branch_power_magnitude_vector_1[:, branch_index]
                - np.abs(branch_power_magnitude_vector_1_reference.loc[:, branch])
                <=
                delta * np.abs(branch_power_magnitude_vector_1_reference.loc[:, branch])
            )
            optimization_problem.constraints.append(
                optimization_problem.branch_power_magnitude_vector_2[:, branch_index]
                - np.abs(branch_power_magnitude_vector_2_reference.loc[:, branch])
                >=
                -delta * np.abs(branch_power_magnitude_vector_2_reference.loc[:, branch])
            )
            optimization_problem.constraints.append(
                optimization_problem.branch_power_magnitude_vector_2[:, branch_index]
                - np.abs(branch_power_magnitude_vector_2_reference.loc[:, branch])
                <=
                delta * np.abs(branch_power_magnitude_vector_2_reference.loc[:, branch])
            )

        # DERs.
        for der_index, der in enumerate(electric_grid_model.ders):
            der_model = der_model_set_reference.der_models[der[1]]
            if not issubclass(type(der_model), fledge.der_models.FlexibleDERModel):
                # If not flexible, then there should not be a trust-region constraint on it
                continue
            # Check if load (negative nominal power value) or generator (positive...)
            if np.real(electric_grid_model.der_power_vector_reference[der_index]) < 0:
                factor = 1
            else:
                factor = -1

            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector[:, der_index]
                - np.real(der_power_vector_reference[der])
                <=
                -factor * delta * np.real(der_power_vector_reference[der])
            )
            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector[:, der_index]
                - np.real(der_power_vector_reference[der])
                >=
                factor * delta * np.real(der_power_vector_reference[der])
            )
            optimization_problem.constraints.append(
                optimization_problem.der_reactive_power_vector[:, der_index]
                - np.imag(der_power_vector_reference[der])
                <=
                -factor * delta * np.imag(der_power_vector_reference[der])
            )
            optimization_problem.constraints.append(
                optimization_problem.der_reactive_power_vector[:, der_index]
                - np.imag(der_power_vector_reference[der])
                >=
                factor * delta * np.imag(der_power_vector_reference[der])
            )

        # ---------------------------------------------------------------------------------------------------------
        # Define objective and solve.
        # NOTE: the objective is independent of the linearized power flow, hence we can use any of the models
        linear_electric_grid_models_per_timestep[timesteps[0]].define_optimization_objective(
            optimization_problem,
            price_data,
            timesteps
        )
        # Solve optimization problem.
        print('Solving optimal power flow...', end='\r')
        optimization_problem.solve()
        optimization_problems_iter.append(optimization_problem)

        # ---------------------------------------------------------------------------------------------------------
        # Trust-region evaluation and update
        print('Trust-region evaluation and update...', end='\r')
        # Obtain der power change value.
        der_active_power_vector_change = (
            np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=np.float)
        )
        der_reactive_power_vector_change = (
            np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=np.float)
        )

        for der_index, der in enumerate(electric_grid_model.ders):
            der_active_power_vector_change[:, der_index] = (
                optimization_problem.der_active_power_vector[:, der_index].value
                - np.real(der_power_vector_reference[der])
            )
            der_reactive_power_vector_change[:, der_index] = (
                optimization_problem.der_reactive_power_vector[:, der_index].value
                - np.imag(der_power_vector_reference[der])
            )
        # Change to p.u. value
        der_active_power_vector_change_per_unit = (
                der_active_power_vector_change / np.real(electric_grid_model.der_power_vector_reference))
        # For DERs with 0 reactive nominal power, we divide by zero here, that is why np settings are set to ignore
        der_reactive_power_vector_change_per_unit = (
                der_reactive_power_vector_change / np.imag(electric_grid_model.der_power_vector_reference))

        der_power_vector_change_per_unit_max = (
            max(
                np.max(abs(der_active_power_vector_change_per_unit)),
                np.max(abs(der_reactive_power_vector_change_per_unit))
            )
        )

        # Print change variables.
        # print(f"der_active_power_vector_change = {der_active_power_vector_change}")
        # print(f"der_reactive_power_vector_change = {der_reactive_power_vector_change}")
        print(f"der_power_vector_change_per_unit_max = {der_power_vector_change_per_unit_max}")

        # ---------------------------------------------------------------------------------------------------------
        # Check trust-region conditions and obtain DER power vector / power flow solution candidates for next iteration.
        # - Only if termination condition is not met, otherwise risk of division by zero.
        if der_power_vector_change_per_unit_max > epsilon:
            # Get new der vector and power flow solution candidate.
            results = fledge.problems.Results()
            results.update(der_model_set_reference.get_optimization_results(optimization_problem))
            # NOTE: results function independent of linear electric grid model, we can call it from any timestep
            results.update(
                linear_electric_grid_models_per_timestep[timesteps[0]].get_optimization_results(
                    optimization_problem,
                    None,
                    timesteps
                )
            )
            der_model_set_candidate = change_der_set_points_based_on_results(
                der_model_set=der_model_set_reference,
                results=results
            )
            power_flow_solutions_per_timestep_candidate = get_power_flow_solutions_per_timestep(
                electric_grid_model=electric_grid_model,
                der_model_set_new_setpoints=der_model_set_candidate,
                timesteps=timesteps
            )

            # Obtain objective values.
            objective_linear_model = (
                optimization_problem.objective.value
            )

            objective_power_flow = evaluate_optimization_objective_based_on_power_flow(
                optimization_problem=optimization_problem,
                electric_grid_model=electric_grid_model,
                power_flow_solutions=power_flow_solutions_per_timestep_candidate,
                timesteps=timesteps
            )

            # Check if power flow of candidate is violating line limits, to then increase the radius delta
            # This is only to save some iterations
            branch_power_magnitude_vector_1_candidate = get_branch_power_vector_per_timestep(
                electric_grid_model,
                power_flow_solutions_per_timestep_candidate,
                timesteps,
                1
            )
            branch_power_magnitude_vector_2_candidate = get_branch_power_vector_per_timestep(
                electric_grid_model,
                power_flow_solutions_per_timestep_candidate,
                timesteps,
                2
            )
            pf_violation_flag_1 = (
                    np.abs(branch_power_magnitude_vector_maximum
                           - branch_power_magnitude_vector_1_candidate) < 0).any(axis=None)
            pf_violation_flag_2 = (
                    np.abs(branch_power_magnitude_vector_maximum
                           - branch_power_magnitude_vector_2_candidate) < 0).any(axis=None)

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
    linear_electric_grid_model = linear_electric_grid_models_per_timestep[timesteps[0]]
    results = fledge.problems.Results()
    results.update(
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            None,
            timesteps,
        )
    )
    results.update(
        der_model_set_reference.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Store results to CSV.
    results.save(results_path)

    # Obtain DLMPs.
    dlmps = (
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_data,
            scenario_data.timesteps
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs to CSV.
    dlmps.save(results_path)

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")
    print(f'Time elapsed for trust region: {(end_time - start_time)}')
    print(f'Trust region iterations: {trust_region_iteration_count}')


def run_presolve_with_der_models(
        der_model_set: fledge.der_models.DERModelSet,
        price_data: fledge.data_interface.PriceData
) -> fledge.problems.Results:
    # Pre-solve optimal operation problem without underlying electric grid to get realistic initial values
    # Obtain all DERs
    print('Running pre-solve for der models only...', end='\r')

    # Instantiate decentralized DER optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem
    )
    # Add constraint on electricity use of flex building
    for der_name in der_model_set.der_models.keys():
        der_model = der_model_set.der_models[der_name]
        if type(der_model) is fledge.der_models.FlexibleBuildingModel:
            der_model.output_maximum_timeseries['grid_electric_power'] = (
                    (-1) * der_model.active_power_nominal_timeseries
            )
            # # Put a constraint on cooling power (= 0) to effectively disable cooling in the HVAC system
            # der_model.output_maximum_timeseries['zone_generic_cool_thermal_power_cooling'] = 0

    # Define objective (DER operation cost minimization).
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    # Solve decentralized DER optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = fledge.problems.Results()
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )
    return results


def change_der_set_points_based_on_results(
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
                    der_model.active_power_nominal_timeseries = (
                        results.output_vector[(der_name, 'grid_electric_power')]
                    ) * (-1)
                    if type(der_model) is fledge.der_models.FlexibleBuildingModel:
                        power_factor = der_model.power_factor_nominal
                    else:
                        power_factor = 0.95
                    der_model.reactive_power_nominal_timeseries = (
                        results.output_vector[(der_name, 'grid_electric_power')] * np.tan(np.arccos(power_factor))
                    ) * (-1)
    else:
        print('Results object does not contain any data on active power output. ')
        raise ValueError

    return der_model_set


def get_power_flow_solutions_per_timestep(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        der_model_set_new_setpoints: fledge.der_models.DERModelSet,
        timesteps: pd.Index
):
    der_power_vector = get_der_power_vector(electric_grid_model, der_model_set_new_setpoints, timesteps)
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


def get_node_voltage_vector_per_timestep(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        power_flow_solutions_per_timestep: dict,
        timesteps: pd.Index
) -> pd.DataFrame:
    node_voltage_vector_per_timestep = (
        pd.DataFrame(columns=electric_grid_model.nodes, index=timesteps, dtype=np.complex)
    )
    for timestep in power_flow_solutions_per_timestep.keys():
        node_voltage_vector_per_timestep.loc[timestep, :] = (
            power_flow_solutions_per_timestep[timestep].node_voltage_vector
        )

    return node_voltage_vector_per_timestep


def get_branch_power_vector_per_timestep(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        power_flow_solutions_per_timestep: dict,
        timesteps: pd.Index,
        direction: int
) -> pd.DataFrame:
    branch_power_vector_per_timestep = (
        pd.DataFrame(columns=electric_grid_model.branches, index=timesteps, dtype=np.complex)
    )
    for timestep in power_flow_solutions_per_timestep.keys():
        if direction == 1:
            branch_power_vector_per_timestep.loc[timestep, :] = (
                power_flow_solutions_per_timestep[timestep].branch_power_vector_1
            )
        elif direction == 2:
            branch_power_vector_per_timestep.loc[timestep, :] = (
                power_flow_solutions_per_timestep[timestep].branch_power_vector_2
            )
        else:
            print(f'No valid branch flow direction provided. Possible values: 1 or 2, provided value: {direction}')
            raise ValueError

    return branch_power_vector_per_timestep


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


if __name__ == '__main__':
    main()
