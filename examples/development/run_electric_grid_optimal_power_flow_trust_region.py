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

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'

    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Instantiate iteration variables.
    sigma = 0.0
    der_power_vector_change_max = np.inf
    trust_region_iteration_count = 0
    power_flow_solutions_iter = []
    # linear_electric_grid_models_iter = []
    objective_power_flows_iter = []
    optimization_problems_iter = []

    # Define trust-region parameters according to [2].
    delta = 0.1  # 3.0 / 0.5 / range: (0, delta_max] / If too big, no power flow solution.
    delta_max = 0.2  # 4.0 / 1.0
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

    # Obtain the base case power flow, using the active_power_nominal values given as input as initial dispatch
    # quantities. This represents the initial solution candidate.
    # For the base power flow, we use the nominal power flow results
    der_model_set_reference = fledge.der_models.DERModelSet(scenario_name)
    power_flow_solutions_per_timestep_candidate = get_power_flow_solutions_per_timestep(
        electric_grid_model=electric_grid_model,
        der_model_set_new_setpoints=der_model_set_reference,
        timesteps=timesteps
    )
    # Assign as initial power flow solution candidate (for each timestep)
    der_model_set_candidate = der_model_set_reference

    while (
            (der_power_vector_change_max > epsilon)
            and (trust_region_iteration_count < trust_region_iteration_limit)
    ):

        # Print progress.
        print(f"Starting trust-region iteration #{trust_region_iteration_count}")

        # Check trust-region solution acceptance conditions.
        if (trust_region_iteration_count == 0) or (sigma > tau):

            # Accept der power vector and power flow solution candidate.
            # DER power vector is stored in the der_model_set for every DER and every timestep
            der_model_set_reference = der_model_set_candidate  # TODO: is this correct?
            power_flow_solutions_per_timestep = power_flow_solutions_per_timestep_candidate
            power_flow_solutions_iter.append(power_flow_solutions_per_timestep)

            # Obtain new
            # TODO: adapt to LinearElectricGridModelLocal
            # Get linear electric grid model for all timesteps
            linear_electric_grid_models_per_timestep = get_linear_electric_grid_models_per_timestep(
                electric_grid_model,
                power_flow_solutions_per_timestep,
                timesteps)

            # Currently not needed
            # linear_electric_grid_models_iter.append(linear_electric_grid_model)

            # Instantiate optimization problem to evaluate the objective based on power flow solution setpoints
            optimization_problem = fledge.utils.OptimizationProblem()

            # Define optimization variables.
            # The variables of the linearized electric grid model are independent of the linearization,
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

        # Instantiate / reset optimization problem.
        optimization_problem = fledge.utils.OptimizationProblem()

        # Define linear electric grid model variables.
        linear_electric_grid_models_per_timestep[timesteps[0]].define_optimization_variables(
            optimization_problem=optimization_problem,
            timesteps=timesteps
        )
        # Define DER model variables
        # TODO: check if using the reference is correct here?
        der_model_set_reference.define_optimization_variables(
            optimization_problem
        )

        # Define linear electric grid model constraints.
        node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
        node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
        branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference
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
        # TODO: check if using reference is correct
        der_model_set_reference.define_optimization_constraints(
            optimization_problem,
            electric_grid_model
        )

        # TODO: DERs are currently assumed to be only loads, hence negative power values.
        # TODO: Why are power vectors constraint? by 0.5 / 1.5?
        # TODO: Could this be done with the regular der_model.define_der_constraints()?
        # time_index = 0
        # for timestep in timesteps:
        #     for der_index, der in enumerate(electric_grid_model.ders):
        #         optimization_problem.constraints.append(
        #             optimization_problem.der_active_power_vector[time_index, der_index]
        #             <=
        #             0.5 * np.real(electric_grid_model.der_power_vector_reference[der_index])
        #         )
        #         optimization_problem.constraints.append(
        #             optimization_problem.der_active_power_vector[time_index, der_index]
        #             >=
        #             1.5 * np.real(electric_grid_model.der_power_vector_reference[der_index])
        #         )
        #         # Fixed power factor for reactive power based on nominal power factor.
        #         optimization_problem.constraints.append(
        #             optimization_problem.der_reactive_power_vector[time_index, der_index]
        #             ==
        #             optimization_problem.der_active_power_vector[time_index, der_index]
        #             * np.imag(electric_grid_model.der_power_vector_reference[der_index])
        #             / np.real(electric_grid_model.der_power_vector_reference[der_index])
        #         )
        #     time_index += 1

        # Define trust region constraints.
        # The trust-region permissible value for variables to move is determined by radius delta, which is included
        # in all inequality constraints [1]
        # We redefine the approximate state and dispatch quantities as the measure of change in their
        # operating state at the current iteration
        # TODO: Check if this is already correct? Shouldn't it be the power_flow_solution instead of the
        #  reference electric grid model?
        # DERs.
        # TODO: DERs are currently assumed to be only loads, hence negative power values.
        for der_index, der in enumerate(electric_grid_model.ders):
            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector[:, der_index]
                - np.real(electric_grid_model.der_power_vector_reference[der_index])
                <=
                -delta * np.real(electric_grid_model.der_power_vector_reference[der_index])
            )
            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector[:, der_index]
                - np.real(electric_grid_model.der_power_vector_reference[der_index])
                >=
                delta * np.real(electric_grid_model.der_power_vector_reference[der_index])
            )

        # Voltage.
        for node_index, node in enumerate(electric_grid_model.nodes):
            optimization_problem.constraints.append(
                optimization_problem.node_voltage_magnitude_vector[:, node_index]
                - np.abs(electric_grid_model.node_voltage_vector_reference[node_index])
                >=
                -delta * np.abs(electric_grid_model.node_voltage_vector_reference[node_index])
            )
            optimization_problem.constraints.append(
                optimization_problem.node_voltage_magnitude_vector[:, node_index]
                - np.abs(electric_grid_model.node_voltage_vector_reference[node_index])
                <=
                delta * np.abs(electric_grid_model.node_voltage_vector_reference[node_index])
            )

        # Branch flows.
        # TODO: branch power vector is not defined as "change" anymore! --> build helper!
        # TODO: check if it is correct to always refer to power_flow_sultion of first iteration!?
        #  --> Hanif assigns newPGBase
        # TODO: Do we need the "squared" value or does it also work with the "regular" change?
        # In Hanif's implementation this is actually not represented

        # for branch_index, branch in enumerate(electric_grid_model.branches):
        #     optimization_problem.constraints.append(
        #         optimization_problem.branch_power_magnitude_vector_1[0, branch_index]
        #         - np.abs(electric_grid_model.branch_power_vector_magnitude_reference[branch_index])
        #         >=
        #         -delta * np.abs(power_flow_solutions[0].branch_power_vector_1[branch_index])
        #     )
        #     optimization_problem.constraints.append(
        #         optimization_problem.branch_power_magnitude_vector_1[0, branch_index]
        #         - np.abs(electric_grid_model.branch_power_vector_magnitude_reference[branch_index])
        #         <=
        #         delta * np.abs(power_flow_solutions[0].branch_power_vector_1[branch_index])
        #     )
        #     optimization_problem.constraints.append(
        #         optimization_problem.branch_power_magnitude_vector_2[0, branch_index]
        #         - np.abs(electric_grid_model.branch_power_vector_magnitude_reference[branch_index])
        #         >=
        #         -delta * np.abs(power_flow_solutions[0].branch_power_vector_2[branch_index])
        #     )
        #     optimization_problem.constraints.append(
        #         optimization_problem.branch_power_magnitude_vector_2[0, branch_index]
        #         - np.abs(electric_grid_model.branch_power_vector_magnitude_reference[branch_index])
        #         <=
        #         delta * np.abs(power_flow_solutions[0].branch_power_vector_2[branch_index])
        #     )

        # Loss.
        # TODO: find out why sum of losses from pf solution?
        # In Hanif's implementation this is actually not represented
        # optimization_problem.constraints.append(
        #     optimization_problem.loss_active[0]
        #     - np.sum(np.real(power_flow_solutions[0].loss))
        #     >=
        #     -delta * np.sum(np.real(power_flow_solutions[0].loss))
        # )
        # optimization_problem.constraints.append(
        #     optimization_problem.loss_active[0]
        #     - np.sum(np.real(power_flow_solutions[0].loss))
        #     <=
        #     delta * np.sum(np.real(power_flow_solutions[0].loss))
        # )
        # optimization_problem.constraints.append(
        #     optimization_problem.loss_reactive[0]
        #     - np.sum(np.imag(power_flow_solutions[0].loss))
        #     >=
        #     -delta * np.sum(np.imag(power_flow_solutions[0].loss))
        # )
        # optimization_problem.constraints.append(
        #     optimization_problem.loss_reactive[0]
        #     - np.sum(np.imag(power_flow_solutions[0].loss))
        #     <=
        #     delta * np.sum(np.imag(power_flow_solutions[0].loss))
        # )

        # Define objective.
        linear_electric_grid_models_per_timestep[timesteps[0]].define_optimization_objective(
            optimization_problem,
            price_data,
            timesteps
        )
        # Solve optimization problem.
        optimization_problem.solve()
        optimization_problems_iter.append(optimization_problem)

        # Obtain der power change value.
        der_active_power_vector_change = (
            np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=np.float)
        )
        der_reactive_power_vector_change = (
            np.zeros([len(timesteps), len(electric_grid_model.ders)], dtype=np.float)
        )
        # TODO: Check if correct / makes sense over all timesteps?
        for der_index, der in enumerate(electric_grid_model.ders):
            der_active_power_vector_change[:, der_index] = (
                optimization_problem.der_active_power_vector[:, der_index].value
                - np.real(electric_grid_model.der_power_vector_reference[der_index])
            )
            der_reactive_power_vector_change[:, der_index] = (
                optimization_problem.der_reactive_power_vector[:, der_index].value
                - np.imag(electric_grid_model.der_power_vector_reference[der_index])
            )
        der_power_vector_change_max = (
            max(
                np.max(abs(der_active_power_vector_change)),
                np.max(abs(der_reactive_power_vector_change))
            )
        )

        # Print change variables.
        # print(f"der_active_power_vector_change = {der_active_power_vector_change}")
        # print(f"der_reactive_power_vector_change = {der_reactive_power_vector_change}")
        print(f"der_power_vector_change_max = {der_power_vector_change_max}")

        # Check trust-region conditions and obtain DER power vector / power flow solution candidates for next iteration.
        # - Only if termination condition is not met, otherwise risk of division by zero.
        if der_power_vector_change_max > epsilon:
            # TODO: adapt to every timestep
            # Get new der vector and power flow solution candidate.
            results = fledge.problems.Results()
            results.update(der_model_set_reference.get_optimization_results(optimization_problem))
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
            # TODO: what to use here?
            objective_linear_model = (
                optimization_problem.objective.value
            )

            objective_power_flow = evaluate_optimization_objective_based_on_power_flow(
                optimization_problem=optimization_problem,
                electric_grid_model=electric_grid_model,
                power_flow_solutions=power_flow_solutions_per_timestep_candidate,
                timesteps=timesteps
            )

            # Check trust-region range conditions.
            sigma = (
                (objective_power_flows_iter[-1] - objective_power_flow)
                / (objective_power_flows_iter[-1] - objective_linear_model)
            )
            if sigma <= eta:
                delta *= gamma
            elif sigma > (1.0 - eta):
                delta = min(2 * delta, delta_max)

            # Print trust-region parameters.
            print(f"objective_power_flow = {objective_power_flow}")
            print(f"objective_linear_model = {objective_linear_model}")
            print(f"objective_power_flows[-1] = {objective_power_flows_iter[-1]}")
            print(f"sigma = {sigma}")
            print(f"delta = {delta}")

        # Iterate counter.
        trust_region_iteration_count += 1

    # Obtain results.
    # TODO: which lin electric grid model and power flow solution is the correct one to pass to the results?
    results = fledge.problems.Results()
    results.update(
        linear_electric_grid_models_per_timestep[timesteps[0]].get_optimization_results(
            optimization_problem,
            power_flow_solutions_iter[0][timesteps[0]],
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

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


def change_der_set_points_based_on_results(
        der_model_set: fledge.der_models.DERModelSet,
        results: fledge.problems.Results
) -> fledge.der_models.DERModelSet:
    for der_name in der_model_set.der_names:
        der_model = der_model_set.der_models[der_name]
        der_type = der_model.der_type
        der_model.active_power_nominal_timeseries = \
            results.der_active_power_vector[der_type, der_name]
        der_model.reactive_power_nominal_timeseries = \
            results.der_reactive_power_vector[der_type, der_name]

    return der_model_set


def get_power_flow_solutions_per_timestep(
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


def get_linear_electric_grid_models_per_timestep(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault,
        power_flow_solutions: dict,
        timesteps
):
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
