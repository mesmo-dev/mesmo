"""Example script for setting up and solving an optimal power flow problem with trust-region algorithm."""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'

    results_path = fledge.utils.get_results_path(os.path.basename(__file__)[:-3], scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain models.
    electric_grid_model = (
        fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    )
    der_power_vector_reference_candidate = electric_grid_model.der_power_vector_reference
    power_flow_solution_candidate = (
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(
            electric_grid_model,
            der_power_vector_reference_candidate
        )
    )

    # Instantiate iteration variables.
    sigma = 0.0
    der_power_vector_change_max = np.inf
    trust_region_iteration_count = 0
    power_flow_solutions = []
    linear_electric_grid_models = []
    objective_power_flows = []
    optimization_problems = []

    # Define trust-region parameters.
    delta = 0.1  # 3.0 / 0.5 / range: (0, delta_max] / If too big, no power flow solution.
    delta_max = 0.2  # 4.0 / 1.0
    gamma = 0.5  # 0.5 / range: (0, 1)
    eta = 0.1  # 0.1 / range: (0, 0.5]
    tau = 0.1  # 0.1 / range: [0, 0.25)
    epsilon = 1.0e-3  # 1e-3 / 1e-4
    trust_region_iteration_limit = 100

    while (
            (der_power_vector_change_max > epsilon)
            and (trust_region_iteration_count < trust_region_iteration_limit)
    ):

        # Print progress.
        print(f"Starting trust-region iteration #{trust_region_iteration_count}")

        # Check trust-region solution acceptance conditions.
        if (trust_region_iteration_count == 0) or (sigma > tau):

            # Accept der power vector and power flow solution candidate.
            der_power_vector_reference = der_power_vector_reference_candidate
            power_flow_solution = power_flow_solution_candidate
            power_flow_solutions.append(power_flow_solution)

            # Obtain new
            linear_electric_grid_model = (
                fledge.electric_grid_models.LinearElectricGridModelGlobal(
                    electric_grid_model,
                    power_flow_solution
                )
            )
            linear_electric_grid_models.append(linear_electric_grid_model)

            # Store objective value.
            objective_power_flow = (
                - np.sum(np.real(der_power_vector_reference_candidate))
                + np.sum(np.real(power_flow_solution_candidate.loss))
            )
            objective_power_flows.append(objective_power_flow)

        # Instantiate / reset optimization problem.
        optimization_problem = fledge.utils.OptimizationProblem()

        # Define linear electric grid model variables.
        linear_electric_grid_model.define_optimization_variables(optimization_problem)

        # Define linear electric grid model constraints.
        linear_electric_grid_model.define_optimization_constraints(optimization_problem)

        # Define DER constraints.
        # TODO: DERs are currently assumed to be only loads, hence negative power values.
        for der_index, der in enumerate(electric_grid_model.ders):
            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector_change[0, der_index]
                <=
                0.5 * np.real(electric_grid_model.der_power_vector_reference[der_index])
                - np.real(der_power_vector_reference[der_index])
            )
            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector_change[0, der_index]
                >=
                1.5 * np.real(electric_grid_model.der_power_vector_reference[der_index])
                - np.real(der_power_vector_reference[der_index])
            )
            # Fixed power factor for reactive power based on nominal power factor.
            optimization_problem.constraints.append(
                optimization_problem.der_reactive_power_vector_change[0, der_index]
                ==
                optimization_problem.der_active_power_vector_change[0, der_index]
                * np.imag(electric_grid_model.der_power_vector_reference[der_index])
                / np.real(electric_grid_model.der_power_vector_reference[der_index])
            )

        # Define trust region constraints.

        # DER.
        # TODO: DERs are currently assumed to be only loads, hence negative power values.
        for der_index, der in enumerate(electric_grid_model.ders):
            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector_change[0, der_index]
                <=
                -delta * np.real(electric_grid_model.der_power_vector_reference[der_index])
            )
            optimization_problem.constraints.append(
                optimization_problem.der_active_power_vector_change[0, der_index]
                >=
                delta * np.real(electric_grid_model.der_power_vector_reference[der_index])
            )

        # Voltage.
        for node_index, node in enumerate(electric_grid_model.nodes):
            optimization_problem.constraints.append(
                optimization_problem.voltage_magnitude_vector_change[0, node_index]
                >=
                -delta * np.abs(electric_grid_model.node_voltage_vector_reference[node_index])
            )
            optimization_problem.constraints.append(
                optimization_problem.voltage_magnitude_vector_change[0, node_index]
                <=
                delta * np.abs(electric_grid_model.node_voltage_vector_reference[node_index])
            )

        # Branch flows.
        for branch_index, branch in enumerate(electric_grid_model.branches):
            optimization_problem.constraints.append(
                optimization_problem.branch_power_vector_1_squared_change[0, branch_index]
                >=
                -delta * np.abs(power_flow_solutions[0].branch_power_vector_1[branch_index] ** 2)
            )
            optimization_problem.constraints.append(
                optimization_problem.branch_power_vector_1_squared_change[0, branch_index]
                <=
                delta * np.abs(power_flow_solutions[0].branch_power_vector_1[branch_index] ** 2)
            )
            optimization_problem.constraints.append(
                optimization_problem.branch_power_vector_2_squared_change[0, branch_index]
                >=
                -delta * np.abs(power_flow_solutions[0].branch_power_vector_2[branch_index] ** 2)
            )
            optimization_problem.constraints.append(
                optimization_problem.branch_power_vector_2_squared_change[0, branch_index]
                <=
                delta * np.abs(power_flow_solutions[0].branch_power_vector_2[branch_index] ** 2)
            )

        # Loss.
        optimization_problem.constraints.append(
            optimization_problem.loss_active_change[0]
            >=
            -delta * np.sum(np.real(power_flow_solutions[0].loss))
        )
        optimization_problem.constraints.append(
            optimization_problem.loss_active_change[0]
            <=
            delta * np.sum(np.real(power_flow_solutions[0].loss))
        )
        optimization_problem.constraints.append(
            optimization_problem.loss_reactive_change[0]
            >=
            -delta * np.sum(np.imag(power_flow_solutions[0].loss))
        )
        optimization_problem.constraints.append(
            optimization_problem.loss_reactive_change[0]
            <=
            delta * np.sum(np.imag(power_flow_solutions[0].loss))
        )

        # Define objective.
        optimization_problem.objective += (
            # DER active power.
            # TODO: DERs are currently assumed to be only loads, hence negative values.
            -1.0 * sum(sum(optimization_problem.der_active_power_vector_change))
        )
        optimization_problem.objective += (
            # Active loss.
            sum(sum(optimization_problem.loss_active_change))
            + sum(np.real(power_flow_solution.loss))
        )

        # Solve optimization problem.
        optimization_problem.solve()
        optimization_problems.append(optimization_problem)

        # Obtain der power change value.
        der_active_power_vector_change = (
            np.zeros(len(electric_grid_model.ders), dtype=np.float)
        )
        der_reactive_power_vector_change = (
            np.zeros(len(electric_grid_model.ders), dtype=np.float)
        )
        for der_index, der in enumerate(electric_grid_model.ders):
            der_active_power_vector_change[der_index] = (
                optimization_problem.der_active_power_vector_change[0, der_index].value
            )
            der_reactive_power_vector_change[der_index] = (
                optimization_problem.der_reactive_power_vector_change[0, der_index].value
            )
        der_power_vector_change_max = (
            max(
                np.max(abs(der_active_power_vector_change)),
                np.max(abs(der_reactive_power_vector_change))
            )
        )

        # Print change variables.
        print(f"der_active_power_vector_change = {der_active_power_vector_change}")
        print(f"der_reactive_power_vector_change = {der_reactive_power_vector_change}")
        print(f"der_power_vector_change_max = {der_power_vector_change_max}")

        # Check trust-region conditions and obtain DER power vector / power flow solution candidates for next iteration.
        # - Only if termination condition is not met, otherwise risk of division by zero.
        if der_power_vector_change_max > epsilon:
            # Get new der vector and power flow solution candidate.
            der_power_vector_reference_candidate = (
                der_power_vector_reference
                + der_active_power_vector_change
                + 1.0j * der_reactive_power_vector_change
            )
            power_flow_solution_candidate = (
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint(
                    electric_grid_model,
                    der_power_vector_reference_candidate
                )
            )

            # Obtain objective values.
            objective_power_flow = (
                - np.sum(np.real(der_power_vector_reference_candidate))
                + np.sum(np.real(power_flow_solution_candidate.loss))
            )
            objective_linear_model = (
                optimization_problem.objective.value
            )

            # Check trust-region range conditions.
            sigma = (
                (objective_power_flows[-1] - objective_power_flow)
                / (objective_power_flows[-1] - objective_linear_model)
            )
            if sigma <= eta:
                delta *= gamma
            elif sigma > (1.0 - eta):
                delta = min(2 * delta, delta_max)

            # Print trust-region parameters.
            print(f"objective_power_flow = {objective_power_flow}")
            print(f"objective_linear_model = {objective_linear_model}")
            print(f"objective_power_flows[-1] = {objective_power_flows[-1]}")
            print(f"sigma = {sigma}")
            print(f"delta = {delta}")

        # Iterate counter.
        trust_region_iteration_count += 1

    # Obtain results.
    results = (
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            power_flow_solutions[0],
            in_per_unit=True,
            with_mean=True
        )
    )

    # Print results.
    print(results)

    # Store results to CSV.
    results.to_csv(results_path)

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
