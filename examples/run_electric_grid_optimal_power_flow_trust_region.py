"""Example script for setting up and solving an optimal power flow problem with trust-region algorithm."""

import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.electric_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_6node'

    results_path = (
        os.path.join(
            fledge.config.results_path,
            f'run_electric_grid_optimal_power_flow_trust_region_{fledge.config.timestamp}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()

    # Obtain models.
    electric_grid_model = (
        fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    )
    der_power_vector_reference_candidate = electric_grid_model.der_power_vector_nominal
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
                - np.sum(np.real(der_power_vector_reference_candidate).ravel())
                + np.sum(np.real(power_flow_solution_candidate.loss.ravel()))
            )
            objective_power_flows.append(objective_power_flow)

        # Instantiate / reset optimization problem.
        optimization_problem = pyo.ConcreteModel()

        # Define linear electric grid model variables.
        linear_electric_grid_model.define_optimization_variables(optimization_problem)

        # Define linear electric grid model constraints.
        linear_electric_grid_model.define_optimization_constraints(optimization_problem)

        # Define DER constraints.
        # TODO: DERs are currently assumed to be only loads, hence negative power values.
        optimization_problem.der_constraints = pyo.ConstraintList()
        for der_index, der in enumerate(electric_grid_model.ders):
            optimization_problem.der_constraints.add(
                optimization_problem.der_active_power_vector_change[0, der]
                <=
                0.5 * np.real(electric_grid_model.der_power_vector_nominal[der_index])
                - np.real(der_power_vector_reference[der_index])
            )
            optimization_problem.der_constraints.add(
                optimization_problem.der_active_power_vector_change[0, der]
                >=
                1.5 * np.real(electric_grid_model.der_power_vector_nominal[der_index])
                - np.real(der_power_vector_reference[der_index])
            )
            # Fixed power factor for reactive power based on nominal power factor.
            optimization_problem.der_constraints.add(
                optimization_problem.der_reactive_power_vector_change[0, der]
                ==
                optimization_problem.der_active_power_vector_change[0, der]
                * np.imag(electric_grid_model.der_power_vector_nominal[der_index])
                / np.real(electric_grid_model.der_power_vector_nominal[der_index])
            )

        # Define trust region constraints.
        optimization_problem.trust_region_constraints = pyo.ConstraintList()

        # DER.
        # TODO: DERs are currently assumed to be only loads, hence negative power values.
        for der_index, der in enumerate(electric_grid_model.ders):
            optimization_problem.trust_region_constraints.add(
                optimization_problem.der_active_power_vector_change[0, der]
                <=
                -delta * np.real(electric_grid_model.der_power_vector_nominal.ravel()[der_index])
            )
            optimization_problem.trust_region_constraints.add(
                optimization_problem.der_active_power_vector_change[0, der]
                >=
                delta * np.real(electric_grid_model.der_power_vector_nominal.ravel()[der_index])
            )

        # Voltage.
        for node_index, node in enumerate(electric_grid_model.nodes):
            optimization_problem.trust_region_constraints.add(
                optimization_problem.voltage_magnitude_vector_change[0, node]
                >=
                -delta * np.abs(electric_grid_model.node_voltage_vector_no_load.ravel()[node_index])
            )
            optimization_problem.trust_region_constraints.add(
                optimization_problem.voltage_magnitude_vector_change[0, node]
                <=
                delta * np.abs(electric_grid_model.node_voltage_vector_no_load.ravel()[node_index])
            )

        # Branch flows.
        for branch_index, branch in enumerate(electric_grid_model.branches):
            optimization_problem.trust_region_constraints.add(
                optimization_problem.branch_power_vector_1_squared_change[0, branch]
                >=
                -delta * np.abs(power_flow_solutions[0].branch_power_vector_1.ravel()[branch_index] ** 2)
            )
            optimization_problem.trust_region_constraints.add(
                optimization_problem.branch_power_vector_1_squared_change[0, branch]
                <=
                delta * np.abs(power_flow_solutions[0].branch_power_vector_1.ravel()[branch_index] ** 2)
            )
            optimization_problem.trust_region_constraints.add(
                optimization_problem.branch_power_vector_2_squared_change[0, branch]
                >=
                -delta * np.abs(power_flow_solutions[0].branch_power_vector_2.ravel()[branch_index] ** 2)
            )
            optimization_problem.trust_region_constraints.add(
                optimization_problem.branch_power_vector_2_squared_change[0, branch]
                <=
                delta * np.abs(power_flow_solutions[0].branch_power_vector_2.ravel()[branch_index] ** 2)
            )

        # Loss.
        optimization_problem.trust_region_constraints.add(
            optimization_problem.loss_active_change[0]
            >=
            -delta * np.sum(np.real(power_flow_solutions[0].loss))
        )
        optimization_problem.trust_region_constraints.add(
            optimization_problem.loss_active_change[0]
            <=
            delta * np.sum(np.real(power_flow_solutions[0].loss))
        )
        optimization_problem.trust_region_constraints.add(
            optimization_problem.loss_reactive_change[0]
            >=
            -delta * np.sum(np.imag(power_flow_solutions[0].loss))
        )
        optimization_problem.trust_region_constraints.add(
            optimization_problem.loss_reactive_change[0]
            <=
            delta * np.sum(np.imag(power_flow_solutions[0].loss))
        )

        # Define objective.
        optimization_problem.objective = (
            pyo.Objective(
                expr=0.0,
                sense=pyo.minimize
            )
        )
        optimization_problem.objective.expr += (
            # DER active power.
            # TODO: DERs are currently assumed to be only loads, hence negative values.
            -1.0 * sum(
                optimization_problem.der_active_power_vector_change[0, der]
                + np.real(der_power_vector_reference[der_index])
                for der_index, der in enumerate(electric_grid_model.ders)
            )
        )
        optimization_problem.objective.expr += (
            # Active loss.
            optimization_problem.loss_active_change[0]
            + np.sum(np.real(power_flow_solution.loss))
        )

        # Solve optimization problem.
        optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
        optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
        optimization_problems.append(optimization_problem)
        try:
            assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
        except AssertionError:
            raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
        # optimization_problem.display()

        # Obtain der power change value.
        der_active_power_vector_change = (
            np.zeros((len(electric_grid_model.ders), 1), dtype=np.float)
        )
        der_reactive_power_vector_change = (
            np.zeros((len(electric_grid_model.ders), 1), dtype=np.float)
        )
        for der_index, der in enumerate(electric_grid_model.ders):
            der_active_power_vector_change[der_index] = (
                optimization_problem.der_active_power_vector_change[0, der].value
            )
            der_reactive_power_vector_change[der_index] = (
                optimization_problem.der_reactive_power_vector_change[0, der].value
            )
        der_power_vector_change_max = (
            max(
                np.max(abs(der_active_power_vector_change).ravel()),
                np.max(abs(der_reactive_power_vector_change).ravel())
            )
        )

        # Print change variables.
        print(f"der_active_power_vector_change = {der_active_power_vector_change.ravel()}")
        print(f"der_reactive_power_vector_change = {der_reactive_power_vector_change.ravel()}")
        print(f"der_power_vector_change_max = {der_power_vector_change_max}")

        # Check trust-region conditions and obtain DER power vector / power flow solution candidates for next iteration.
        # - Only if termination condition is not met, otherwise risk of division by zero.
        if der_power_vector_change_max > epsilon:
            # Get new der vector and power flow solution candidate.
            der_power_vector_reference_candidate = (
                der_power_vector_reference
                + der_active_power_vector_change.ravel()
                + 1.0j * der_reactive_power_vector_change.ravel()
            )
            power_flow_solution_candidate = (
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint(
                    electric_grid_model,
                    der_power_vector_reference_candidate
                )
            )

            # Obtain objective values.
            objective_power_flow = (
                - np.sum(np.real(der_power_vector_reference_candidate).ravel())
                + np.sum(np.real(power_flow_solution_candidate.loss.ravel()))
            )
            objective_linear_model = (
                pyo.value(optimization_problem.objective)
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
    (
        der_active_power_vector,
        der_reactive_power_vector,
        voltage_magnitude_vector,
        branch_power_vector_1_squared,
        branch_power_vector_2_squared,
        loss_active,
        loss_reactive
    ) = linear_electric_grid_model.get_optimization_results(
        optimization_problem,
        power_flow_solutions[0],
        in_per_unit=True,
        with_mean=True
    )

    # Print results.
    print(f"der_active_power_vector = \n{der_active_power_vector}")
    print(f"der_reactive_power_vector = \n{der_reactive_power_vector}")
    print(f"voltage_magnitude_vector = \n{voltage_magnitude_vector}")
    print(f"branch_power_vector_1_squared = \n{branch_power_vector_1_squared}")
    print(f"branch_power_vector_2_squared = \n{branch_power_vector_2_squared}")
    print(f"loss_active = \n{loss_active}")
    print(f"loss_reactive = \n{loss_reactive}")

    # Store results as CSV.
    der_active_power_vector.to_csv(os.path.join(results_path, 'der_active_power_vector.csv'))
    der_reactive_power_vector.to_csv(os.path.join(results_path, 'der_reactive_power_vector.csv'))
    voltage_magnitude_vector.to_csv(os.path.join(results_path, 'voltage_magnitude_vector.csv'))
    branch_power_vector_1_squared.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared.csv'))
    branch_power_vector_2_squared.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared.csv'))
    loss_active.to_csv(os.path.join(results_path, 'loss_active.csv'))
    loss_reactive.to_csv(os.path.join(results_path, 'loss_reactive.csv'))

    # Print results path.
    print("Results are stored in: " + results_path)


if __name__ == '__main__':
    main()
