"""Run script for reproducing results of the Paper: 'Coordinated Market Clearing for Combined Thermal and Electric
Distribution Grid Operation'.
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy.sparse as sp

import mesmo


def main(scenario_number=None, admm_rho=None):

    # TODO: Review convergence parameters / result value units.

    # Settings.
    admm_iteration_limit = 1000
    admm_rho = 1e-1 if admm_rho is None else admm_rho
    admm_primal_residual_termination_limit = 1e-1
    admm_dual_residual_termination_limit = 1e-1
    scenario_number = 2 if scenario_number is None else scenario_number
    # Choices (Note that constrained cases may not solve reliably):
    # 1 - unconstrained operation,
    # 2 - constrained thermal grid branch flow,
    # 3 - constrained thermal grid pressure head,
    # 4 - constrained electric grid branch power,
    # 5 - constrained electric grid voltage
    scenario_name = "paper_2021_troitzsch_admm"

    # Obtain results path.
    results_path = mesmo.utils.get_results_path(
        __file__, f"scenario{scenario_number}_rho{admm_rho:.0e}_{scenario_name}"
    )

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name)

    # Obtain models.
    mesmo.utils.log_time(f"model setup")
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModel(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint("paper_2021_troitzsch_admm_dlmp")
    linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(
        electric_grid_model, power_flow_solution
    )
    thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_grid_model.plant_efficiency = 10.0  # Change model parameter to incentivize use of thermal grid.
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    thermal_power_flow_solution = mesmo.thermal_grid_models.ThermalPowerFlowSolution("paper_2021_troitzsch_admm_dlmp")
    linear_thermal_grid_model_set = mesmo.thermal_grid_models.LinearThermalGridModelSet(
        thermal_grid_model, thermal_power_flow_solution
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)
    mesmo.utils.log_time(f"model setup")

    # Define thermal grid limits.
    node_head_vector_minimum = 100.0 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 100.0 * np.abs(thermal_power_flow_solution.branch_flow_vector)
    # Modify limits for scenarios.
    if scenario_number in [2]:
        branch_flow_vector_maximum[mesmo.utils.get_index(thermal_grid_model.branches, branch_name="4")] *= 0.2 / 100.0
    elif scenario_number in [3]:
        node_head_vector_minimum[mesmo.utils.get_index(thermal_grid_model.nodes, node_name="15")] *= 0.2 / 100.0
    else:
        pass

    # Define electric grid limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 100.0 * electric_grid_model.branch_power_vector_magnitude_reference
    # Modify limits for scenarios.
    if scenario_number in [4]:
        branch_power_magnitude_vector_maximum[mesmo.utils.get_index(electric_grid_model.branches, branch_name="4")] *= (
            8.5 / 100.0
        )
    elif scenario_number in [5]:
        node_voltage_magnitude_vector_minimum[mesmo.utils.get_index(electric_grid_model.nodes, node_name="15")] *= (
            0.9985 / 0.5
        )
    else:
        pass

    # Instantiate ADMM variables.
    admm_iteration = 0
    admm_continue = True
    admm_primal_residuals = pd.DataFrame(columns=["Thermal pw.", "Active pw.", "Reactive pw."])
    admm_dual_residuals = pd.DataFrame(columns=["Thermal pw.", "Active pw.", "Reactive pw."])

    # Instantiate optimization problems.
    # TODO: Consider timestep_interval_hours in ADMM objective.
    optimization_problem_baseline = mesmo.solutions.OptimizationProblem()
    optimization_problem_electric = mesmo.solutions.OptimizationProblem()
    optimization_problem_thermal = mesmo.solutions.OptimizationProblem()
    optimization_problem_aggregator = mesmo.solutions.OptimizationProblem()

    # Instantiate ADMM optimization parameters.
    optimization_problem_electric.define_parameter(
        "admm_exchange_der_active_power", np.zeros((len(scenario_data.timesteps) * len(electric_grid_model.ders), 1))
    )
    optimization_problem_electric.define_parameter(
        "admm_exchange_der_reactive_power", np.zeros((len(scenario_data.timesteps) * len(electric_grid_model.ders), 1))
    )
    optimization_problem_thermal.define_parameter(
        "admm_exchange_der_thermal_power", np.zeros((len(scenario_data.timesteps) * len(thermal_grid_model.ders), 1))
    )
    optimization_problem_aggregator.define_parameter(
        "admm_exchange_der_active_power", np.zeros((len(scenario_data.timesteps) * len(electric_grid_model.ders), 1))
    )
    optimization_problem_aggregator.define_parameter(
        "admm_exchange_der_reactive_power", np.zeros((len(scenario_data.timesteps) * len(electric_grid_model.ders), 1))
    )
    optimization_problem_aggregator.define_parameter(
        "admm_exchange_der_thermal_power", np.zeros((len(scenario_data.timesteps) * len(thermal_grid_model.ders), 1))
    )
    optimization_problem_electric.define_parameter(
        "admm_lambda_electric_der_active_power",
        np.zeros((1, len(scenario_data.timesteps) * len(electric_grid_model.ders))),
    )
    optimization_problem_electric.define_parameter(
        "admm_lambda_electric_der_reactive_power",
        np.zeros((1, len(scenario_data.timesteps) * len(electric_grid_model.ders))),
    )
    optimization_problem_thermal.define_parameter(
        "admm_lambda_thermal_der_thermal_power",
        np.zeros((1, len(scenario_data.timesteps) * len(thermal_grid_model.ders))),
    )
    optimization_problem_aggregator.define_parameter(
        "admm_lambda_aggregator_der_active_power",
        np.zeros((1, len(scenario_data.timesteps) * len(electric_grid_model.ders))),
    )
    optimization_problem_aggregator.define_parameter(
        "admm_lambda_aggregator_der_reactive_power",
        np.zeros((1, len(scenario_data.timesteps) * len(electric_grid_model.ders))),
    )
    optimization_problem_aggregator.define_parameter(
        "admm_lambda_aggregator_der_thermal_power",
        np.zeros((1, len(scenario_data.timesteps) * len(thermal_grid_model.ders))),
    )

    # ADMM: Centralized / baseline problem.

    # Log progress.
    mesmo.utils.log_time(f"baseline problem setup")

    # Define linear electric grid problem.
    linear_electric_grid_model_set.define_optimization_variables(optimization_problem_baseline)
    linear_electric_grid_model_set.define_optimization_parameters(
        optimization_problem_baseline,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
    )
    linear_electric_grid_model_set.define_optimization_constraints(optimization_problem_baseline)
    linear_electric_grid_model_set.define_optimization_objective(optimization_problem_baseline)

    # Define thermal grid problem.
    linear_thermal_grid_model_set.define_optimization_variables(optimization_problem_baseline)
    linear_thermal_grid_model_set.define_optimization_parameters(
        optimization_problem_baseline,
        price_data,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum,
    )
    linear_thermal_grid_model_set.define_optimization_constraints(optimization_problem_baseline)
    linear_thermal_grid_model_set.define_optimization_objective(optimization_problem_baseline)

    # Define DER problem.
    der_model_set.define_optimization_variables(optimization_problem_baseline)
    der_model_set.define_optimization_parameters(optimization_problem_baseline, price_data)
    der_model_set.define_optimization_constraints(optimization_problem_baseline)
    der_model_set.define_optimization_objective(optimization_problem_baseline)

    # Log progress.
    mesmo.utils.log_time(f"baseline problem setup")

    # Solve baseline problem.
    mesmo.utils.log_time(f"baseline problem solution")
    optimization_problem_baseline.solve()
    mesmo.utils.log_time(f"baseline problem solution")

    # Get baseline results.
    results_baseline = mesmo.problems.Results()
    results_baseline.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem_baseline))
    results_baseline.update(
        linear_electric_grid_model_set.get_optimization_dlmps(optimization_problem_baseline, price_data)
    )
    results_baseline.update(linear_thermal_grid_model_set.get_optimization_results(optimization_problem_baseline))
    results_baseline.update(
        linear_thermal_grid_model_set.get_optimization_dlmps(optimization_problem_baseline, price_data)
    )
    results_baseline.update(der_model_set.get_optimization_results(optimization_problem_baseline))

    # ADMM: Electric sub-problem.

    # Log progress.
    mesmo.utils.log_time(f"electric sub-problem setup")

    # Define linear electric grid problem.
    linear_electric_grid_model_set.define_optimization_variables(optimization_problem_electric)
    linear_electric_grid_model_set.define_optimization_parameters(
        optimization_problem_electric,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
    )
    linear_electric_grid_model_set.define_optimization_constraints(optimization_problem_electric)
    linear_electric_grid_model_set.define_optimization_objective(optimization_problem_electric)

    # Define ADMM objective.
    optimization_problem_electric.define_variable(
        "deviation_der_active_power_vector", timestep=scenario_data.timesteps, der=electric_grid_model.ders
    )
    optimization_problem_electric.define_variable(
        "deviation_der_reactive_power_vector", timestep=scenario_data.timesteps, der=electric_grid_model.ders
    )
    optimization_problem_electric.define_constraint(
        ("variable", 1.0, dict(name="deviation_der_active_power_vector")),
        "==",
        ("variable", -1.0, dict(name="der_active_power_vector")),
        ("constant", "admm_exchange_der_active_power"),
    )
    optimization_problem_electric.define_constraint(
        ("variable", 1.0, dict(name="deviation_der_reactive_power_vector")),
        "==",
        ("variable", -1.0, dict(name="der_reactive_power_vector")),
        ("constant", "admm_exchange_der_reactive_power"),
    )
    optimization_problem_electric.define_objective(
        ("variable", "admm_lambda_electric_der_active_power", dict(name="der_active_power_vector")),
        ("variable", "admm_lambda_electric_der_reactive_power", dict(name="der_reactive_power_vector")),
        (
            "variable",
            (
                0.5
                * admm_rho
                * sp.block_diag(
                    [sp.diags(np.real(electric_grid_model.der_power_vector_reference) ** 2)]
                    * len(scenario_data.timesteps)
                )
            ),
            dict(name="deviation_der_active_power_vector"),
            dict(name="deviation_der_active_power_vector"),
        ),
        (
            "variable",
            (
                0.5
                * admm_rho
                * sp.block_diag(
                    [sp.diags(np.imag(electric_grid_model.der_power_vector_reference) ** 2)]
                    * len(scenario_data.timesteps)
                )
            ),
            dict(name="deviation_der_reactive_power_vector"),
            dict(name="deviation_der_reactive_power_vector"),
        ),
    )

    # Log progress.
    mesmo.utils.log_time(f"electric sub-problem setup")

    # ADMM: Thermal sub-problem.

    # Log progress.
    mesmo.utils.log_time(f"thermal sub-problem setup")

    # Define thermal grid problem.
    linear_thermal_grid_model_set.define_optimization_variables(optimization_problem_thermal)
    linear_thermal_grid_model_set.define_optimization_parameters(
        optimization_problem_thermal,
        price_data,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum,
    )
    linear_thermal_grid_model_set.define_optimization_objective(optimization_problem_thermal)
    linear_thermal_grid_model_set.define_optimization_constraints(optimization_problem_thermal)

    # Define ADMM objective.
    optimization_problem_thermal.define_variable(
        "deviation_der_thermal_power_vector", timestep=scenario_data.timesteps, der=thermal_grid_model.ders
    )
    optimization_problem_thermal.define_constraint(
        ("variable", 1.0, dict(name="deviation_der_thermal_power_vector")),
        "==",
        ("variable", -1.0, dict(name="der_thermal_power_vector")),
        ("constant", "admm_exchange_der_thermal_power"),
    )
    optimization_problem_thermal.define_objective(
        ("variable", "admm_lambda_thermal_der_thermal_power", dict(name="der_thermal_power_vector")),
        (
            "variable",
            (
                0.5
                * admm_rho
                * sp.block_diag(
                    [sp.diags(thermal_grid_model.der_thermal_power_vector_reference**2)]
                    * len(scenario_data.timesteps)
                )
            ),
            dict(name="deviation_der_thermal_power_vector"),
            dict(name="deviation_der_thermal_power_vector"),
        ),
    )

    # Log progress.
    mesmo.utils.log_time(f"thermal sub-problem setup")

    # ADMM: Aggregator sub-problem.

    # Log progress.
    mesmo.utils.log_time(f"aggregator sub-problem setup")

    # Define DER problem.
    der_model_set.define_optimization_variables(optimization_problem_aggregator)
    der_model_set.define_optimization_parameters(optimization_problem_aggregator, price_data)
    der_model_set.define_optimization_constraints(optimization_problem_aggregator)
    der_model_set.define_optimization_objective(optimization_problem_aggregator)

    # Define ADMM objective.
    optimization_problem_aggregator.define_variable(
        "deviation_der_active_power_vector", timestep=scenario_data.timesteps, der=electric_grid_model.ders
    )
    optimization_problem_aggregator.define_variable(
        "deviation_der_reactive_power_vector", timestep=scenario_data.timesteps, der=electric_grid_model.ders
    )
    optimization_problem_aggregator.define_variable(
        "deviation_der_thermal_power_vector", timestep=scenario_data.timesteps, der=thermal_grid_model.ders
    )
    optimization_problem_aggregator.define_constraint(
        ("variable", 1.0, dict(name="deviation_der_active_power_vector")),
        "==",
        ("variable", -1.0, dict(name="der_active_power_vector")),
        ("constant", "admm_exchange_der_active_power"),
    )
    optimization_problem_aggregator.define_constraint(
        ("variable", 1.0, dict(name="deviation_der_reactive_power_vector")),
        "==",
        ("variable", -1.0, dict(name="der_reactive_power_vector")),
        ("constant", "admm_exchange_der_reactive_power"),
    )
    optimization_problem_aggregator.define_constraint(
        ("variable", 1.0, dict(name="deviation_der_thermal_power_vector")),
        "==",
        ("variable", -1.0, dict(name="der_thermal_power_vector")),
        ("constant", "admm_exchange_der_thermal_power"),
    )
    optimization_problem_aggregator.define_objective(
        ("variable", "admm_lambda_aggregator_der_active_power", dict(name="der_active_power_vector")),
        ("variable", "admm_lambda_aggregator_der_reactive_power", dict(name="der_reactive_power_vector")),
        ("variable", "admm_lambda_aggregator_der_thermal_power", dict(name="der_thermal_power_vector")),
        (
            "variable",
            (
                0.5
                * admm_rho
                * sp.block_diag(
                    [sp.diags(np.real(electric_grid_model.der_power_vector_reference) ** 2)]
                    * len(scenario_data.timesteps)
                )
            ),
            dict(name="deviation_der_active_power_vector"),
            dict(name="deviation_der_active_power_vector"),
        ),
        (
            "variable",
            (
                0.5
                * admm_rho
                * sp.block_diag(
                    [sp.diags(np.imag(electric_grid_model.der_power_vector_reference) ** 2)]
                    * len(scenario_data.timesteps)
                )
            ),
            dict(name="deviation_der_reactive_power_vector"),
            dict(name="deviation_der_reactive_power_vector"),
        ),
        (
            "variable",
            (
                0.5
                * admm_rho
                * sp.block_diag(
                    [sp.diags(thermal_grid_model.der_thermal_power_vector_reference**2)]
                    * len(scenario_data.timesteps)
                )
            ),
            dict(name="deviation_der_thermal_power_vector"),
            dict(name="deviation_der_thermal_power_vector"),
        ),
    )

    # Log progress.
    mesmo.utils.log_time(f"aggregator sub-problem setup")

    try:
        while admm_continue:

            # Iterate ADMM counter.
            admm_iteration += 1

            # Solve ADMM sub-problems.
            mesmo.utils.log_time(f"ADMM sub-problem solution #{admm_iteration}")
            optimization_problem_electric.solve()
            optimization_problem_thermal.solve()
            optimization_problem_aggregator.solve()
            mesmo.utils.log_time(f"ADMM sub-problem solution #{admm_iteration}")

            # Print objective values.
            print(f"optimization_problem_electric.objective = {optimization_problem_electric.objective}")
            print(f"optimization_problem_thermal.objective = {optimization_problem_thermal.objective}")
            print(f"optimization_problem_aggregator.objective = {optimization_problem_aggregator.objective}")

            # ADMM intermediate steps.

            # Log progress.
            mesmo.utils.log_time(f"ADMM intermediate steps #{admm_iteration}")

            # Get electric sub-problem results.
            results_electric = linear_electric_grid_model_set.get_optimization_results(optimization_problem_electric)

            # Get thermal sub-problem results.
            results_thermal = linear_thermal_grid_model_set.get_optimization_results(optimization_problem_thermal)

            # Get aggregator sub-problem results.
            results_aggregator = der_model_set.get_optimization_results(optimization_problem_aggregator)

            # Update ADMM variables.
            admm_exchange_der_active_power = 0.5 * (
                results_electric["der_active_power_vector"].values
                + results_aggregator["der_active_power_vector"].values
            )
            optimization_problem_electric.parameters[
                "admm_exchange_der_active_power"
            ] = optimization_problem_aggregator.parameters["admm_exchange_der_active_power"] = (
                admm_exchange_der_active_power / np.real(electric_grid_model.der_power_vector_reference)
            ).ravel()
            admm_exchange_der_reactive_power = 0.5 * (
                results_electric["der_reactive_power_vector"].values
                + results_aggregator["der_reactive_power_vector"].values
            )
            optimization_problem_electric.parameters[
                "admm_exchange_der_reactive_power"
            ] = optimization_problem_aggregator.parameters["admm_exchange_der_reactive_power"] = (
                admm_exchange_der_reactive_power / np.imag(electric_grid_model.der_power_vector_reference)
            ).ravel()
            admm_exchange_der_thermal_power = 0.5 * (
                results_thermal["der_thermal_power_vector"].values
                + results_aggregator["der_thermal_power_vector"].values
            )
            optimization_problem_thermal.parameters[
                "admm_exchange_der_thermal_power"
            ] = optimization_problem_aggregator.parameters["admm_exchange_der_thermal_power"] = (
                admm_exchange_der_thermal_power / thermal_grid_model.der_thermal_power_vector_reference
            ).ravel()
            optimization_problem_electric.parameters["admm_lambda_electric_der_active_power"] = (
                optimization_problem_electric.parameters["admm_lambda_electric_der_active_power"]
                + admm_rho
                * (
                    (results_electric["der_active_power_vector"].values - admm_exchange_der_active_power)
                    * np.real(electric_grid_model.der_power_vector_reference)
                ).ravel()
            )
            optimization_problem_electric.parameters["admm_lambda_electric_der_reactive_power"] = (
                optimization_problem_electric.parameters["admm_lambda_electric_der_reactive_power"]
                + admm_rho
                * (
                    (results_electric["der_reactive_power_vector"].values - admm_exchange_der_reactive_power)
                    * np.imag(electric_grid_model.der_power_vector_reference)
                ).ravel()
            )
            optimization_problem_thermal.parameters["admm_lambda_thermal_der_thermal_power"] = (
                optimization_problem_thermal.parameters["admm_lambda_thermal_der_thermal_power"]
                + admm_rho
                * (
                    (results_thermal["der_thermal_power_vector"].values - admm_exchange_der_thermal_power)
                    * thermal_grid_model.der_thermal_power_vector_reference
                ).ravel()
            )
            optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_active_power"] = (
                optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_active_power"]
                + admm_rho
                * (
                    (results_aggregator["der_active_power_vector"].values - admm_exchange_der_active_power)
                    * np.real(electric_grid_model.der_power_vector_reference)
                ).ravel()
            )
            optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_reactive_power"] = (
                optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_reactive_power"]
                + admm_rho
                * (
                    (results_aggregator["der_reactive_power_vector"].values - admm_exchange_der_reactive_power)
                    * np.imag(electric_grid_model.der_power_vector_reference)
                ).ravel()
            )
            optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_thermal_power"] = (
                optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_thermal_power"]
                + admm_rho
                * (
                    (results_aggregator["der_thermal_power_vector"].values - admm_exchange_der_thermal_power)
                    * thermal_grid_model.der_thermal_power_vector_reference
                ).ravel()
            )

            # Calculate residuals.
            admm_primal_residual_der_active_power = np.sum(
                np.sum(
                    np.abs(
                        results_aggregator["der_active_power_vector"].values
                        - results_electric["der_active_power_vector"].values
                    )
                )
            )
            admm_primal_residual_der_reactive_power = np.sum(
                np.sum(
                    np.abs(
                        results_aggregator["der_reactive_power_vector"].values
                        - results_electric["der_reactive_power_vector"].values
                    )
                )
            )
            admm_primal_residual_der_thermal_power = np.sum(
                np.sum(
                    np.abs(
                        results_aggregator["der_thermal_power_vector"].values
                        - results_thermal["der_thermal_power_vector"].values
                    )
                )
            )
            admm_dual_residual_der_active_power = np.sum(
                np.sum(
                    np.abs(
                        optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_active_power"]
                        + optimization_problem_electric.parameters["admm_lambda_electric_der_active_power"]
                    )
                )
            )
            admm_dual_residual_der_reactive_power = np.sum(
                np.sum(
                    np.abs(
                        optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_reactive_power"]
                        + optimization_problem_electric.parameters["admm_lambda_electric_der_reactive_power"]
                    )
                )
            )
            admm_dual_residual_der_thermal_power = np.sum(
                np.sum(
                    np.abs(
                        optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_thermal_power"]
                        + optimization_problem_thermal.parameters["admm_lambda_thermal_der_thermal_power"]
                    )
                )
            )

            admm_primal_residuals = admm_primal_residuals.append(
                pd.Series(
                    {
                        "Thermal pw.": admm_primal_residual_der_thermal_power,
                        "Active pw.": admm_primal_residual_der_active_power,
                        "Reactive pw.": admm_primal_residual_der_reactive_power,
                    }
                ),
                ignore_index=True,
            )
            admm_dual_residuals = admm_dual_residuals.append(
                pd.Series(
                    {
                        "Thermal pw.": admm_dual_residual_der_thermal_power,
                        "Active pw.": admm_dual_residual_der_active_power,
                        "Reactive pw.": admm_dual_residual_der_reactive_power,
                    }
                ),
                ignore_index=True,
            )

            # Print residuals.
            print(f"admm_dual_residuals = \n{admm_dual_residuals.tail()}")
            print(f"admm_primal_residuals = \n{admm_primal_residuals.tail()}")

            # Log progress.
            mesmo.utils.log_time(f"ADMM intermediate steps #{admm_iteration}")

            # ADMM termination condition.
            admm_continue = (admm_iteration < admm_iteration_limit) and (
                (admm_primal_residuals.iloc[-1, :].max() > admm_primal_residual_termination_limit)
                or (admm_dual_residuals.iloc[-1, :].max() > admm_dual_residual_termination_limit)
            )

    except KeyboardInterrupt:
        # Enables manual termination of ADMM loop.
        pass

    # Obtain ADMM parameters.
    admm_lambda_electric_der_active_power = pd.DataFrame(
        optimization_problem_electric.parameters["admm_lambda_electric_der_active_power"].reshape(
            (len(scenario_data.timesteps), len(electric_grid_model.ders))
        )
        / np.real(electric_grid_model.der_power_vector_reference),
        index=scenario_data.timesteps,
        columns=electric_grid_model.ders,
    )
    admm_lambda_electric_der_reactive_power = pd.DataFrame(
        optimization_problem_electric.parameters["admm_lambda_electric_der_reactive_power"].reshape(
            (len(scenario_data.timesteps), len(electric_grid_model.ders))
        )
        / np.imag(electric_grid_model.der_power_vector_reference),
        index=scenario_data.timesteps,
        columns=electric_grid_model.ders,
    )
    admm_lambda_thermal_der_thermal_power = pd.DataFrame(
        optimization_problem_thermal.parameters["admm_lambda_thermal_der_thermal_power"].reshape(
            (len(scenario_data.timesteps), len(thermal_grid_model.ders))
        )
        / thermal_grid_model.der_thermal_power_vector_reference,
        index=scenario_data.timesteps,
        columns=thermal_grid_model.ders,
    )
    admm_lambda_aggregator_der_active_power = pd.DataFrame(
        optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_active_power"].reshape(
            (len(scenario_data.timesteps), len(electric_grid_model.ders))
        )
        / np.real(electric_grid_model.der_power_vector_reference),
        index=scenario_data.timesteps,
        columns=electric_grid_model.ders,
    )
    admm_lambda_aggregator_der_reactive_power = pd.DataFrame(
        optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_reactive_power"].reshape(
            (len(scenario_data.timesteps), len(electric_grid_model.ders))
        )
        / np.imag(electric_grid_model.der_power_vector_reference),
        index=scenario_data.timesteps,
        columns=electric_grid_model.ders,
    )
    admm_lambda_aggregator_der_thermal_power = pd.DataFrame(
        optimization_problem_aggregator.parameters["admm_lambda_aggregator_der_thermal_power"].reshape(
            (len(scenario_data.timesteps), len(thermal_grid_model.ders))
        )
        / thermal_grid_model.der_thermal_power_vector_reference,
        index=scenario_data.timesteps,
        columns=thermal_grid_model.ders,
    )

    # Store residuals.
    admm_primal_residuals.to_csv((results_path / "admm_primal_residuals.csv"))
    admm_dual_residuals.to_csv((results_path / "admm_dual_residuals.csv"))

    # Plots.

    # Modify plot defaults to align with paper style.
    pio.templates.default.layout.update(
        font=go.layout.Font(family=mesmo.config.config["plots"]["plotly_font_family"], size=20)
    )
    if pio.kaleido.scope is not None:
        pio.kaleido.scope.default_width = 1000
        pio.kaleido.scope.default_height = 285
    pio.orca.config.default_width = 1000
    pio.orca.config.default_height = 285

    # Primal residuals.
    figure = px.line(admm_primal_residuals)
    figure.update_traces(line=dict(width=3))
    figure.update_layout(
        yaxis=go.layout.YAxis(type="log"),
        xaxis_title="Iteration",
        yaxis_title="Primal residual<br>[MW<sub>th</sub>h, MWh, MVA<sub>r</sub>h]",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto", orientation="h"),
        margin=go.layout.Margin(l=120, b=50, r=10, t=10),
    )
    # figure.show()
    mesmo.utils.write_figure_plotly(figure, (results_path / "admm_primal_residuals"))

    # Dual residuals.
    figure = px.line(admm_dual_residuals)
    figure.update_traces(line=dict(width=3))
    figure.update_layout(
        yaxis=go.layout.YAxis(type="log"),
        xaxis_title="Iteration",
        yaxis_title="Dual residual [S$]",
        legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto", orientation="h"),
        margin=go.layout.Margin(b=50, r=10, t=10),
    )
    # figure.show()
    mesmo.utils.write_figure_plotly(figure, (results_path / "admm_dual_residuals"))

    # Thermal power.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline["der_thermal_power_vector"].index,
                y=results_baseline["der_thermal_power_vector"].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name="Centralized op.",
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_thermal["der_thermal_power_vector"].index,
                y=results_thermal["der_thermal_power_vector"].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name="Thermal grid op.",
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator["der_thermal_power_vector"].index,
                y=results_aggregator["der_thermal_power_vector"]
                .loc[:, (slice(None), der_name)]
                .iloc[:, 0]
                .abs()
                .values,
                name="Flexible load agg.",
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Thermal power [MW<sub>th</sub>]",
            xaxis=go.layout.XAxis(tickformat="%H:%M"),
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
            margin=go.layout.Margin(b=30, r=30, t=10),
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, (results_path / f"thermal_power_{der_name}"))

    # Active power.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline["der_active_power_vector"].index,
                y=results_baseline["der_active_power_vector"].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name="Centralized op.",
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_electric["der_active_power_vector"].index,
                y=results_electric["der_active_power_vector"].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name="Electric grid op.",
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator["der_active_power_vector"].index,
                y=results_aggregator["der_active_power_vector"].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name="Flexible load agg.",
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Active power [MW]",
            xaxis=go.layout.XAxis(tickformat="%H:%M"),
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
            margin=go.layout.Margin(b=30, r=30, t=10),
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, (results_path / f"active_power_{der_name}"))

    # Reactive power.
    for der_name in der_model_set.der_names:
        # figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline["der_reactive_power_vector"].index,
                y=results_baseline["der_reactive_power_vector"].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name="Centralized op.",
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_electric["der_reactive_power_vector"].index,
                y=results_electric["der_reactive_power_vector"].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name="Electric grid op.",
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator["der_reactive_power_vector"].index,
                y=results_aggregator["der_reactive_power_vector"]
                .loc[:, (slice(None), der_name)]
                .iloc[:, 0]
                .abs()
                .values,
                name="Flexible load agg.",
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Reactive power [MVA<sub>r</sub>]",
            xaxis=go.layout.XAxis(tickformat="%H:%M"),
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
            margin=go.layout.Margin(b=30, r=30, t=10),
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, (results_path / f"reactive_power_{der_name}"))

    # Thermal power price.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline["thermal_grid_total_dlmp_der_thermal_power"].index,
                y=(
                    results_baseline["thermal_grid_total_dlmp_der_thermal_power"]
                    .loc[:, (slice(None), der_name)]
                    .iloc[:, 0]
                ).values,
                name="Centralized op.",
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_thermal_der_thermal_power.index,
                y=(admm_lambda_thermal_der_thermal_power.loc[:, (slice(None), der_name)].iloc[:, 0]).values,
                name="Thermal grid op.",
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_aggregator_der_thermal_power.index,
                y=(-1.0 * admm_lambda_aggregator_der_thermal_power.loc[:, (slice(None), der_name)].iloc[:, 0]).values,
                name="Flexible load agg.",
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Price [S$/MW<sub>th</sub>h]",
            xaxis=go.layout.XAxis(tickformat="%H:%M"),
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
            margin=go.layout.Margin(b=30, r=30, t=10),
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, (results_path / f"price_thermal_power_{der_name}"))

    # Active power price.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline["electric_grid_total_dlmp_der_active_power"].index,
                y=(
                    results_baseline["electric_grid_total_dlmp_der_active_power"]
                    .loc[:, (slice(None), der_name)]
                    .iloc[:, 0]
                ).values,
                name="Centralized op.",
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_electric_der_active_power.index,
                y=(admm_lambda_electric_der_active_power.loc[:, (slice(None), der_name)].iloc[:, 0]).values,
                name="Electric grid op.",
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_aggregator_der_active_power.index,
                y=(-1.0 * admm_lambda_aggregator_der_active_power.loc[:, (slice(None), der_name)].iloc[:, 0]).values,
                name="Flexible load agg.",
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Price [S$/MWh]",
            xaxis=go.layout.XAxis(tickformat="%H:%M"),
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
            margin=go.layout.Margin(b=30, r=30, t=10),
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, (results_path / f"price_active_power_{der_name}"))

    # Reactive power price.
    for der_name in der_model_set.der_names:
        # figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline["electric_grid_total_dlmp_der_reactive_power"].index,
                y=(
                    results_baseline["electric_grid_total_dlmp_der_reactive_power"]
                    .loc[:, (slice(None), der_name)]
                    .iloc[:, 0]
                ).values,
                name="Centralized op.",
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_electric_der_reactive_power.index,
                y=(admm_lambda_electric_der_reactive_power.loc[:, (slice(None), der_name)].iloc[:, 0]).values,
                name="Electric grid op.",
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_aggregator_der_reactive_power.index,
                y=(-1.0 * admm_lambda_aggregator_der_reactive_power.loc[:, (slice(None), der_name)].iloc[:, 0]).values,
                name="Flexible load agg.",
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Price [S$/MVA<sub>r</sub>h]",
            xaxis=go.layout.XAxis(tickformat="%H:%M"),
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
            margin=go.layout.Margin(b=30, r=30, t=10),
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, (results_path / f"price_reactive_power_{der_name}"))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":

    run_all = False

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    if run_all:
        for scenario_number in [None]:
            for admm_rho in [
                1e12,
                1e11,
                1e10,
                1e9,
                1e8,
                1e7,
                1e6,
                1e5,
                1e4,
                1e3,
                1e2,
                1e1,
                1e0,
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
                1e-8,
                1e-9,
                1e-10,
                1e-11,
                1e-12,
            ]:
                try:

                    # Reset timings.
                    mesmo.utils.log_times = dict()

                    # Run ADMM solution.
                    main(scenario_number=scenario_number, admm_rho=admm_rho)

                except (AssertionError, cp.error.SolverError):
                    pass
    else:
        main()
