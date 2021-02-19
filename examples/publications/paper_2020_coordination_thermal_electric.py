"""Run script for reproducing results of the Paper XXX."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.plots
import fledge.problems
import fledge.thermal_grid_models
import fledge.utils


def main(
        scenario_number=None,
        admm_rho=None
):

    # Settings.
    admm_iteration_limit = 10000
    admm_rho = 1e-9 if admm_rho is None else admm_rho
    admm_primal_residual_termination_limit = 1e4
    admm_dual_residual_termination_limit = 1e-3
    scenario_number = 2 if scenario_number is None else scenario_number
    # Choices:
    # 1 - unconstrained operation,
    # 2 - constrained thermal grid branch flow,
    # 3 - constrained thermal grid pressure head,
    # 4 - constrained electric grid branch power,
    # 5 - constrained electric grid voltage
    scenario_name = 'paper_2020_3'

    # Obtain results path.
    results_path = fledge.utils.get_results_path(__file__, f'scenario{scenario_number}_rho{admm_rho}_{scenario_name}')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain models.
    fledge.utils.log_time(f"model setup")
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint('singapore_tanjongpagar_modified')
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution('singapore_tanjongpagar_modified')
    linear_thermal_grid_model = (
        fledge.thermal_grid_models.LinearThermalGridModel(
            thermal_grid_model,
            thermal_power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    fledge.utils.log_time(f"model setup")

    # Define thermal grid limits.
    node_head_vector_minimum = 100.0 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 100.0 * np.abs(thermal_power_flow_solution.branch_flow_vector)
    # Modify limits for scenarios.
    if scenario_number in [2]:
        branch_flow_vector_maximum[
            fledge.utils.get_index(thermal_grid_model.branches, branch_name='4')
        ] *= 0.2 / 100.0
    elif scenario_number in [3]:
        node_head_vector_minimum[
            fledge.utils.get_index(thermal_grid_model.nodes, node_name='15')
        ] *= 0.2 / 100.0
    else:
        pass

    # Define electric grid limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 100.0 * electric_grid_model.branch_power_vector_magnitude_reference
    # Modify limits for scenarios.
    if scenario_number in [4]:
        branch_power_magnitude_vector_maximum[
            fledge.utils.get_index(electric_grid_model.branches, branch_name='4')
        ] *= 8.5 / 100.0
    elif scenario_number in [5]:
        node_voltage_magnitude_vector_minimum[
            fledge.utils.get_index(electric_grid_model.nodes, node_name='15')
        ] *= 0.9985 / 0.5
    else:
        pass

    # Instantiate ADMM variables.
    admm_iteration = 0
    admm_continue = True
    admm_primal_residuals = pd.DataFrame(columns=['Thermal pw.', 'Active pw.', 'Reactive pw.'])
    admm_dual_residuals = pd.DataFrame(columns=['Thermal pw.', 'Active pw.', 'Reactive pw.'])

    # Instantiate ADMM optimization parameters.
    admm_exchange_der_active_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(electric_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(electric_grid_model.ders)))
        )
    )
    admm_exchange_der_reactive_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(electric_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(electric_grid_model.ders)))
        )
    )
    admm_exchange_der_thermal_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(thermal_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(thermal_grid_model.ders)))
        )
    )
    admm_lambda_electric_der_active_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(electric_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(electric_grid_model.ders)))
        )
    )
    admm_lambda_electric_der_reactive_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(electric_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(electric_grid_model.ders)))
        )
    )
    admm_lambda_thermal_der_thermal_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(thermal_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(thermal_grid_model.ders)))
        )
    )
    admm_lambda_aggregator_der_active_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(electric_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(electric_grid_model.ders)))
        )
    )
    admm_lambda_aggregator_der_reactive_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(electric_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(electric_grid_model.ders)))
        )
    )
    admm_lambda_aggregator_der_thermal_power = (
        cp.Parameter(
            (len(scenario_data.timesteps), len(thermal_grid_model.ders)),
            value=np.zeros((len(scenario_data.timesteps), len(thermal_grid_model.ders)))
        )
    )

    # Instantiate optimization problems.
    # TODO: Consider timestep_interval_hours in ADMM objective.
    optimization_problem_baseline = fledge.utils.OptimizationProblem()
    optimization_problem_electric = fledge.utils.OptimizationProblem()
    optimization_problem_thermal = fledge.utils.OptimizationProblem()
    optimization_problem_aggregator = fledge.utils.OptimizationProblem()

    # ADMM: Centralized / baseline problem.

    # Log progress.
    fledge.utils.log_time(f"baseline problem setup")

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem_baseline,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem_baseline,
        scenario_data.timesteps,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )

    # Define thermal grid model variables.
    linear_thermal_grid_model.define_optimization_variables(
        optimization_problem_baseline,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    linear_thermal_grid_model.define_optimization_constraints(
        optimization_problem_baseline,
        scenario_data.timesteps,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem_baseline
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem_baseline,
        electric_grid_model=electric_grid_model,
        thermal_grid_model=thermal_grid_model
    )

    # Define electric grid objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem_baseline,
        price_data,
        timesteps=scenario_data.timesteps
    )

    # Define thermal grid objective.
    linear_thermal_grid_model.define_optimization_objective(
        optimization_problem_baseline,
        price_data,
        timesteps=scenario_data.timesteps
    )

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem_baseline,
        price_data,
        electric_grid_model=electric_grid_model,
        thermal_grid_model=thermal_grid_model
    )

    # Log progress.
    fledge.utils.log_time(f"baseline problem setup")

    # Solve baseline problem.
    fledge.utils.log_time(f"baseline problem solution")
    optimization_problem_baseline.solve()
    fledge.utils.log_time(f"baseline problem solution")

    # Get baseline results.
    results_baseline = fledge.problems.Results()
    results_baseline.update(
        linear_electric_grid_model.get_optimization_results(
            optimization_problem_baseline,
            power_flow_solution,
            scenario_data.timesteps
        )
    )
    results_baseline.update(
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem_baseline,
            price_data,
            scenario_data.timesteps
        )
    )
    results_baseline.update(
        linear_thermal_grid_model.get_optimization_results(
            optimization_problem_baseline,
            scenario_data.timesteps
        )
    )
    results_baseline.update(
        linear_thermal_grid_model.get_optimization_dlmps(
            optimization_problem_baseline,
            price_data,
            scenario_data.timesteps
        )
    )
    results_baseline.update(
        der_model_set.get_optimization_results(
            optimization_problem_baseline
        )
    )

    # ADMM: Electric sub-problem.

    # Log progress.
    fledge.utils.log_time(f"electric sub-problem setup")

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem_electric,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem_electric,
        scenario_data.timesteps,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )

    # Define electric grid objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem_electric,
        price_data,
        timesteps=scenario_data.timesteps
    )

    # Define ADMM objective.
    optimization_problem_electric.objective += (
        cp.sum(
            cp.multiply(
                admm_lambda_electric_der_active_power,
                (
                    optimization_problem_electric.der_active_power_vector
                    # - admm_exchange_der_active_power
                )
            )
            + cp.multiply(
                admm_lambda_electric_der_reactive_power,
                (
                    optimization_problem_electric.der_reactive_power_vector
                    # - admm_exchange_der_reactive_power
                )
            )
            + 0.5 * admm_rho
            * (
                optimization_problem_electric.der_active_power_vector
                - admm_exchange_der_active_power
            ) ** 2
            + 0.5 * admm_rho
            * (
                optimization_problem_electric.der_reactive_power_vector
                - admm_exchange_der_reactive_power
            ) ** 2
        )
    )

    # Log progress.
    fledge.utils.log_time(f"electric sub-problem setup")

    # ADMM: Thermal sub-problem.

    # Log progress.
    fledge.utils.log_time(f"thermal sub-problem setup")

    # Define thermal grid model variables.
    linear_thermal_grid_model.define_optimization_variables(
        optimization_problem_thermal,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    linear_thermal_grid_model.define_optimization_constraints(
        optimization_problem_thermal,
        scenario_data.timesteps,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum
    )

    # Define thermal grid objective.
    linear_thermal_grid_model.define_optimization_objective(
        optimization_problem_thermal,
        price_data,
        timesteps=scenario_data.timesteps
    )

    # Define ADMM objective.
    optimization_problem_thermal.objective += (
        cp.sum(
            cp.multiply(
                admm_lambda_thermal_der_thermal_power,
                (
                    optimization_problem_thermal.der_thermal_power_vector
                    # - admm_exchange_der_thermal_power
                )
            )
            + 0.5 * admm_rho
            * (
                optimization_problem_thermal.der_thermal_power_vector
                - admm_exchange_der_thermal_power
            ) ** 2
        )
    )

    # Log progress.
    fledge.utils.log_time(f"thermal sub-problem setup")

    # ADMM: Aggregator sub-problem.

    # Log progress.
    fledge.utils.log_time(f"aggregator sub-problem setup")

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem_aggregator
    )

    # Define DER connection variables.
    optimization_problem_aggregator.der_active_power_vector = (
        cp.Variable((len(scenario_data.timesteps), len(electric_grid_model.ders)))
    )
    optimization_problem_aggregator.der_reactive_power_vector = (
        cp.Variable((len(scenario_data.timesteps), len(electric_grid_model.ders)))
    )
    optimization_problem_aggregator.der_thermal_power_vector = (
        cp.Variable((len(scenario_data.timesteps), len(thermal_grid_model.ders)))
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem_aggregator,
        electric_grid_model=electric_grid_model,
        thermal_grid_model=thermal_grid_model
    )

    # # Define DER objective.
    # der_model_set.define_optimization_objective(
    #     optimization_problem_aggregator,
    #     price_data,
    #     electric_grid_model=electric_grid_model,
    #     thermal_grid_model=thermal_grid_model
    # )

    # Define ADMM objective.
    optimization_problem_aggregator.objective += (
        cp.sum(
            cp.multiply(
                admm_lambda_aggregator_der_active_power,
                (
                    optimization_problem_aggregator.der_active_power_vector
                    # - admm_exchange_der_active_power
                )
            )
            + cp.multiply(
                admm_lambda_aggregator_der_reactive_power,
                (
                    optimization_problem_aggregator.der_reactive_power_vector
                    # - admm_exchange_der_reactive_power
                )
            )
            + 0.5 * admm_rho
            * (
                optimization_problem_aggregator.der_active_power_vector
                - admm_exchange_der_active_power
            ) ** 2
            + 0.5 * admm_rho
            * (
                optimization_problem_aggregator.der_reactive_power_vector
                - admm_exchange_der_reactive_power
            ) ** 2
        )
        + cp.sum(
            cp.multiply(
                admm_lambda_aggregator_der_thermal_power,
                (
                    optimization_problem_aggregator.der_thermal_power_vector
                    # - admm_exchange_der_thermal_power
                )
            )
            + 0.5 * admm_rho
            * (
                optimization_problem_aggregator.der_thermal_power_vector
                - admm_exchange_der_thermal_power
            ) ** 2
        )
    )

    # Log progress.
    fledge.utils.log_time(f"aggregator sub-problem setup")

    try:
        while admm_continue:

            # Iterate ADMM counter.
            admm_iteration += 1

            # Solve ADMM sub-problems.
            fledge.utils.log_time(f"ADMM sub-problem solution #{admm_iteration}")
            optimization_problem_electric.solve(keep_problem=True)
            optimization_problem_thermal.solve(keep_problem=True)
            optimization_problem_aggregator.solve(keep_problem=True)
            fledge.utils.log_time(f"ADMM sub-problem solution #{admm_iteration}")

            # Print objective values.
            print(f"optimization_problem_electric.objective = {optimization_problem_electric.objective.value}")
            print(f"optimization_problem_thermal.objective = {optimization_problem_thermal.objective.value}")
            print(f"optimization_problem_aggregator.objective = {optimization_problem_aggregator.objective.value}")

            # ADMM intermediate steps.

            # Log progress.
            fledge.utils.log_time(f"ADMM intermediate steps #{admm_iteration}")

            # Get electric sub-problem results.
            results_electric = (
                linear_electric_grid_model.get_optimization_results(
                    optimization_problem_electric,
                    power_flow_solution,
                    scenario_data.timesteps
                )
            )

            # Get thermal sub-problem results.
            results_thermal = (
                linear_thermal_grid_model.get_optimization_results(
                    optimization_problem_thermal,
                    scenario_data.timesteps
                )
            )

            # Get aggregator sub-problem results.
            results_aggregator = fledge.problems.Results()
            results_aggregator.update(
                der_model_set.get_optimization_results(
                    optimization_problem_aggregator
                )
            )
            der_active_power_vector = (
                pd.DataFrame(
                    optimization_problem_aggregator.der_active_power_vector.value,
                    columns=electric_grid_model.ders,
                    index=scenario_data.timesteps
                )
            )
            der_reactive_power_vector = (
                pd.DataFrame(
                    optimization_problem_aggregator.der_reactive_power_vector.value,
                    columns=electric_grid_model.ders,
                    index=scenario_data.timesteps
                )
            )
            der_thermal_power_vector = (
                pd.DataFrame(
                    optimization_problem_aggregator.der_thermal_power_vector.value,
                    columns=thermal_grid_model.ders,
                    index=scenario_data.timesteps
                )
            )
            results_aggregator.update(
                fledge.problems.Results(
                    der_active_power_vector=der_active_power_vector,
                    der_reactive_power_vector=der_reactive_power_vector,
                    der_thermal_power_vector=der_thermal_power_vector
                )
            )

            # Update ADMM variables.
            admm_exchange_der_active_power.value = (
                0.5 * (
                    results_electric['der_active_power_vector'].values
                    + results_aggregator['der_active_power_vector'].values
                )
            )
            admm_exchange_der_reactive_power.value = (
                0.5 * (
                    results_electric['der_reactive_power_vector'].values
                    + results_aggregator['der_reactive_power_vector'].values
                )
            )
            admm_exchange_der_thermal_power.value = (
                0.5 * (
                    results_thermal['der_thermal_power_vector'].values
                    + results_aggregator['der_thermal_power_vector'].values
                )
            )
            admm_lambda_electric_der_active_power.value = (
                admm_lambda_electric_der_active_power.value
                + admm_rho * (
                    results_electric['der_active_power_vector'].values
                    - admm_exchange_der_active_power.value
                )
            )
            admm_lambda_electric_der_reactive_power.value = (
                admm_lambda_electric_der_reactive_power.value
                + admm_rho * (
                    results_electric['der_reactive_power_vector'].values
                    - admm_exchange_der_reactive_power.value
                )
            )
            admm_lambda_thermal_der_thermal_power.value = (
                admm_lambda_thermal_der_thermal_power.value
                + admm_rho * (
                    results_thermal['der_thermal_power_vector'].values
                    - admm_exchange_der_thermal_power.value
                )
            )
            admm_lambda_aggregator_der_active_power.value = (
                admm_lambda_aggregator_der_active_power.value
                + admm_rho * (
                    results_aggregator['der_active_power_vector'].values
                    - admm_exchange_der_active_power.value
                )
            )
            admm_lambda_aggregator_der_reactive_power.value = (
                admm_lambda_aggregator_der_reactive_power.value
                + admm_rho * (
                    results_aggregator['der_reactive_power_vector'].values
                    - admm_exchange_der_reactive_power.value
                )
            )
            admm_lambda_aggregator_der_thermal_power.value = (
                admm_lambda_aggregator_der_thermal_power.value
                + admm_rho * (
                    results_aggregator['der_thermal_power_vector'].values
                    - admm_exchange_der_thermal_power.value
                )
            )

            # Calculate residuals.
            admm_primal_residual_der_active_power = (
                np.sum(np.sum(np.abs(
                    results_aggregator['der_active_power_vector'].values
                    - results_electric['der_active_power_vector'].values
                )))
            )
            admm_primal_residual_der_reactive_power = (
                np.sum(np.sum(np.abs(
                    results_aggregator['der_reactive_power_vector'].values
                    - results_electric['der_reactive_power_vector'].values
                )))
            )
            admm_primal_residual_der_thermal_power = (
                np.sum(np.sum(np.abs(
                    results_aggregator['der_thermal_power_vector'].values
                    - results_thermal['der_thermal_power_vector'].values
                )))
            )
            admm_dual_residual_der_active_power = (
                np.sum(np.sum(np.abs(
                    admm_lambda_aggregator_der_active_power.value
                    + admm_lambda_electric_der_active_power.value
                )))
            )
            admm_dual_residual_der_reactive_power = (
                np.sum(np.sum(np.abs(
                    admm_lambda_aggregator_der_reactive_power.value
                    + admm_lambda_electric_der_reactive_power.value
                )))
            )
            admm_dual_residual_der_thermal_power = (
                np.sum(np.sum(np.abs(
                    admm_lambda_aggregator_der_thermal_power.value
                    + admm_lambda_thermal_der_thermal_power.value
                )))
            )

            admm_primal_residuals = admm_primal_residuals.append(
                pd.Series({
                    'Thermal pw.': admm_primal_residual_der_thermal_power,
                    'Active pw.': admm_primal_residual_der_active_power,
                    'Reactive pw.': admm_primal_residual_der_reactive_power,
                }),
                ignore_index=True
            )
            admm_dual_residuals = admm_dual_residuals.append(
                pd.Series({
                    'Thermal pw.': admm_dual_residual_der_thermal_power,
                    'Active pw.': admm_dual_residual_der_active_power,
                    'Reactive pw.': admm_dual_residual_der_reactive_power
                }),
                ignore_index=True
            )

            # Print residuals.
            print(f"admm_dual_residuals = \n{admm_dual_residuals.tail()}")
            print(f"admm_primal_residuals = \n{admm_primal_residuals.tail()}")

            # Log progress.
            fledge.utils.log_time(f"ADMM intermediate steps #{admm_iteration}")

            # ADMM termination condition.
            admm_continue = (
                (admm_iteration < admm_iteration_limit)
                and (
                    (admm_primal_residuals.iloc[-1, :].max() > admm_primal_residual_termination_limit)
                    or (admm_dual_residuals.iloc[-1, :].max() > admm_dual_residual_termination_limit)
                )
            )

    except KeyboardInterrupt:
        # Enables manual termination of ADMM loop.
        pass

    # Obtain ADMM parameters.
    admm_lambda_electric_der_active_power = (
        pd.DataFrame(
            admm_lambda_electric_der_active_power.value,
            index=scenario_data.timesteps,
            columns=electric_grid_model.ders
        )
    )
    admm_lambda_electric_der_reactive_power = (
        pd.DataFrame(
            admm_lambda_electric_der_reactive_power.value,
            index=scenario_data.timesteps,
            columns=electric_grid_model.ders
        )
    )
    admm_lambda_thermal_der_thermal_power = (
        pd.DataFrame(
            admm_lambda_thermal_der_thermal_power.value,
            index=scenario_data.timesteps,
            columns=thermal_grid_model.ders
        )
    )
    admm_lambda_aggregator_der_active_power = (
        pd.DataFrame(
            admm_lambda_aggregator_der_active_power.value,
            index=scenario_data.timesteps,
            columns=electric_grid_model.ders
        )
    )
    admm_lambda_aggregator_der_reactive_power = (
        pd.DataFrame(
            admm_lambda_aggregator_der_reactive_power.value,
            index=scenario_data.timesteps,
            columns=electric_grid_model.ders
        )
    )
    admm_lambda_aggregator_der_thermal_power = (
        pd.DataFrame(
            admm_lambda_aggregator_der_thermal_power.value,
            index=scenario_data.timesteps,
            columns=thermal_grid_model.ders
        )
    )

    # Store residuals.
    admm_primal_residuals.to_csv(os.path.join(results_path, 'admm_primal_residuals.csv'))
    admm_dual_residuals.to_csv(os.path.join(results_path, 'admm_dual_residuals.csv'))

    # Plots.

    # Modify plot defaults to align with paper style.
    pio.templates.default.layout.update(
        font=go.layout.Font(
            family=fledge.config.config['plots']['plotly_font_family'],
            size=20
        )
    )
    pio.kaleido.scope.default_width = pio.orca.config.default_width = pio.kaleido.scope.default_width = 1000
    pio.kaleido.scope.default_height = pio.orca.config.default_height = pio.kaleido.scope.default_height = 285

    # Primal residuals.
    figure = px.line(admm_primal_residuals)
    figure.update_traces(line=dict(width=3))
    figure.update_layout(
        yaxis=go.layout.YAxis(type='log'),
        xaxis_title="Iteration",
        yaxis_title="Primal residual<br>[MW<sub>th</sub>h, MWh, MVA<sub>r</sub>h]",
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto', orientation='h'),
        margin=go.layout.Margin(l=120, b=50, r=10, t=10)
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, 'admm_primal_residuals'))

    # Dual residuals.
    figure = px.line(admm_dual_residuals)
    figure.update_traces(line=dict(width=3))
    figure.update_layout(
        yaxis=go.layout.YAxis(type='log'),
        xaxis_title="Iteration",
        yaxis_title="Dual residual [S$]",
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto', orientation='h'),
        margin=go.layout.Margin(b=50, r=10, t=10)
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, 'admm_dual_residuals'))

    # Thermal power.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline['der_thermal_power_vector'].index,
                y=results_baseline['der_thermal_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_thermal['der_thermal_power_vector'].index,
                y=results_thermal['der_thermal_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Thermal grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator['der_thermal_power_vector'].index,
                y=results_aggregator['der_thermal_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Thermal power [MW<sub>th</sub>]",
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto'),
            margin=go.layout.Margin(b=30, r=30, t=10)
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'thermal_power_{der_name}'))

    # Active power.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline['der_active_power_vector'].index,
                y=results_baseline['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_electric['der_active_power_vector'].index,
                y=results_electric['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Electric grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator['der_active_power_vector'].index,
                y=results_aggregator['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Active power [MW]",
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto'),
            margin=go.layout.Margin(b=30, r=30, t=10)
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'active_power_{der_name}'))

    # Reactive power.
    for der_name in der_model_set.der_names:
        # figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline['der_reactive_power_vector'].index,
                y=results_baseline['der_reactive_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_electric['der_reactive_power_vector'].index,
                y=results_electric['der_reactive_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Electric grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator['der_reactive_power_vector'].index,
                y=results_aggregator['der_reactive_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs().values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Reactive power [MVA<sub>r</sub>]",
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto'),
            margin=go.layout.Margin(b=30, r=30, t=10)
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'reactive_power_{der_name}'))

    # Thermal power price.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline['thermal_grid_total_dlmp_der_thermal_power'].index,
                y=(
                    results_baseline['thermal_grid_total_dlmp_der_thermal_power'].loc[:, (slice(None), der_name)].iloc[:, 0]
                ).values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_thermal_der_thermal_power.index,
                y=(
                    admm_lambda_thermal_der_thermal_power.loc[:, (slice(None), der_name)].iloc[:, 0]
                ).values,
                name='Thermal grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_aggregator_der_thermal_power.index,
                y=(
                    -1.0 * admm_lambda_aggregator_der_thermal_power.loc[:, (slice(None), der_name)].iloc[:, 0]
                ).values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Price [S$/MW<sub>th</sub>h]",
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto'),
            margin=go.layout.Margin(b=30, r=30, t=10)
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_thermal_power_{der_name}'))

    # Active power price.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline['electric_grid_total_dlmp_der_active_power'].index,
                y=(
                    results_baseline['electric_grid_total_dlmp_der_active_power'].loc[:, (slice(None), der_name)].iloc[:, 0]
                    / 1e3
                ).values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_electric_der_active_power.index,
                y=(
                    admm_lambda_electric_der_active_power.loc[:, (slice(None), der_name)].iloc[:, 0]
                    / 1e3
                ).values,
                name='Electric grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_aggregator_der_active_power.index,
                y=(
                    -1.0 * admm_lambda_aggregator_der_active_power.loc[:, (slice(None), der_name)].iloc[:, 0]
                    / 1e3
                ).values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Price [S$/MWh]",
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto'),
            margin=go.layout.Margin(b=30, r=30, t=10)
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_active_power_{der_name}'))

    # Reactive power price.
    for der_name in der_model_set.der_names:
        # figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline['electric_grid_total_dlmp_der_reactive_power'].index,
                y=(
                    results_baseline['electric_grid_total_dlmp_der_reactive_power'].loc[:, (slice(None), der_name)].iloc[:, 0]
                    / 1e3
                ).values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_electric_der_reactive_power.index,
                y=(
                    admm_lambda_electric_der_reactive_power.loc[:, (slice(None), der_name)].iloc[:, 0]
                    / 1e3
                ).values,
                name='Electric grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=admm_lambda_aggregator_der_reactive_power.index,
                y=(
                    -1.0 * admm_lambda_aggregator_der_reactive_power.loc[:, (slice(None), der_name)].iloc[:, 0]
                    / 1e3
                ).values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Price [S$/MVA<sub>r</sub>h]",
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto'),
            margin=go.layout.Margin(b=30, r=30, t=10)
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_reactive_power_{der_name}'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':

    run_all = False

    if run_all:
        for scenario_number in [None]:
            for admm_rho in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
                try:

                    # Reset timings.
                    fledge.utils.log_times = dict()

                    # Run ADMM solution.
                    main(
                        scenario_number=scenario_number,
                        admm_rho=admm_rho
                    )

                except (AssertionError, cp.error.SolverError):
                    pass
    else:
        main()
