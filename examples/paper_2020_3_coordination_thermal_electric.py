"""Run script for reproducing results of the Paper XXX."""

import matplotlib.dates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.plots
import fledge.thermal_grid_models
import fledge.utils


def main(
        scenario_number=None,
        admm_rho=None
):

    # Settings.
    admm_iteration_limit = 100
    admm_rho = 1e-9 if admm_rho is None else admm_rho
    admm_residual_termination_limit = 1e7
    scenario_number = 1 if scenario_number is None else scenario_number
    # Choices:
    # 1 - unconstrained operation,
    # 2 - constrained thermal grid branch flow,
    # 3 - constrained thermal grid pressure head,
    # 4 - constrained electric grid branch power,
    # 5 - constrained electric grid voltage,
    # 6 - added cooling plant (cheaper than source),
    # 7 - added cooling plant (more expensive than source),
    # 8 - added cooling plant (more expensive than source) + constrained thermal grid branch flow,
    # 9 - added very large cooling plant (cheaper than source) + constrained thermal grid branch flow,
    # 10 - added PV plant (cheaper than source),
    # 11 - added PV plant (more expensive than source),
    # 12 - added PV plant (more expensive than source) + constrained electric grid branch power,
    # 13 - added very large PV plant (cheaper than source) + constrained electric grid branch power,
    # 14 - added cooling plant (more expensive than source) + added very large PV plant (cheaper than source)
    #      + constrained electric grid branch power.
    # 15 - added cooling plant (more expensive than source) + added very large PV plant (cheaper than source)
    #      + constrained electric grid branch power + constrained thermal grid branch flow.
    if scenario_number in [1, 2, 3, 4, 5]:
        scenario_name = 'paper_2020_2_scenario_1_2_3_4_5'
    elif scenario_number in [6, 7, 8]:
        scenario_name = 'paper_2020_2_scenario_6_7_8'
    elif scenario_number in [9]:
        scenario_name = 'paper_2020_2_scenario_9'
    elif scenario_number in [10, 11, 12]:
        scenario_name = 'paper_2020_2_scenario_10_11_12'
    elif scenario_number in [13]:
        scenario_name = 'paper_2020_2_scenario_13'
    elif scenario_number in [14]:
        scenario_name = 'paper_2020_2_scenario_14'
    elif scenario_number in [15]:
        scenario_name = 'paper_2020_2_scenario_15'
    else:
        scenario_name = 'singapore_tanjongpagar_modified'

    # Obtain results path.
    results_path = (
        fledge.utils.get_results_path(
            f'paper_2020_3_coordination_thermal_electric_scenario_{scenario_number}_admm_rho{admm_rho}',
            scenario_name
        )
    )

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
    if scenario_number in [2, 8, 9, 15]:
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
    voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_vector_squared_maximum = 100.0 * (electric_grid_model.branch_power_vector_magnitude_reference ** 2)
    # Modify limits for scenarios.
    if scenario_number in [4, 12, 13, 14, 15]:
        branch_power_vector_squared_maximum[
            fledge.utils.get_index(electric_grid_model.branches, branch_name='4')
        ] *= 8.5 / 100.0
    elif scenario_number in [5]:
        voltage_magnitude_vector_minimum[
            fledge.utils.get_index(electric_grid_model.nodes, node_name='15')
        ] *= 0.9985 / 0.5
    else:
        pass

    # Modify DER models depending on scenario.
    # Cooling plant.
    if scenario_number in [6, 9]:
        der_model_set.flexible_der_models['23'].control_output_matrix.at['thermal_power', 'active_power'] *= 2.0
    elif scenario_number in [7, 8, 14, 15]:
        # Cooling plant COP must remain larger than building cooling COP, otherwise cooling plant never dispatched.
        der_model_set.flexible_der_models['23'].control_output_matrix.at['thermal_power', 'active_power'] *= 0.8
    # PV plant.
    if scenario_number in [11, 12]:
        der_model_set.flexible_der_models['24'].marginal_cost = 0.1
    if scenario_number in [15]:
        der_model_set.flexible_der_models['24'].marginal_cost = 0.04

    # Instantiate ADMM variables.
    admm_iteration = 0
    admm_continue = True
    admm_exchange_der_active_power = (
        pd.DataFrame(
            0.0,
            index=scenario_data.timesteps,
            columns=electric_grid_model.ders
        )
    )
    admm_exchange_der_reactive_power = (
        pd.DataFrame(
            0.0,
            index=scenario_data.timesteps,
            columns=electric_grid_model.ders
        )
    )
    admm_lambda_electric_der_active_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=electric_grid_model.ders)
    )
    admm_lambda_electric_der_reactive_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=electric_grid_model.ders)
    )
    admm_lambda_thermal_der_active_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=electric_grid_model.ders)
    )
    admm_lambda_thermal_der_reactive_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=electric_grid_model.ders)
    )
    admm_residuals = pd.DataFrame()

    # Instantiate optimization problems.
    optimization_problem_baseline = pyo.ConcreteModel()
    optimization_problem_electric = pyo.ConcreteModel()
    optimization_problem_thermal = pyo.ConcreteModel()

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
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
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
        power_flow_solution=power_flow_solution,
        thermal_grid_model=thermal_grid_model,
        thermal_power_flow_solution=thermal_power_flow_solution
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
    fledge.utils.solve_optimization(optimization_problem_baseline, enable_duals=True)
    fledge.utils.log_time(f"baseline problem solution")

    # Get baseline results.
    results_baseline = (
        linear_electric_grid_model.get_optimization_results(
            optimization_problem_baseline,
            power_flow_solution,
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
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
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

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem_thermal
    )

    # Define DER electric grid connection variables.
    optimization_problem_thermal.der_active_power_vector_change = (
        pyo.Var(scenario_data.timesteps.to_list(), electric_grid_model.ders.to_list())
    )
    optimization_problem_thermal.der_reactive_power_vector_change = (
        pyo.Var(scenario_data.timesteps.to_list(), electric_grid_model.ders.to_list())
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem_thermal,
        electric_grid_model=electric_grid_model,
        power_flow_solution=power_flow_solution,
        thermal_grid_model=thermal_grid_model,
        thermal_power_flow_solution=thermal_power_flow_solution
    )

    # Log progress.
    fledge.utils.log_time(f"thermal sub-problem setup")

    try:
        while admm_continue:

            # Iterate ADMM counter.
            admm_iteration += 1

            # ADMM: Electric sub-problem.

            # Log progress.
            fledge.utils.log_time(f"electric sub-problem update #{admm_iteration}")

            # Reset objective, if any.
            if optimization_problem_electric.find_component('objective') is not None:
                optimization_problem_electric.objective.expr = 0.0

            # Define electric grid objective.
            linear_electric_grid_model.define_optimization_objective(
                optimization_problem_electric,
                price_data,
                timesteps=scenario_data.timesteps
            )

            # Define ADMM objective.
            optimization_problem_electric.objective.expr += (
                sum(
                    admm_lambda_electric_der_active_power.at[timestep, der]
                    * (
                        optimization_problem_electric.der_active_power_vector_change[timestep, der]
                        + np.real(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_active_power.at[timestep, der]
                    )
                    + admm_lambda_electric_der_reactive_power.at[timestep, der]
                    * (
                        optimization_problem_electric.der_reactive_power_vector_change[timestep, der]
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_reactive_power.at[timestep, der]
                    )
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_electric.der_active_power_vector_change[timestep, der]
                        + np.real(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_active_power.at[timestep, der]
                    ) ** 2
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_electric.der_reactive_power_vector_change[timestep, der]
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_reactive_power.at[timestep, der]
                    ) ** 2
                    for timestep in scenario_data.timesteps
                    for der_index, der in enumerate(electric_grid_model.ders)
                )
            )

            # Log progress.
            fledge.utils.log_time(f"electric sub-problem update #{admm_iteration}")

            # ADMM: Thermal sub-problem.

            # Log progress.
            fledge.utils.log_time(f"thermal sub-problem update #{admm_iteration}")

            # Reset objective, if any.
            if optimization_problem_thermal.find_component('objective') is not None:
                optimization_problem_thermal.objective.expr = 0.0

            # Define thermal grid objective.
            linear_thermal_grid_model.define_optimization_objective(
                optimization_problem_thermal,
                price_data,
                timesteps=scenario_data.timesteps
            )

            # # Define DER objective.
            # der_model_set.define_optimization_objective(
            #     optimization_problem_thermal,
            #     price_data,
            #     electric_grid_model=electric_grid_model,
            #     thermal_grid_model=thermal_grid_model
            # )

            # Define ADMM objective.
            optimization_problem_thermal.objective.expr += (
                sum(
                    admm_lambda_thermal_der_active_power.at[timestep, der]
                    * (
                        optimization_problem_thermal.der_active_power_vector_change[timestep, der]
                        + np.real(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_active_power.at[timestep, der]
                    )
                    + admm_lambda_thermal_der_reactive_power.at[timestep, der]
                    * (
                        optimization_problem_thermal.der_reactive_power_vector_change[timestep, der]
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_reactive_power.at[timestep, der]
                    )
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_thermal.der_active_power_vector_change[timestep, der]
                        + np.real(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_active_power.at[timestep, der]
                    ) ** 2
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_thermal.der_reactive_power_vector_change[timestep, der]
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_reactive_power.at[timestep, der]
                    ) ** 2
                    for timestep in scenario_data.timesteps
                    for der_index, der in enumerate(electric_grid_model.ders)
                )
            )

            # Log progress.
            fledge.utils.log_time(f"thermal sub-problem update #{admm_iteration}")

            # Solve electric sub-problem.
            fledge.utils.log_time(f"electric sub-problem solution #{admm_iteration}")
            fledge.utils.solve_optimization(optimization_problem_electric)
            fledge.utils.log_time(f"electric sub-problem solution #{admm_iteration}")

            # Solve thermal sub-problem.
            fledge.utils.log_time(f"thermal sub-problem solution #{admm_iteration}")
            fledge.utils.solve_optimization(optimization_problem_thermal)
            fledge.utils.log_time(f"thermal sub-problem solution #{admm_iteration}")

            # Print objective values.
            print(f"optimization_problem_electric.objective = {pyo.value(optimization_problem_electric.objective.expr)}")
            print(f"optimization_problem_thermal.objective = {pyo.value(optimization_problem_thermal.objective.expr)}")

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
            results_thermal.update(
                der_model_set.get_optimization_results(
                    optimization_problem_thermal
                )
            )
            der_active_power_vector = (
                pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
            )
            der_reactive_power_vector = (
                pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
            )
            for timestep in scenario_data.timesteps:
                for der_index, der in enumerate(electric_grid_model.ders):
                    der_active_power_vector.at[timestep, der] = (
                        optimization_problem_thermal.der_active_power_vector_change[timestep, der].value
                        + np.real(power_flow_solution.der_power_vector[der_index])
                    )
                    der_reactive_power_vector.at[timestep, der] = (
                        optimization_problem_thermal.der_reactive_power_vector_change[timestep, der].value
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                    )
            results_thermal.update(
                fledge.data_interface.ResultsDict(
                    der_active_power_vector=der_active_power_vector,
                    der_reactive_power_vector=der_reactive_power_vector
                )
            )

            # Update ADMM variables.
            admm_exchange_der_active_power = (
                0.5 * (
                    results_electric['der_active_power_vector']
                    + results_thermal['der_active_power_vector']
                )
            )
            admm_exchange_der_reactive_power = (
                0.5 * (
                    results_electric['der_reactive_power_vector']
                    + results_thermal['der_reactive_power_vector']
                )
            )
            admm_lambda_electric_der_active_power = (
                admm_lambda_electric_der_active_power
                + admm_rho * (
                    results_electric['der_active_power_vector']
                    - admm_exchange_der_active_power
                )
            )
            admm_lambda_electric_der_reactive_power = (
                admm_lambda_electric_der_reactive_power
                + admm_rho * (
                    results_electric['der_reactive_power_vector']
                    - admm_exchange_der_reactive_power
                )
            )
            admm_lambda_thermal_der_active_power = (
                admm_lambda_thermal_der_active_power
                + admm_rho * (
                    results_thermal['der_active_power_vector']
                    - admm_exchange_der_active_power
                )
            )
            admm_lambda_thermal_der_reactive_power = (
                admm_lambda_thermal_der_reactive_power
                + admm_rho * (
                    results_thermal['der_reactive_power_vector']
                    - admm_exchange_der_reactive_power
                )
            )

            # Calculate residuals.
            admm_residual_electric_der_active_power = (
                (
                    results_electric['der_active_power_vector']
                    - results_baseline['der_active_power_vector']
                ).abs().sum().sum()
            )
            admm_residual_electric_der_reactive_power = (
                (
                    results_electric['der_reactive_power_vector']
                    - results_baseline['der_reactive_power_vector']
                ).abs().sum().sum()
            )
            admm_residual_thermal_der_active_power = (
                (
                    results_thermal['der_active_power_vector']
                    - results_baseline['der_active_power_vector']
                ).abs().sum().sum()
            )
            admm_residual_thermal_der_reactive_power = (
                (
                    results_thermal['der_reactive_power_vector']
                    - results_baseline['der_reactive_power_vector']
                ).abs().sum().sum()
            )
            admm_difference_der_active_power = (
                (
                    results_thermal['der_active_power_vector']
                    - results_electric['der_active_power_vector']
                ).abs().sum().sum()
            )
            admm_difference_der_reactive_power = (
                (
                    results_thermal['der_reactive_power_vector']
                    - results_electric['der_reactive_power_vector']
                ).abs().sum().sum()
            )
            admm_residuals = admm_residuals.append(
                dict(
                    admm_residual_electric_der_active_power=admm_residual_electric_der_active_power,
                    admm_residual_electric_der_reactive_power=admm_residual_electric_der_reactive_power,
                    admm_residual_thermal_der_active_power=admm_residual_thermal_der_active_power,
                    admm_residual_thermal_der_reactive_power=admm_residual_thermal_der_reactive_power,
                    admm_difference_der_active_power=admm_difference_der_active_power,
                    admm_difference_der_reactive_power=admm_difference_der_reactive_power
                ),
                ignore_index=True
            )

            # Print residuals.
            print(f"admm_residuals = \n{admm_residuals}")

            # Plot residuals.
            figure = (
                px.line(
                    admm_residuals / admm_residuals.max(),
                    title=f'admm_rho = {admm_rho}'
                )
            )
            figure.write_html(os.path.join(results_path, 'admm_residuals.html'))
            if admm_iteration == 1:
                fledge.utils.launch(os.path.join(results_path, 'admm_residuals.html'))

            # Plot active power.
            values = (
                pd.concat(
                    [
                        results_baseline['der_active_power_vector'],
                        results_electric['der_active_power_vector'],
                        results_thermal['der_active_power_vector']
                    ],
                    axis='columns',
                    keys=['baseline', 'electric', 'thermal'],
                    names=['solution_type']
                ).droplevel('der_type', axis='columns').abs()
            )
            for der_name in der_model_set.der_names:
                figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
                figure.update_traces(fill='tozeroy')
                # figure.show()
                figure.write_image(os.path.join(results_path, f'active_power_{der_name}.png'))

            # Plot reactive power.
            values = (
                pd.concat(
                    [
                        results_baseline['der_reactive_power_vector'],
                        results_electric['der_reactive_power_vector'],
                        results_thermal['der_reactive_power_vector']
                    ],
                    axis='columns',
                    keys=['baseline', 'electric', 'thermal'],
                    names=['solution_type']
                ).droplevel('der_type', axis='columns').abs()
            )
            for der_name in der_model_set.der_names:
                figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
                figure.update_traces(fill='tozeroy')
                # figure.show()
                figure.write_image(os.path.join(results_path, f'reactive_power_{der_name}.png'))

            if admm_iteration == 1:
                fledge.utils.launch(results_path)

            # Log progress.
            fledge.utils.log_time(f"ADMM intermediate steps #{admm_iteration}")

            # ADMM termination condition.
            admm_continue = (
                True
                if (admm_iteration < admm_iteration_limit)
                and (admm_residuals.iloc[-1, :].max() > admm_residual_termination_limit)
                else False
            )

    except KeyboardInterrupt:
        # Enables manual termination of ADMM loop.
        pass

    # Plot residuals.

    # Absolute residuals.
    figure = px.line(admm_residuals)
    figure.update_layout(legend=dict(orientation='h'))
    # figure.show()
    figure.write_image(os.path.join(results_path, 'admm_residuals_absolute.png'))

    # Relative residuals.
    figure = px.line(admm_residuals / admm_residuals.max())
    figure.update_layout(legend=dict(orientation='h'))
    # figure.show()
    figure.write_image(os.path.join(results_path, 'admm_residuals_relative.png'))

    # Store residuals.
    admm_residuals.to_csv(os.path.join(results_path, 'admm_residuals.csv'))

    # Plot results.

    # Active power.
    values = (
        pd.concat(
            [
                results_baseline['der_active_power_vector'],
                results_electric['der_active_power_vector'],
                results_thermal['der_active_power_vector']
            ],
            axis='columns',
            keys=['baseline', 'electric', 'thermal'],
            names=['solution_type']
        ).droplevel('der_type', axis='columns').abs()
    )
    for der_name in der_model_set.der_names:
        figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
        figure.update_traces(fill='tozeroy')
        # figure.show()
        figure.write_image(os.path.join(results_path, f'active_power_{der_name}.png'))

    # Reactive power.
    values = (
        pd.concat(
            [
                results_baseline['der_reactive_power_vector'],
                results_electric['der_reactive_power_vector'],
                results_thermal['der_reactive_power_vector']
            ],
            axis='columns',
            keys=['baseline', 'electric', 'thermal'],
            names=['solution_type']
        ).droplevel('der_type', axis='columns').abs()
    )
    for der_name in der_model_set.der_names:
        figure = px.line(values.loc[:, (slice(None), der_name)].droplevel('der_name', axis='columns'), line_shape='hv')
        figure.update_traces(fill='tozeroy')
        # figure.show()
        figure.write_image(os.path.join(results_path, f'reactive_power_{der_name}.png'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':

    run_all = False

    if run_all:
        for scenario_number in [1]:
            for admm_rho in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
                try:
                    main(
                        scenario_number=scenario_number,
                        admm_rho=admm_rho
                    )
                except AssertionError:
                    pass
    else:
        main()
