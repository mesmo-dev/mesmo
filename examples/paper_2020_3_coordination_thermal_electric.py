"""Run script for reproducing results of the Paper XXX."""

import matplotlib.dates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
    admm_iteration_limit = 10000
    admm_rho = 1e-9 if admm_rho is None else admm_rho
    admm_primal_residual_termination_limit = 1e1
    admm_dual_residual_termination_limit = 1e-3
    scenario_number = 2 if scenario_number is None else scenario_number
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
    admm_exchange_der_thermal_power = (
        pd.DataFrame(
            0.0,
            index=scenario_data.timesteps,
            columns=thermal_grid_model.ders
        )
    )
    admm_lambda_electric_der_active_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=electric_grid_model.ders)
    )
    admm_lambda_electric_der_reactive_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=electric_grid_model.ders)
    )
    admm_lambda_thermal_der_thermal_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=thermal_grid_model.ders)
    )
    # For DERs, instantiate lambda with price time series.
    admm_lambda_aggregator_der_active_power = (
        -1.0
        * pd.DataFrame(
            # Initialize with price timeseries.
            price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].to_dict(),
            index=electric_grid_model.ders,
            columns=scenario_data.timesteps
        ).T
    )
    admm_lambda_aggregator_der_reactive_power = (
        pd.DataFrame(0.0, index=scenario_data.timesteps, columns=electric_grid_model.ders)
    )
    admm_lambda_aggregator_der_thermal_power = (
        -1.0
        * (thermal_grid_model.cooling_plant_efficiency ** -1)
        * pd.DataFrame(
            # Initialize with price timeseries.
            price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].to_dict(),
            index=thermal_grid_model.ders,
            columns=scenario_data.timesteps
        ).T
    )
    admm_primal_residuals = pd.DataFrame(columns=['Thermal pw.', 'Active pw.', 'Reactive pw.'])
    admm_dual_residuals = pd.DataFrame(columns=['Thermal pw.', 'Active pw.', 'Reactive pw.'])

    # Instantiate optimization problems.
    optimization_problem_baseline = pyo.ConcreteModel()
    optimization_problem_electric = pyo.ConcreteModel()
    optimization_problem_thermal = pyo.ConcreteModel()
    optimization_problem_aggregator = pyo.ConcreteModel()

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
    optimization_problem_aggregator.der_active_power_vector_change = (
        pyo.Var(scenario_data.timesteps.to_list(), electric_grid_model.ders.to_list())
    )
    optimization_problem_aggregator.der_reactive_power_vector_change = (
        pyo.Var(scenario_data.timesteps.to_list(), electric_grid_model.ders.to_list())
    )
    optimization_problem_aggregator.der_thermal_power_vector = (
        pyo.Var(scenario_data.timesteps.to_list(), thermal_grid_model.ders.to_list())
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem_aggregator,
        electric_grid_model=electric_grid_model,
        power_flow_solution=power_flow_solution,
        thermal_grid_model=thermal_grid_model,
        thermal_power_flow_solution=thermal_power_flow_solution
    )

    # Log progress.
    fledge.utils.log_time(f"aggregator sub-problem setup")

    try:
        while admm_continue:

            # Iterate ADMM counter.
            admm_iteration += 1

            # TODO: Consider timestep_interval_hours in ADMM objective.

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

            # Define ADMM objective.
            optimization_problem_thermal.objective.expr += (
                sum(
                    admm_lambda_thermal_der_thermal_power.at[timestep, der]
                    * (
                        optimization_problem_thermal.der_thermal_power_vector[timestep, der]
                        - admm_exchange_der_thermal_power.at[timestep, der]
                    )
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_thermal.der_thermal_power_vector[timestep, der]
                        - admm_exchange_der_thermal_power.at[timestep, der]
                    ) ** 2
                    for timestep in scenario_data.timesteps
                    for der in thermal_grid_model.ders
                )
            )

            # Log progress.
            fledge.utils.log_time(f"thermal sub-problem update #{admm_iteration}")

            # ADMM :Aggregator sub-problem.

            # Log progress.
            fledge.utils.log_time(f"aggregator sub-problem update #{admm_iteration}")

            # Reset objective, if any.
            if optimization_problem_aggregator.find_component('objective') is not None:
                optimization_problem_aggregator.objective.expr = 0.0
            else:
                optimization_problem_aggregator.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)

            # # Define DER objective.
            # der_model_set.define_optimization_objective(
            #     optimization_problem_aggregator,
            #     price_data,
            #     electric_grid_model=electric_grid_model,
            #     thermal_grid_model=thermal_grid_model
            # )

            # Define ADMM objective.
            optimization_problem_aggregator.objective.expr += (
                sum(
                    admm_lambda_aggregator_der_active_power.at[timestep, der]
                    * (
                        optimization_problem_aggregator.der_active_power_vector_change[timestep, der]
                        + np.real(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_active_power.at[timestep, der]
                    )
                    + admm_lambda_aggregator_der_reactive_power.at[timestep, der]
                    * (
                        optimization_problem_aggregator.der_reactive_power_vector_change[timestep, der]
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_reactive_power.at[timestep, der]
                    )
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_aggregator.der_active_power_vector_change[timestep, der]
                        + np.real(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_active_power.at[timestep, der]
                    ) ** 2
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_aggregator.der_reactive_power_vector_change[timestep, der]
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                        - admm_exchange_der_reactive_power.at[timestep, der]
                    ) ** 2
                    for timestep in scenario_data.timesteps
                    for der_index, der in enumerate(electric_grid_model.ders)
                )
                + sum(
                    admm_lambda_aggregator_der_thermal_power.at[timestep, der]
                    * (
                        optimization_problem_aggregator.der_thermal_power_vector[timestep, der]
                        - admm_exchange_der_thermal_power.at[timestep, der]
                    )
                    + 0.5 * admm_rho
                    * (
                        optimization_problem_aggregator.der_thermal_power_vector[timestep, der]
                        - admm_exchange_der_thermal_power.at[timestep, der]
                    ) ** 2
                    for timestep in scenario_data.timesteps
                    for der in thermal_grid_model.ders
                )
            )

            # Log progress.
            fledge.utils.log_time(f"aggregator sub-problem update #{admm_iteration}")

            # Solve electric sub-problem.
            fledge.utils.log_time(f"electric sub-problem solution #{admm_iteration}")
            fledge.utils.solve_optimization(optimization_problem_electric)
            fledge.utils.log_time(f"electric sub-problem solution #{admm_iteration}")

            # Solve thermal sub-problem.
            fledge.utils.log_time(f"thermal sub-problem solution #{admm_iteration}")
            fledge.utils.solve_optimization(optimization_problem_thermal)
            fledge.utils.log_time(f"thermal sub-problem solution #{admm_iteration}")

            # Solve aggregator sub-problem.
            fledge.utils.log_time(f"aggregator sub-problem solution #{admm_iteration}")
            fledge.utils.solve_optimization(optimization_problem_aggregator)
            fledge.utils.log_time(f"aggregator sub-problem solution #{admm_iteration}")

            # Print objective values.
            print(f"optimization_problem_electric.objective = {pyo.value(optimization_problem_electric.objective.expr)}")
            print(f"optimization_problem_thermal.objective = {pyo.value(optimization_problem_thermal.objective.expr)}")
            print(f"optimization_problem_aggregator.objective = {pyo.value(optimization_problem_aggregator.objective.expr)}")

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
            results_aggregator = (
                der_model_set.get_optimization_results(
                    optimization_problem_aggregator
                )
            )
            der_active_power_vector = (
                pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
            )
            der_reactive_power_vector = (
                pd.DataFrame(columns=electric_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
            )
            der_thermal_power_vector = (
                pd.DataFrame(columns=thermal_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
            )
            for timestep in scenario_data.timesteps:
                for der_index, der in enumerate(electric_grid_model.ders):
                    der_active_power_vector.at[timestep, der] = (
                        optimization_problem_aggregator.der_active_power_vector_change[timestep, der].value
                        + np.real(power_flow_solution.der_power_vector[der_index])
                    )
                    der_reactive_power_vector.at[timestep, der] = (
                        optimization_problem_aggregator.der_reactive_power_vector_change[timestep, der].value
                        + np.imag(power_flow_solution.der_power_vector[der_index])
                    )
                for der in thermal_grid_model.ders:
                    der_thermal_power_vector.at[timestep, der] = (
                        optimization_problem_aggregator.der_thermal_power_vector[timestep, der].value
                    )
            results_aggregator.update(
                fledge.data_interface.ResultsDict(
                    der_active_power_vector=der_active_power_vector,
                    der_reactive_power_vector=der_reactive_power_vector,
                    der_thermal_power_vector=der_thermal_power_vector
                )
            )

            # Update ADMM variables.
            admm_exchange_der_active_power = (
                0.5 * (
                    results_electric['der_active_power_vector']
                    + results_aggregator['der_active_power_vector']
                )
            )
            admm_exchange_der_reactive_power = (
                0.5 * (
                    results_electric['der_reactive_power_vector']
                    + results_aggregator['der_reactive_power_vector']
                )
            )
            admm_exchange_der_thermal_power = (
                0.5 * (
                    results_thermal['der_thermal_power_vector']
                    + results_aggregator['der_thermal_power_vector']
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
            admm_lambda_thermal_der_thermal_power = (
                admm_lambda_thermal_der_thermal_power
                + admm_rho * (
                    results_thermal['der_thermal_power_vector']
                    - admm_exchange_der_thermal_power
                )
            )
            admm_lambda_aggregator_der_active_power = (
                admm_lambda_aggregator_der_active_power
                + admm_rho * (
                    results_aggregator['der_active_power_vector']
                    - admm_exchange_der_active_power
                )
            )
            admm_lambda_aggregator_der_reactive_power = (
                admm_lambda_aggregator_der_reactive_power
                + admm_rho * (
                    results_aggregator['der_reactive_power_vector']
                    - admm_exchange_der_reactive_power
                )
            )
            admm_lambda_aggregator_der_thermal_power = (
                admm_lambda_aggregator_der_thermal_power
                + admm_rho * (
                    results_aggregator['der_thermal_power_vector']
                    - admm_exchange_der_thermal_power
                )
            )

            # Calculate residuals.
            # admm_residual_electric_der_active_power = (
            #     (
            #         results_electric['der_active_power_vector']
            #         - results_baseline['der_active_power_vector']
            #     ).abs().sum().sum()
            # )
            # admm_residual_electric_der_reactive_power = (
            #     (
            #         results_electric['der_reactive_power_vector']
            #         - results_baseline['der_reactive_power_vector']
            #     ).abs().sum().sum()
            # )
            # admm_residual_thermal_der_thermal_power = (
            #     (
            #         results_thermal['der_thermal_power_vector']
            #         - results_baseline['der_thermal_power_vector']
            #     ).abs().sum().sum()
            # )
            # admm_residual_aggregator_der_active_power = (
            #     (
            #         results_aggregator['der_active_power_vector']
            #         - results_baseline['der_active_power_vector']
            #     ).abs().sum().sum()
            # )
            # admm_residual_aggregator_der_reactive_power = (
            #     (
            #         results_aggregator['der_reactive_power_vector']
            #         - results_baseline['der_reactive_power_vector']
            #     ).abs().sum().sum()
            # )
            # admm_residual_aggregator_der_thermal_power = (
            #     (
            #         results_aggregator['der_thermal_power_vector']
            #         - results_baseline['der_thermal_power_vector']
            #     ).abs().sum().sum()
            # )
            admm_primal_residual_der_active_power = (
                (
                    results_aggregator['der_active_power_vector']
                    - results_electric['der_active_power_vector']
                ).abs().sum().sum()
            )
            admm_primal_residual_der_reactive_power = (
                (
                    results_aggregator['der_reactive_power_vector']
                    - results_electric['der_reactive_power_vector']
                ).abs().sum().sum()
            )
            admm_primal_residual_der_thermal_power = (
                (
                    results_aggregator['der_thermal_power_vector']
                    - results_thermal['der_thermal_power_vector']
                ).abs().sum().sum()
            )
            admm_dual_residual_der_active_power = (
                (
                    admm_lambda_aggregator_der_active_power
                    + admm_lambda_electric_der_active_power
                ).abs().sum().sum()
            )
            admm_dual_residual_der_reactive_power = (
                (
                    admm_lambda_aggregator_der_reactive_power
                    + admm_lambda_electric_der_reactive_power
                ).abs().sum().sum()
            )
            admm_dual_residual_der_thermal_power = (
                (
                    admm_lambda_aggregator_der_thermal_power
                    + admm_lambda_thermal_der_thermal_power
                ).abs().sum().sum()
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
            print(f"admm_dual_residuals = \n{admm_dual_residuals}")
            print(f"admm_primal_residuals = \n{admm_primal_residuals}")

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
        yaxis_title="Primal residual",
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto', orientation='h'),
        margin=go.layout.Margin(r=10, t=10)
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, 'admm_primal_residuals'))

    # Dual residuals.
    figure = px.line(admm_dual_residuals)
    figure.update_traces(line=dict(width=3))
    figure.update_layout(
        yaxis=go.layout.YAxis(type='log'),
        xaxis_title="Iteration",
        yaxis_title="Dual residual",
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto', orientation='h'),
        margin=go.layout.Margin(r=10, t=10)
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, 'admm_dual_residuals'))

    # Thermal power.
    for der_name in der_model_set.der_names:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=results_baseline['der_thermal_power_vector'].index,
                y=(
                    results_baseline['der_thermal_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_thermal['der_thermal_power_vector'].index,
                y=(
                    results_thermal['der_thermal_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Thermal grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator['der_thermal_power_vector'].index,
                y=(
                    results_aggregator['der_thermal_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Thermal power [MWth]",
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
                y=(
                    results_baseline['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_electric['der_active_power_vector'].index,
                y=(
                    results_electric['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Electric grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator['der_active_power_vector'].index,
                y=(
                    results_aggregator['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
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
                y=(
                    results_baseline['der_reactive_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Centralized op.',
                # fill='tozeroy',
                # line=go.scatter.Line(width=9, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_electric['der_reactive_power_vector'].index,
                y=(
                    results_electric['der_reactive_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Electric grid op.',
                # line=go.scatter.Line(width=6, shape='hv')
            )
        )
        figure.add_trace(
            go.Bar(
                x=results_aggregator['der_reactive_power_vector'].index,
                y=(
                    results_aggregator['der_reactive_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
                    / 1e6
                ).values,
                name='Flexible load agg.',
                # line=go.scatter.Line(width=3, shape='hv')
            )
        )
        figure.update_layout(
            xaxis_title=None,
            yaxis_title="Reactive power [MVAr]",
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto'),
            margin=go.layout.Margin(b=30, r=30, t=10)
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'reactive_power_{der_name}'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':

    run_all = False

    if run_all:
        for scenario_number in [None]:
            for admm_rho in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
                try:
                    main(
                        scenario_number=scenario_number,
                        admm_rho=admm_rho
                    )
                except AssertionError:
                    pass
    else:
        main()
