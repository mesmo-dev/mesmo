"""Example script for setting up and solving a multi-grid optimal operation problem."""

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import cobmo.database_interface
import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.thermal_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    results_path = (
        os.path.join(
            fledge.config.results_path,
            f'run_multi_grid_optimal_operation_{fledge.config.timestamp}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()
    cobmo.database_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    price_data = fledge.database_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_name = 'energy'
    price_timeseries = price_data.price_timeseries_dict[price_name]

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_grid_model.ets_head_loss = 0.0  # TODO: Document modifications for Thermal Electric DLMP paper
    thermal_grid_model.cooling_plant_efficiency = 10.0  # TODO: Document modifications for Thermal Electric DLMP paper
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    linear_thermal_grid_model = (
        fledge.thermal_grid_models.LinearThermalGridModel(
            thermal_grid_model,
            thermal_power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model variables.
    linear_thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    linear_thermal_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem
    )

    # Define constraints for the connection with the DER power vector of the electric and thermal grids.
    der_model_set.define_optimization_connection_grid(
        optimization_problem,
        power_flow_solution,
        electric_grid_model,
        thermal_power_flow_solution,
        thermal_grid_model
    )

    # Define limit constraints.

    # Electric grid.
    voltage_magnitude_vector_minimum = 0.5 * np.abs(power_flow_solution.node_voltage_vector)
    voltage_magnitude_vector_maximum = 1.5 * np.abs(power_flow_solution.node_voltage_vector)
    branch_power_vector_squared_maximum = 1.5 * np.abs(power_flow_solution.branch_power_vector_1 ** 2)
    linear_electric_grid_model.define_optimization_limits(
        optimization_problem,
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum,
        timesteps=scenario_data.timesteps
    )

    # Thermal grid.
    node_head_vector_minimum = 1.5 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 1.5 * thermal_power_flow_solution.branch_flow_vector
    linear_thermal_grid_model.define_optimization_limits(
        optimization_problem,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum,
        timesteps=scenario_data.timesteps
    )

    # Define electric grid objective.
    # TODO: Not considering loss costs due to unreliable loss model.
    # if optimization_problem.find_component('objective') is None:
    #     optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    # optimization_problem.objective.expr += (
    #     sum(
    #         price_timeseries.at[timestep, 'price_value']
    #         * (
    #             optimization_problem.loss_active_change[timestep]
    #             + np.sum(np.real(power_flow_solution.loss))
    #         )
    #         for timestep in scenario_data.timesteps
    #     )
    # )

    # Define objective.
    linear_thermal_grid_model.define_optimization_objective(
        optimization_problem,
        price_timeseries,
        scenario_data.timesteps
    )

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_timeseries
    )

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

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
        power_flow_solution,
        scenario_data.timesteps,
        in_per_unit=False,
        with_mean=True
    )
    (
        der_thermal_power_vector,
        node_head_vector,
        branch_flow_vector,
        pump_power
    ) = linear_thermal_grid_model.get_optimization_results(
        optimization_problem,
        scenario_data.timesteps,
        in_per_unit=False,
        with_mean=True
    )

    # Print results.
    print(f"der_active_power_vector = \n{der_active_power_vector.to_string()}")
    print(f"der_reactive_power_vector = \n{der_reactive_power_vector.to_string()}")
    print(f"voltage_magnitude_vector = \n{voltage_magnitude_vector.to_string()}")
    print(f"branch_power_vector_1_squared = \n{branch_power_vector_1_squared.to_string()}")
    print(f"branch_power_vector_2_squared = \n{branch_power_vector_2_squared.to_string()}")
    print(f"loss_active = \n{loss_active.to_string()}")
    print(f"loss_reactive = \n{loss_reactive.to_string()}")
    print(f"der_thermal_power_vector = \n{der_thermal_power_vector.to_string()}")
    print(f"node_head_vector = \n{node_head_vector.to_string()}")
    print(f"branch_flow_vector = \n{branch_flow_vector.to_string()}")
    print(f"pump_power = \n{pump_power.to_string()}")

    # Store results as CSV.
    der_active_power_vector.to_csv(os.path.join(results_path, 'der_active_power_vector.csv'))
    der_reactive_power_vector.to_csv(os.path.join(results_path, 'der_reactive_power_vector.csv'))
    voltage_magnitude_vector.to_csv(os.path.join(results_path, 'voltage_magnitude_vector.csv'))
    branch_power_vector_1_squared.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared.csv'))
    branch_power_vector_2_squared.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared.csv'))
    loss_active.to_csv(os.path.join(results_path, 'loss_active.csv'))
    loss_reactive.to_csv(os.path.join(results_path, 'loss_reactive.csv'))
    der_thermal_power_vector.to_csv(os.path.join(results_path, 'der_thermal_power_vector.csv'))
    node_head_vector.to_csv(os.path.join(results_path, 'node_head_vector.csv'))
    branch_flow_vector.to_csv(os.path.join(results_path, 'branch_flow_vector.csv'))
    pump_power.to_csv(os.path.join(results_path, 'pump_power.csv'))

    # Obtain DLMPs.
    (
        voltage_magnitude_vector_minimum_dlmp,
        voltage_magnitude_vector_maximum_dlmp,
        branch_power_vector_1_squared_maximum_dlmp,
        branch_power_vector_2_squared_maximum_dlmp,
        loss_active_dlmp,
        loss_reactive_dlmp,
        electric_grid_energy_dlmp,
        electric_grid_voltage_dlmp,
        electric_grid_congestion_dlmp,
        electric_grid_loss_dlmp
    ) = linear_electric_grid_model.get_optimization_dlmps(
        optimization_problem,
        price_timeseries,
        scenario_data.timesteps
    )
    (
        node_head_vector_minimum_dlmp,
        branch_flow_vector_maximum_dlmp,
        pump_power_dlmp,
        thermal_grid_energy_dlmp,
        thermal_grid_head_dlmp,
        thermal_grid_congestion_dlmp,
        thermal_grid_pump_dlmp
    ) = linear_thermal_grid_model.get_optimization_dlmps(
        optimization_problem,
        price_timeseries,
        scenario_data.timesteps
    )

    # Print DLMPs.
    print(f"voltage_magnitude_vector_minimum_dlmp = \n{voltage_magnitude_vector_minimum_dlmp.to_string()}")
    print(f"voltage_magnitude_vector_maximum_dlmp = \n{voltage_magnitude_vector_maximum_dlmp.to_string()}")
    print(f"branch_power_vector_1_squared_maximum_dlmp = \n{branch_power_vector_1_squared_maximum_dlmp.to_string()}")
    print(f"branch_power_vector_2_squared_maximum_dlmp = \n{branch_power_vector_2_squared_maximum_dlmp.to_string()}")
    print(f"loss_active_dlmp = \n{loss_active_dlmp.to_string()}")
    print(f"loss_reactive_dlmp = \n{loss_reactive_dlmp.to_string()}")
    print(f"node_head_vector_minimum_dlmp = \n{node_head_vector_minimum_dlmp.to_string()}")
    print(f"branch_flow_vector_maximum_dlmp = \n{branch_flow_vector_maximum_dlmp.to_string()}")
    print(f"pump_power_dlmp = \n{pump_power_dlmp.to_string()}")
    print(f"electric_grid_energy_dlmp = \n{electric_grid_energy_dlmp.to_string()}")
    print(f"electric_grid_voltage_dlmp = \n{electric_grid_voltage_dlmp.to_string()}")
    print(f"electric_grid_congestion_dlmp = \n{electric_grid_congestion_dlmp.to_string()}")
    print(f"electric_grid_loss_dlmp = \n{electric_grid_loss_dlmp.to_string()}")
    print(f"thermal_grid_energy_dlmp = \n{thermal_grid_energy_dlmp.to_string()}")
    print(f"thermal_grid_head_dlmp = \n{thermal_grid_head_dlmp.to_string()}")
    print(f"thermal_grid_congestion_dlmp = \n{thermal_grid_congestion_dlmp.to_string()}")
    print(f"thermal_grid_pump_dlmp = \n{thermal_grid_pump_dlmp.to_string()}")

    # Store DLMPs as CSV.
    voltage_magnitude_vector_minimum_dlmp.to_csv(os.path.join(results_path, 'voltage_magnitude_vector_minimum_dlmp.csv'))
    voltage_magnitude_vector_maximum_dlmp.to_csv(os.path.join(results_path, 'voltage_magnitude_vector_maximum_dlmp.csv'))
    branch_power_vector_1_squared_maximum_dlmp.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared_maximum_dlmp.csv'))
    branch_power_vector_2_squared_maximum_dlmp.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared_maximum_dlmp.csv'))
    loss_active_dlmp.to_csv(os.path.join(results_path, 'loss_active_dlmp.csv'))
    loss_reactive_dlmp.to_csv(os.path.join(results_path, 'loss_reactive_dlmp.csv'))
    node_head_vector_minimum_dlmp.to_csv(os.path.join(results_path, 'node_head_vector_minimum_dlmp.csv'))
    branch_flow_vector_maximum_dlmp.to_csv(os.path.join(results_path, 'branch_flow_vector_maximum_dlmp.csv'))
    pump_power_dlmp.to_csv(os.path.join(results_path, 'pump_power_dlmp.csv'))
    electric_grid_energy_dlmp.to_csv(os.path.join(results_path, 'electric_grid_energy_dlmp.csv'))
    electric_grid_voltage_dlmp.to_csv(os.path.join(results_path, 'electric_grid_voltage_dlmp.csv'))
    electric_grid_congestion_dlmp.to_csv(os.path.join(results_path, 'electric_grid_congestion_dlmp.csv'))
    electric_grid_loss_dlmp.to_csv(os.path.join(results_path, 'electric_grid_loss_dlmp.csv'))
    thermal_grid_energy_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_energy_dlmp.csv'))
    thermal_grid_head_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_head_dlmp.csv'))
    thermal_grid_congestion_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_congestion_dlmp.csv'))
    thermal_grid_pump_dlmp.to_csv(os.path.join(results_path, 'thermal_grid_pump_dlmp.csv'))

    # Print results path.
    print("Results are stored in: " + results_path)


if __name__ == '__main__':
    main()
