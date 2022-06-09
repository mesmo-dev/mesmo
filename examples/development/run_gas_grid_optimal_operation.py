"""Example script for setting up and solving a gas grid optimal operation problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mesmo.data_interface
import mesmo.electric_grid_models
import mesmo.thermal_grid_models
import mesmo.gas_grid_models

import mesmo


def main():

    # Settings.
    scenario_name = 'paper_2022_kleinschmidt_isgt_scenario_1'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name)



    # Obtain models.
    mesmo.utils.log_time(f"model setup")
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModel(scenario_name)
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(
        electric_grid_model, power_flow_solution
    )
    thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_power_flow_solution = mesmo.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    linear_thermal_grid_model_set = mesmo.thermal_grid_models.LinearThermalGridModelSet(
        thermal_grid_model, thermal_power_flow_solution
    )
    gas_grid_model = mesmo.gas_grid_models.GasGridModel(scenario_name)
    gas_flow_solution = mesmo.gas_grid_models.GasFlowSolution(gas_grid_model)
    linear_gas_grid_model_set = (
        mesmo.gas_grid_models.LinearGasGridModelSet(
            gas_grid_model,
            gas_flow_solution
        )
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)
    mesmo.utils.log_time(f"model setup")

    # Instantiate optimization problem.
    optimization_problem = mesmo.utils.OptimizationProblem()

    # Define linear electric grid problem.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(power_flow_solution.node_voltage_vector)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(power_flow_solution.node_voltage_vector)
    branch_power_magnitude_vector_maximum = 1.5 * np.abs(power_flow_solution.branch_power_vector_1)
    linear_electric_grid_model_set.define_optimization_problem(
        optimization_problem,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,
    )

    # Define thermal grid problem.
    node_head_vector_minimum = 1.5 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 1.5 * thermal_power_flow_solution.branch_flow_vector
    # Modify limits for scenarios.
    linear_thermal_grid_model_set.define_optimization_problem(
        optimization_problem,
        price_data,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum,
    )

    # Define gas grid problem.
    node_pressure_vector_minimum = 1.5 * gas_flow_solution.node_pressure_vector
    branch_gas_flow_vector_maximum = 10.0 * gas_flow_solution.branch_gas_flow_vector
    linear_gas_grid_model_set.define_optimization_problem(
        optimization_problem,
        price_data,
        node_pressure_vector_minimum=node_pressure_vector_minimum,
        branch_gas_flow_vector_maximum=branch_gas_flow_vector_maximum
    )

    # Define DER problem.
    der_model_set.define_optimization_problem(optimization_problem, price_data)

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = mesmo.problems.Results()
    results.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
    results.update(linear_thermal_grid_model_set.get_optimization_results(optimization_problem))
    results.update(linear_gas_grid_model_set.get_optimization_results(optimization_problem))
    results.update(der_model_set.get_optimization_results(optimization_problem))

    # Print results.
    print(results)

    # Store results to CSV.
    results.save(results_path)

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
