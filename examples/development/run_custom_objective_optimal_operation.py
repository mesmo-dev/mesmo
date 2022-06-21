"""Example script for setting up and solving an electric grid optimal operation problem with custom objective."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = "singapore_6node"
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    price_data = mesmo.data_interface.PriceData(scenario_name)

    # Obtain models.
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModel(scenario_name)
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(
        electric_grid_model, power_flow_solution
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = mesmo.solutions.OptimizationProblem()

    # Define electric grid problem.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference
    linear_electric_grid_model_set.define_optimization_variables(optimization_problem)
    linear_electric_grid_model_set.define_optimization_parameters(
        optimization_problem,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum,)
    linear_electric_grid_model_set.define_optimization_constraints(optimization_problem)
    # Not defining objective via method.
    # linear_electric_grid_model_set.define_optimization_objective(optimization_problem)

    # Define DER problem.
    der_model_set.define_optimization_problem(optimization_problem, price_data)
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_parameters(optimization_problem, price_data)
    der_model_set.define_optimization_constraints(optimization_problem)
    # Not defining objective via method.
    # der_model_set.define_optimization_objective(optimization_problem)

    # Define objective.
    # Modify the following to define a custom objective.

    # Define objective for electric loads.
    # - Defined as cost of electric power supply at the DER node.
    # - Cost for load / demand, revenue for generation / supply.
    # - Only defined here, if not yet defined as cost of electric supply at electric grid source node
    #   in `mesmo.electric_grid_models.LinearElectricGridModelSet.define_optimization_objective`.
    if len(der_model_set.electric_ders) > 0:
        optimization_problem.define_objective(
            (
                "variable",
                "der_active_power_cost",
                dict(
                    name="der_active_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
            ),
            (
                "variable",
                "der_active_power_cost_sensitivity",
                dict(
                    name="der_active_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
                dict(
                    name="der_active_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
            ),
            (
                "variable",
                "der_reactive_power_cost",
                dict(
                    name="der_reactive_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
            ),
            (
                "variable",
                "der_reactive_power_cost_sensitivity",
                dict(
                    name="der_reactive_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
                dict(
                    name="der_reactive_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
            ),
        )

    # Define objective for thermal loads.
    # - Defined as cost of thermal power supply at the DER node.
    # - Only defined here, if not yet defined as cost of thermal supply at thermal grid source node
    #   in `mesmo.thermal_grid_models.LinearThermalGridModelSet.define_optimization_objective`.
    if len(der_model_set.thermal_ders) > 0:
        optimization_problem.define_objective(
            (
                "variable",
                "der_thermal_power_cost",
                dict(
                    name="der_thermal_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.thermal_ders,
                ),
            ),
            (
                "variable",
                "der_thermal_power_cost_sensitivity",
                dict(
                    name="der_thermal_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.thermal_ders,
                ),
                dict(
                    name="der_thermal_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.thermal_ders,
                ),
            ),
        )

    # Define objective for electric generators.
    # - That is: Active power generation cost.
    # - Always defined here as the cost of electric power generation at the DER node.
    if len(der_model_set.electric_ders) > 0:
        optimization_problem.define_objective(
            (
                "variable",
                "der_active_power_marginal_cost",
                dict(
                    name="der_active_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
            ),
        )
        optimization_problem.define_objective(
            (
                "variable",
                "der_reactive_power_marginal_cost",
                dict(
                    name="der_reactive_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.electric_ders,
                ),
            ),
        )

    # Define objective for thermal generators.
    # - That is: Thermal power generation cost.
    # - Always defined here as the cost of thermal power generation at the DER node.
    if len(der_model_set.thermal_ders) > 0:
        optimization_problem.define_objective(
            (
                "variable",
                "der_thermal_power_marginal_cost",
                dict(
                    name="der_thermal_power_vector",
                    timestep=der_model_set.timesteps,
                    der=der_model_set.thermal_ders,
                ),
            ),
        )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = mesmo.problems.Results()
    results.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
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
