"""Example script for setting up and solving a thermal grid optimal operation problem."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.thermal_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    price_data = fledge.database_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_name = 'energy'
    price_timeseries = price_data.price_timeseries_dict[price_name]

    # Obtain models.
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define thermal grid model variables.
    thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    thermal_grid_model.define_optimization_constraints(
        optimization_problem,
        thermal_power_flow_solution,
        scenario_data.timesteps
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(optimization_problem)

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem
    )

    # Define constraints for the connection with the DER power vector of the electric grid.
    der_model_set.define_optimization_connection_grid(
        optimization_problem,
        thermal_power_flow_solution,
        thermal_grid_model
    )

    # Define objective.
    thermal_grid_model.define_optimization_objective(
        optimization_problem,
        thermal_power_flow_solution,
        price_timeseries,
        scenario_data.timesteps
    )

    # Solve optimization problem.
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    if optimization_result.solver.termination_condition is not pyo.TerminationCondition.optimal:
        raise Exception(f"Invalid solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    (
        der_thermal_power_vector,
        branch_flow_vector,
        source_flow,
        branch_head_vector,
        node_head_vector,
        source_head
    ) = thermal_grid_model.get_optimization_results(optimization_problem, scenario_data.timesteps)

    # Print some results.
    print(f"der_thermal_power_vector = \n{der_thermal_power_vector.to_string()}")
    print(f"branch_flow_vector = \n{branch_flow_vector.to_string()}")
    print(f"branch_head_vector = \n{branch_head_vector.to_string()}")
    print(f"node_head_vector = \n{node_head_vector.to_string()}")
    print(f"source_flow = \n{source_flow.to_string()}")
    print(f"source_head = \n{source_head.to_string()}")


if __name__ == "__main__":
    main()
