"""Example script for setting up and solving an thermal grid optimal operation problem."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.thermal_grid_models


def main():

    # Settings.
    scenario_name = "singapore_tanjongpagar"

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)

    # Obtain model.
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define variables.
    optimization_problem.der_flow_vector = (
        pyo.Var(scenario_data.timesteps.to_list(), thermal_grid_model.der_names)
    )
    optimization_problem.branch_flow_vector = (
        pyo.Var(scenario_data.timesteps.to_list(), thermal_grid_model.branches)
    )
    optimization_problem.source_flow = (
        pyo.Var(scenario_data.timesteps.to_list())
    )

    # Define DER constraints.
    # TODO: Arbitrary constraints to demonstrate the functionality.
    optimization_problem.der_constraints = pyo.ConstraintList()
    for timestep in scenario_data.timesteps:
        for der_name in thermal_grid_model.der_names:
            optimization_problem.der_constraints.add(
                optimization_problem.der_flow_vector[timestep, der_name]
                >=
                0.5
            )
            optimization_problem.der_constraints.add(
                optimization_problem.der_flow_vector[timestep, der_name]
                <=
                1.0
            )

    # Define thermal grid constraints.
    optimization_problem.thermal_grid_constraints = pyo.ConstraintList()
    for timestep in scenario_data.timesteps:
        for node_index, node in enumerate(thermal_grid_model.nodes):
            if node[1] == 'source':
                optimization_problem.thermal_grid_constraints.add(
                    optimization_problem.source_flow[timestep]
                    ==
                    sum(
                        thermal_grid_model.branch_node_incidence_matrix[node_index, branch_index]
                        * optimization_problem.branch_flow_vector[timestep, branch]
                        for branch_index, branch in enumerate(thermal_grid_model.branches)
                    )
                )
            else:
                optimization_problem.thermal_grid_constraints.add(
                    sum(
                        thermal_grid_model.der_node_incidence_matrix[node_index, der_index]
                        * optimization_problem.der_flow_vector[timestep, der_name]
                        for der_index, der_name in enumerate(thermal_grid_model.der_names)
                    )
                    ==
                    sum(
                        thermal_grid_model.branch_node_incidence_matrix[node_index, branch_index]
                        * optimization_problem.branch_flow_vector[timestep, branch]
                        for branch_index, branch in enumerate(thermal_grid_model.branches)
                    )
                )

    # Define objective.
    cost = 0.0
    cost += (
        sum(
            optimization_problem.source_flow[timestep]
            for timestep in scenario_data.timesteps
        )
    )
    optimization_problem.objective = (
        pyo.Objective(
            expr=cost,
            sense=pyo.minimize
        )
    )

    # Solve optimization problem.
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    if optimization_result.solver.termination_condition is not pyo.TerminationCondition.optimal:
        raise Exception(f"Invalid solver termination condition: {optimization_result.solver.termination_condition}")
    optimization_problem.display()


if __name__ == "__main__":
    main()
