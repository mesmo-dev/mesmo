"""Example script for setting up and solving an thermal grid optimal operation problem."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface
import fledge.thermal_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = "singapore_tanjongpagar"

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)

    # Obtain model.
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)

    branch_node_incidence_matrix_inverse = (
        scipy.sparse.dok_matrix((len(thermal_grid_model.nodes), len(thermal_grid_model.branches)), dtype=np.float)
    )
    node_index = fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source')
    branch_node_incidence_matrix_inverse[np.ix_(
        node_index,
        range(len(thermal_grid_model.branches))
    )] = (
        scipy.sparse.linalg.inv(np.transpose(
            thermal_grid_model.branch_node_incidence_matrix[node_index, :]
        ))
    )
    branch_node_incidence_matrix_inverse = branch_node_incidence_matrix_inverse.tocsr()

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define variables.
    thermal_grid_model.define_optimization_variables(optimization_problem, scenario_data.timesteps)

    optimization_problem.branch_head_vector = (
        pyo.Var(scenario_data.timesteps.to_list(), thermal_grid_model.branches.to_list())
    )
    optimization_problem.node_head_vector = (
        pyo.Var(scenario_data.timesteps.to_list(), thermal_grid_model.nodes.to_list())
    )
    optimization_problem.source_head = (
        pyo.Var(scenario_data.timesteps.to_list())
    )

    # Define DER constraints.
    # TODO: Arbitrary constraints to demonstrate the functionality.
    optimization_problem.der_constraints = pyo.ConstraintList()
    for timestep in scenario_data.timesteps:
        for der_index, der in enumerate(thermal_grid_model.ders):
            optimization_problem.der_constraints.add(
                optimization_problem.der_thermal_power_vector[timestep, der]
                <=
                0.5 * thermal_grid_model.der_thermal_power_vector_nominal[der_index]
            )
            optimization_problem.der_constraints.add(
                optimization_problem.der_thermal_power_vector[timestep, der]
                >=
                1.0 * thermal_grid_model.der_thermal_power_vector_nominal[der_index]
            )

        for branch_index, branch in enumerate(thermal_grid_model.branches):
            optimization_problem.der_constraints.add(
                optimization_problem.branch_head_vector[timestep, branch]
                ==
                optimization_problem.branch_flow_vector[timestep, branch]
                * thermal_power_flow_solution.branch_flow_vector[branch_index]
                * thermal_power_flow_solution.branch_friction_factor_vector[branch_index]
                * 8.0 * thermal_grid_model.line_length_vector[branch_index]
                / (
                    fledge.config.gravitational_acceleration
                    * thermal_grid_model.line_diameter_vector[branch_index] ** 5
                    * np.pi ** 2
                )
            )

        for node_index, node in enumerate(thermal_grid_model.nodes):
            if thermal_grid_model.nodes.get_level_values('node_type')[node_index] == 'source':
                optimization_problem.der_constraints.add(
                    optimization_problem.node_head_vector[timestep, node]
                    ==
                    0.0
                )
            else:
                optimization_problem.der_constraints.add(
                    optimization_problem.node_head_vector[timestep, node]
                    ==
                    sum(
                        branch_node_incidence_matrix_inverse[node_index, branch_index]
                        * optimization_problem.branch_head_vector[timestep, branch]
                        for branch_index, branch in enumerate(thermal_grid_model.branches)
                    )
                )
            optimization_problem.der_constraints.add(
                -1.0 * optimization_problem.node_head_vector[timestep, node]
                <=
                optimization_problem.source_head[timestep]
            )

    # Define thermal grid constraints.
    thermal_grid_model.define_optimization_constraints(optimization_problem, scenario_data.timesteps)

    # Define objective.
    optimization_problem.objective = (
        pyo.Objective(
            expr=0.0,
            sense=pyo.minimize
        )
    )
    optimization_problem.objective.expr += (
        sum(
            optimization_problem.source_flow[timestep]
            * thermal_grid_model.enthalpy_difference_distribution_water
            * fledge.config.water_density
            / thermal_grid_model.cooling_plant_efficiency
            for timestep in scenario_data.timesteps
        )
    )
    optimization_problem.objective.expr += (
        sum(
            (
                2.0 * optimization_problem.source_head[timestep]
                + thermal_grid_model.ets_head_loss
            )
            * thermal_power_flow_solution.source_flow
            * fledge.config.water_density
            * fledge.config.gravitational_acceleration
            / thermal_grid_model.pump_efficiency_secondary_pump
            for timestep in scenario_data.timesteps
        )
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
        source_flow
    ) = thermal_grid_model.get_optimization_results(optimization_problem, scenario_data.timesteps)

    branch_head_vector = (
        pd.DataFrame(columns=thermal_grid_model.branches, index=scenario_data.timesteps, dtype=np.float)
    )
    node_head_vector = (
        pd.DataFrame(columns=thermal_grid_model.nodes, index=scenario_data.timesteps, dtype=np.float)
    )
    source_head = (
        pd.DataFrame(columns=['total'], index=scenario_data.timesteps, dtype=np.float)
    )

    for timestep in scenario_data.timesteps:

        for branch in thermal_grid_model.branches:
            branch_head_vector.at[timestep, branch] = (
                optimization_problem.branch_head_vector[timestep, branch].value
            )

        for node in thermal_grid_model.nodes:
            node_head_vector.at[timestep, node] = (
                optimization_problem.node_head_vector[timestep, node].value
            )

        source_head.at[timestep, 'total'] = (
            optimization_problem.source_head[timestep].value
        )

    # Print some results.
    print(f"der_thermal_power_vector = \n{der_thermal_power_vector.to_string()}")
    print(f"branch_flow_vector = \n{branch_flow_vector.to_string()}")
    print(f"branch_head_vector = \n{branch_head_vector.to_string()}")
    print(f"node_head_vector = \n{node_head_vector.to_string()}")
    print(f"source_flow = \n{source_flow.to_string()}")
    print(f"source_head = \n{source_head.to_string()}")


if __name__ == "__main__":
    main()
