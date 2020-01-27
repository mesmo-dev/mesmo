"""Example script for setting up and solving a flexible load optimal operation problem."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.power_flow_solvers


def main():

    # Settings.
    scenario_name = "singapore_6node"
    plots = True  # If True, script may produce plots.
    results_path = (
        os.path.join(
            fledge.config.results_path,
            f'run_linear_electric_grid_model_{fledge.config.timestamp}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path) if plots else None

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    timestep_interval = scenario_data.scenario['timestep_interval']  # Define shorthand for indexing 't+1'.
    flexible_load_data = fledge.database_interface.FlexibleLoadData(scenario_name)
    price_data = fledge.database_interface.PriceData(scenario_name)

    # Obtain model.
    der_name = flexible_load_data.flexible_loads['der_name'][0]  # Pick first `der_name`.
    flexible_load_model = fledge.der_models.FlexibleLoadModel(flexible_load_data, der_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define variables.
    optimization_problem.state_vector = pyo.Var(scenario_data.timesteps, flexible_load_model.state_names)
    optimization_problem.control_vector = pyo.Var(scenario_data.timesteps, flexible_load_model.control_names)
    optimization_problem.output_vector = pyo.Var(scenario_data.timesteps, flexible_load_model.output_names)

    # Define constraints.
    optimization_problem.constraints = pyo.ConstraintList()

    # Initial state.
    # TODO: Define initial state in model.
    for state_name in flexible_load_model.state_names:
        optimization_problem.constraints.add(
            optimization_problem.state_vector[scenario_data.timesteps[0], state_name]
            ==
            0.0
        )

    for timestep in scenario_data.timesteps[:-1]:

        # State equation.
        for state_name in flexible_load_model.state_names:
            optimization_problem.constraints.add(
                optimization_problem.state_vector[timestep + timestep_interval, state_name]
                ==
                pyo.quicksum(
                    flexible_load_model.state_matrix.at[state_name, state_name_other]
                    * optimization_problem.state_vector[timestep, state_name_other]
                    for state_name_other in flexible_load_model.state_names
                )
                + pyo.quicksum(
                    flexible_load_model.control_matrix.at[state_name, control_name]
                    * optimization_problem.control_vector[timestep, control_name]
                    for control_name in flexible_load_model.control_names
                )
                + pyo.quicksum(
                    flexible_load_model.disturbance_matrix.at[state_name, disturbance_name]
                    * flexible_load_model.disturbance_timeseries.at[timestep, disturbance_name]
                    for disturbance_name in flexible_load_model.disturbance_names
                )
            )

    for timestep in scenario_data.timesteps:

        # Output equation.
        for output_name in flexible_load_model.output_names:
            optimization_problem.constraints.add(
                optimization_problem.output_vector[timestep, output_name]
                ==
                pyo.quicksum(
                    flexible_load_model.state_output_matrix.at[output_name, state_name]
                    * optimization_problem.state_vector[timestep, state_name]
                    for state_name in flexible_load_model.state_names
                )
                + pyo.quicksum(
                    flexible_load_model.control_output_matrix.at[output_name, control_name]
                    * optimization_problem.control_vector[timestep, control_name]
                    for control_name in flexible_load_model.control_names
                )
                + pyo.quicksum(
                    flexible_load_model.disturbance_output_matrix.at[output_name, disturbance_name]
                    * flexible_load_model.disturbance_timeseries.at[timestep, disturbance_name]
                    for disturbance_name in flexible_load_model.disturbance_names
                )
            )

        # Output limits.
        for output_name in flexible_load_model.output_names:
            optimization_problem.constraints.add(
                optimization_problem.output_vector[timestep, output_name]
                >=
                flexible_load_model.output_minimum_timeseries.at[timestep, output_name]
            )
            optimization_problem.constraints.add(
                optimization_problem.output_vector[timestep, output_name]
                <=
                flexible_load_model.output_maximum_timeseries.at[timestep, output_name]
            )

    # Define objective.
    price_name = 'energy'

    cost = 0.0
    cost += (
        pyo.quicksum(
            -1.0
            * price_data.price_timeseries_dict[price_name].at[timestep, 'price_value']
            * optimization_problem.output_vector[output_name, timestep]
            for timestep in scenario_data.timesteps
            for output_name in ['active_power', 'reactive_power']
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
    # optimization_problem.display()

    # Obtain results.
    state_vector = pd.DataFrame(0.0, index=scenario_data.timesteps, columns=flexible_load_model.state_names)
    control_vector = pd.DataFrame(0.0, index=scenario_data.timesteps, columns=flexible_load_model.control_names)
    output_vector = pd.DataFrame(0.0, index=scenario_data.timesteps, columns=flexible_load_model.output_names)

    for timestep in scenario_data.timesteps:
        for state_name in flexible_load_model.state_names:
            state_vector.at[timestep, state_name] = (
                optimization_problem.state_vector[timestep, state_name].value
            )
        for control_name in flexible_load_model.control_names:
            control_vector.at[timestep, control_name] = (
                optimization_problem.control_vector[timestep, control_name].value
            )
        for output_name in flexible_load_model.output_names:
            output_vector.at[timestep, output_name] = (
                optimization_problem.output_vector[timestep, output_name].value
            )

    # Print results.
    print(f"state_name = \n{state_vector.to_string()}")
    print(f"control_name = \n{control_vector.to_string()}")
    print(f"output_name = \n{output_vector.to_string()}")

    # Plot results.
    if plots:

        for output_name in flexible_load_model.output_names:
            plt.plot(flexible_load_model.output_maximum_timeseries[output_name], label="Maximum", drawstyle='steps-post')
            plt.plot(flexible_load_model.output_minimum_timeseries[output_name], label="Minimum", drawstyle='steps-post')
            plt.plot(output_vector[output_name], label="Optimal", drawstyle='steps-post')
            plt.legend()
            plt.title(f"Output: {output_name}")
            plt.show()
            plt.close()

        plt.plot(price_data.price_timeseries_dict[price_name]['price_value'], drawstyle='steps-post')
        plt.title(f"Price: {price_name}")
        plt.show()
        plt.close()



if __name__ == '__main__':
    main()
