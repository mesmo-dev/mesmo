"""Run script for centralized energy market clearing example."""

import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.market_models
import fledge.utils
import cobmo.config


def main():

    # Settings.
    scenario_name = 'singapore_benchmark'
    results_path = fledge.utils.get_results_path('run_centralised_dispatch', scenario_name)
    residual_demand_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'residual_demand')
    pv_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'pv_generation')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps
    timestep_interval_hours = (der_model_set.timesteps[1] - der_model_set.timesteps[0]) / pd.Timedelta('1h')

    # Load residual demand data.
    residual_demand = (
        pd.read_csv(os.path.join(residual_demand_data_path, 'scenario_downtowncore.csv'), index_col=0, squeeze=True)
    )
    residual_demand.index = timesteps

    # Load PV generation data.
    pv_generation = (
        pd.read_csv(os.path.join(pv_data_path, 'singapore_10GW_intermittent.csv'), index_col=0, squeeze=True)
    )
    pv_generation.index = timesteps

    # Instantiate results variables.
    cleared_prices = pd.Series(0.0, index=timesteps)
    aggregate_load = pd.Series(0.0, index=timesteps)
    electricity_cost = pd.DataFrame(0.0, der_model_set.der_names, ['centralised'])

    # Setup & solve optimization problem.
    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    # Custom objective for centralized clearing.
    optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    for timestep in der_model_set.timesteps:
        optimization_problem.objective.expr += (
            (
                14.198
                + 0.0139
                * (
                    sum(
                        optimization_problem.output_vector[timestep, der_name, 'grid_electric_power']
                        for der_name in der_model_set.der_names
                    ) / 1e6
                    + residual_demand.loc[timestep]
                    - pv_generation.loc[timestep] / 1e3
                )  # Aggregate demand in MW.
            )
            * sum(
                optimization_problem.output_vector[timestep, der_name, 'grid_electric_power']
                for der_name in der_model_set.der_names
            ) / 1e6
        )
    fledge.utils.solve_optimization(optimization_problem)
    results = der_model_set.get_optimization_results(optimization_problem)
    output_vector = results['output_vector']

    # Calculate cleared prices.
    for timestep in timesteps:
        cleared_prices.loc[timestep] = (
            np.exp(
                3.258
                + 0.000211
                * (
                    sum(
                        output_vector.loc[timestep, (der_name, 'grid_electric_power')]
                        for der_name in der_model_set.der_names
                    ) / 1e6
                    + residual_demand.loc[timestep]
                    - pv_generation.loc[timestep] / 1e3
                )
            )
        )
        aggregate_load.loc[timestep] = (
            sum(
                output_vector.loc[timestep, (der_name, 'grid_electric_power')]
                for der_name in der_model_set.der_names
            ) / 1e6
        )

    for der_name in der_model_set.der_names:
        electricity_cost.loc[der_name, 'centralised'] = (
            np.sum(
                output_vector.loc[:, (der_name, 'grid_electric_power')].values
                * cleared_prices.values
                * timestep_interval_hours / 1e3
            )
        )

    # Print results.
    total_electricity_cost = (cleared_prices * aggregate_load).sum() * timestep_interval_hours
    print(f'Total electricity cost: {total_electricity_cost} S$')

    # Store results.
    output_vector.to_csv(os.path.join(results_path, 'output_vector.csv'))
    electricity_cost.to_csv(os.path.join(results_path, 'electricity_cost.csv'))
    cleared_prices.to_csv(os.path.join(results_path, 'cleared_prices.csv'))
    aggregate_load.to_csv(os.path.join(results_path, 'aggregate_load.csv'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
