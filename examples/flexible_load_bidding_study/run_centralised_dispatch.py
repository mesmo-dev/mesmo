"""Run script for energy market clearing example."""

import numpy as np
import pandas as pd
import os
import pyomo.environ as pyo
import datetime as dt
from functools import partial
from multiprocessing import Pool, Manager

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.market_models
import fledge.utils
import forecast.forecast_model  # From cobmo.
import cobmo.config


def main():

    # Settings.
    scenario_name = 'singapore_benchmark'
    results_path = fledge.utils.get_results_path('run_centralised_dispatch', scenario_name)
    residual_demand_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'residual_demand')
    pv_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'pv_generation')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain market model.
    market_model = fledge.market_models.MarketModel(scenario_name)

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps

    # Load residual demand data
    residual_demand = pd.read_csv(os.path.join(residual_demand_data_path, 'scenario_downtowncore.csv'), index_col=0, squeeze=True)
    residual_demand.index = timesteps

    # Load PV generation data
    pv_generation = pd.read_csv(os.path.join(pv_data_path, 'singapore_10GW_intermittent.csv'), index_col=0, squeeze=True)
    pv_generation.index = timesteps

    cleared_prices = pd.Series(0.0, index=timesteps)
    aggregate_load = pd.Series(0.0, index=timesteps)

    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    der_model_set.define_optimization_objective(optimization_problem,
                                                pv_generation,
                                                residual_demand)

    # Solve optimization problem
    results = der_model_set.get_optimization_results(optimization_problem)
    (
        state_vector,
        control_vector,
        output_vector
    ) = results['state_vector'], results['control_vector'], results['output_vector']

    output_vector.to_csv(os.path.join(results_path, 'output_vector.csv'))

    for timestep in timesteps:
        cleared_prices.loc[timestep] = np.exp(
            ( 3.258
             + 0.000211 * (sum(
                        output_vector.loc[timestep, (der_name, 'grid_electric_power')]
                        for der_name in der_model_set.der_names) / 1e6
                         + residual_demand.loc[timestep] - pv_generation.loc[timestep] / 1e3))
        )
        aggregate_load.loc[timestep] = sum(
                        output_vector.loc[timestep, (der_name, 'grid_electric_power')]
                        for der_name in der_model_set.der_names) / 1e6

    cleared_prices.to_csv(os.path.join(results_path, 'cleared_prices.csv'))
    aggregate_load.to_csv(os.path.join(results_path, 'aggregate_load.csv'))

    # Create DataFrame to store electricity costs
    electricity_cost = pd.DataFrame(0.0, der_model_set.der_names, ['centralised'])

    for der_name in der_model_set.der_names:
        electricity_cost.loc[der_name, 'centralised'] = np.sum(
            output_vector.loc[:, (der_name, 'grid_electric_power')].values * cleared_prices.values
            * 0.5 / 1e3
        )

    electricity_cost.to_csv(os.path.join(results_path, 'electricity_cost.csv'))

    total_electricity_cost = (cleared_prices*aggregate_load).sum()*0.5
    print(f'Total electricity cost: {total_electricity_cost}$')


if __name__ == "__main__":
    main()
