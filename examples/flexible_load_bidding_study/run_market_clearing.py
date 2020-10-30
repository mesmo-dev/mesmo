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
    scenario_name = 'singapore_downtowncore'
    results_path = fledge.utils.get_results_path('run_market_clearing', scenario_name)
    price_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'clearing_price')
    residual_demand_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'residual_demand')
    pv_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'pv_generation')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain market model.
    market_model = fledge.market_models.MarketModel(scenario_name)

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps

    # TODO: Remove surrogate prices
    # Replace prices in the time series with surrogate prices
    clearing_prices = pd.read_csv(os.path.join(price_data_path, 'scenario_downtowncore.csv'), index_col=0)
    market_model.price_timeseries.loc[:, 'price_value'] = clearing_prices['clearing_price'].values / 1000
    flat_prices = pd.DataFrame(10.0, market_model.timesteps, ['price_value']) # For benchmarking

    # Load residual demand data
    residual_demand = pd.read_csv(os.path.join(residual_demand_data_path, 'scenario_downtowncore.csv'), index_col=0, squeeze=True)
    residual_demand.index = timesteps

    # Load PV generation data
    pv_generation = pd.read_csv(os.path.join(pv_data_path, 'singapore_20GW_intermittent.csv'), index_col=0, squeeze=True)
    pv_generation.index = timesteps

    # Obtain electric grid model.
    # electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Build initial forecast model
    forecast_model = forecast.forecast_model.forecastModel()
    price_forecast = forecast_model.forecast_prices(steps=len(timesteps))
    price_forecast.index = timesteps
    forecast_timestep = forecast_model.df['timestep'].iloc[-1]

    # Create dicts to store actual and baseline dispatch quantities
    actual_dispatch = dict.fromkeys(der_model_set.der_names)
    baseline_dispatch = dict.fromkeys(der_model_set.der_names)
    system_load = pd.DataFrame(0.0, timesteps, ['baseline', 'actual'])
    for der_name in der_model_set.der_names:
        actual_dispatch[der_name] = pd.Series(0.0, index=timesteps)
        baseline_dispatch[der_name] = pd.Series(0.0, index=timesteps)

    cleared_prices = pd.Series(0.0, index=timesteps)

    # Create dict to store bids
    der_bids = dict.fromkeys(der_model_set.der_names)
    for der_name in der_model_set.der_names:
        der_bids[der_name] = dict.fromkeys(timesteps)
        for timestep in timesteps:
            der_bids[der_name][timestep] = pd.Series(0.0, index=range(0,4))

    # Obtain benchmark costs (flat price)
    for der_name in der_model_set.der_names:
        # Obtain DERModel from the model set
        der_model = der_model_set.flexible_der_models[der_name]
        optimization_problem = pyo.ConcreteModel()
        der_model.define_optimization_variables(optimization_problem)
        der_model.define_optimization_constraints(optimization_problem)
        der_model.define_optimization_objective(optimization_problem, flat_prices)
        results = der_model.get_optimization_results(optimization_problem)
        (
            state_vector,
            control_vector,
            output_vector
        ) = results['state_vector'], results['control_vector'], results['output_vector']
        baseline_dispatch[der_name] = output_vector['grid_electric_power']
        system_load.loc[:, 'baseline'] += output_vector['grid_electric_power']
        output_vector.to_csv(os.path.join(results_path, f'output_vector_{der_name}.csv'))
    peak_scenario_load = system_load['baseline'].max() / 1e6 # in MW.

    # Obtain price points from initial forecast
    initial_price_forecast = price_forecast.copy()

    # Obtain bids from every building in the scenario
    for timestep in timesteps:
        price_forecast.to_csv(os.path.join(results_path, f'price_forecast_{timestep}.csv'.replace(':','_')))
        # Define price range and points for the current timestep
        lower_price_limit = initial_price_forecast.at[timestep, 'lower_limit']
        upper_price_limit = initial_price_forecast.at[timestep, 'upper_limit']
        price_points = np.linspace(lower_price_limit, upper_price_limit, 4)
        func = partial(get_bids, der_model_set, timestep, price_forecast, actual_dispatch, price_points, der_bids)
        with Pool(16) as p:
            results = p.map(func, der_model_set.der_names)
        for index, der_name in enumerate(der_model_set.der_names):
            der_bids[der_name][timestep].index = results[index].index
            der_bids[der_name][timestep].loc[:] = results[index].values
            # Save bids to CSV
            der_bids[der_name][timestep].to_csv(os.path.join(
                results_path, f'der_bids_{der_name}_{timestep}.csv'.replace(':', '_')
                )
            )
        # for der_name in der_model_set.der_names:
        #     # Obtain DERModel from the model set
        #     der_model = der_model_set.flexible_der_models[der_name]
        #     der_bids[der_name][timestep].index = price_points
        #     for price in price_points:
        #         # Create optimization problem and solve
        #         optimization_problem = pyo.ConcreteModel()
        #         der_model.define_optimization_variables(optimization_problem)
        #         der_model.define_optimization_constraints(optimization_problem)
        #         # Additional variables and constraints for robust optimization
        #         der_model.define_optimization_variables_bids(optimization_problem)
        #         der_model.define_optimization_constraints_bids(optimization_problem,
        #                                                        timestep,
        #                                                        price_forecast,
        #                                                        actual_dispatch[der_name])
        #         # Objective function for robust optimization
        #         der_model.define_optimization_objective_bids(optimization_problem,
        #                                                      timestep,
        #                                                      price,
        #                                                      price_forecast)
        #         results = der_model.get_optimization_results(optimization_problem)
        #         (
        #             state_vector,
        #             control_vector,
        #             output_vector
        #         ) = results['state_vector'], results['control_vector'], results['output_vector']
        #
        #         der_bids[der_name][timestep][price] = output_vector.at[timestep, 'grid_electric_power']
        #
        #     # Convert to block bids
        #     for price in reversed(der_bids[der_name][timestep].index):
        #         der_bids[der_name][timestep].loc[der_bids[der_name][timestep].index < price] -= (
        #             der_bids[der_name][timestep].loc[price]
        #         )
        #     # der_bids[der_name][timestep].iloc[:-1] -= der_bids[der_name][timestep].min()
        #     der_bids[der_name][timestep] = -der_bids[der_name][timestep]  # Convert to negative power
        #
        #     # Save bids to CSV
        #     der_bids[der_name][timestep].to_csv(os.path.join(
        #         results_path, f'der_bids_{der_name}_{timestep}.csv'.replace(':','_')
        #         )
        #     )

        # Pass bids to MCE and obtain clearing price and power dispatch in return
        (
            cleared_price,
            active_power_dispatch
        ) = market_model.clear_market_supply_curves(
            der_bids,
            timestep,
            residual_demand,
            pv_generation,
            scenario='default'
        )
        # (
        #     cleared_price,
        #     active_power_dispatch
        # ) = market_model.clear_market(
        #     der_bids,
        #     timestep
        # )
        print(f"Expected price: {price_forecast.at[timestep, 'expected_price']}")
        print(f'Clearing price for timestep {timestep}: {cleared_price}')
        cleared_prices.loc[timestep] = cleared_price

        # Update power dispatch
        for der_name in der_model_set.der_names:
            actual_dispatch[der_name][timestep] = -active_power_dispatch[der_name] # Convert back to positive to follow CoBMo convention

        # # Optional: update forecast with the new market clearing price
        if timestep == timesteps[-1]: # Skip forecast model update if at the last timestep
            continue
        new_timesteps = timesteps[timesteps > timestep]
        forecast_timestep += dt.timedelta(minutes=30)
        forecast_model.update_model(cleared_price*1000, forecast_timestep) # update_model requires the cleared price to be in $/MWh
        price_forecast = forecast_model.forecast_prices(steps=len(new_timesteps))
        price_forecast.index = new_timesteps

    # Create DataFrame to store electricity costs
    electricity_cost = pd.DataFrame(0.0, der_model_set.der_names, ['baseline', 'bids'])

    for der_name in der_model_set.der_names:
        electricity_cost.loc[der_name, 'baseline'] = np.sum(
                baseline_dispatch[der_name].values * cleared_prices.values
                * 0.5 / 1e3
        )
        electricity_cost.loc[der_name, 'bids'] = np.sum(
                actual_dispatch[der_name].values * cleared_prices.values
                * 0.5 / 1e3
        )
        system_load.loc[:, 'actual'] += actual_dispatch[der_name]

    electricity_cost.to_csv(os.path.join(results_path, 'electricity_cost.csv'))
    cleared_prices.to_csv(os.path.join(results_path, 'cleared_prices.csv'))

    # Store results
    for der_name in der_model_set.der_names:
        dispatch_df = pd.DataFrame({'baseline':baseline_dispatch[der_name], 'actual':actual_dispatch[der_name]})
        dispatch_df.to_csv(
            os.path.join(results_path, f'dispatch_profile_{der_name}.csv')
        )
        # baseline_dispatch[der_name].to_csv(
        #     os.path.join(results_path, f'baseline_dispatch_{der_name}.csv')
        # )
    system_load.to_csv(os.path.join(results_path, 'system_load.csv'))


def get_bids(der_model_set, timestep, price_forecast, actual_dispatch, price_points, der_bids, der_name):
    # Obtain DERModel from the model set
    der_model = der_model_set.flexible_der_models[der_name]
    der_bids[der_name][timestep].index = price_points
    for price in price_points:
        # Create optimization problem and solve
        optimization_problem = pyo.ConcreteModel()
        der_model.define_optimization_variables(optimization_problem)
        der_model.define_optimization_constraints(optimization_problem)
        # Additional variables and constraints for robust optimization
        # der_model.define_optimization_variables_bids(optimization_problem)
        der_model.define_optimization_constraints_bids(optimization_problem,
                                                       timestep,
                                                       price_forecast,
                                                       actual_dispatch[der_name])
        # Objective function for robust optimization
        der_model.define_optimization_objective_bids(optimization_problem,
                                                     timestep,
                                                     price,
                                                     price_forecast)
        results = der_model.get_optimization_results(optimization_problem)
        (
            state_vector,
            control_vector,
            output_vector
        ) = results['state_vector'], results['control_vector'], results['output_vector']

        der_bids[der_name][timestep][price] = output_vector.at[timestep, 'grid_electric_power']

    # Convert to block bids
    for price in reversed(der_bids[der_name][timestep].index):
        der_bids[der_name][timestep].loc[der_bids[der_name][timestep].index < price] -= (
            der_bids[der_name][timestep].loc[price]
        )
    der_bids[der_name][timestep] = -der_bids[der_name][timestep]  # Convert to negative power

    return der_bids[der_name][timestep]


if __name__ == "__main__":
    main()
