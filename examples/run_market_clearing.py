"""Run script for energy market clearing example."""

import numpy as np
import pandas as pd
import os
import pyomo.environ as pyo
import datetime as dt

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.market_models
import fledge.utils
import forecast.forecast_model  # From cobmo.


def main():

    # Settings.
    # scenario_name = 'singapore_6node'
    scenario_name = 'singapore_downtowncore'
    results_path = fledge.utils.get_results_path('run_market_clearing', scenario_name)
    # price_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'clearing_price')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain market model.
    market_model = fledge.market_models.MarketModel(scenario_name)

    # Obtain electric grid model.
    # electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain DERs.
    # ders = electric_grid_model.ders
    # der_power_vector = electric_grid_model.der_power_vector_reference
    # der_active_power_vector = np.real(der_power_vector)
    # print(der_active_power_vector)

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps

    # Build initial forecast model
    forecast_model = forecast.forecast_model.forecastModel()
    price_forecast = forecast_model.forecast_prices(steps=len(timesteps))
    price_forecast.index = timesteps
    # forecast_timestep = forecast_model.df['timestep'].iloc[-1]

    # Create dict to store actual dispatch quantities
    # actual_dispatch = pd.DataFrame(0.0, timesteps, ['clearing_price', 'actual_dispatch'])
    # actual_dispatch['actual_dispatch'] = actual_dispatch.apply(lambda x: dict.fromkeys(der_model_set.der_names), axis=1)
    actual_dispatch = dict.fromkeys(der_model_set.der_names)
    for der_name in der_model_set.der_names:
        actual_dispatch[der_name] = pd.Series(0.0, index=timesteps)

    # Create dict to store bids
    der_bids = dict.fromkeys(der_model_set.der_names)
    for der_name in der_model_set.der_names:
        der_bids[der_name] = dict.fromkeys(timesteps)
        for timestep in timesteps:
            der_bids[der_name][timestep] = pd.Series(0.0, index=range(0,5))

    # Obtain bids from every building in the scenario
    for timestep in timesteps:
        # Define price range and points for the current timestep
        lower_price_limit = price_forecast.at[timestep, 'lower_limit']
        upper_price_limit = price_forecast.at[timestep, 'upper_limit']
        price_points = np.linspace(lower_price_limit, upper_price_limit, 4)
        price_points = np.append(price_points, [4.5])
        for der_name in der_model_set.der_names:
            # Obtain DERModel from the model set
            der_model = der_model_set.flexible_der_models[der_name]
            der_bids[der_name][timestep].index = price_points
            for price in price_points:
                # Create optimization problem and solve
                optimization_problem = pyo.ConcreteModel()
                der_model.define_optimization_variables(optimization_problem)
                der_model.define_optimization_constraints(optimization_problem)
                # Additional variables and constraints for robust optimization
                der_model.define_optimization_variables_bids(optimization_problem)
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

                output_vector.to_csv(
                    os.path.join(results_path, f'output_vector_{timestep}_{der_name}_{price}.csv'.replace(':','_')))

                der_bids[der_name][timestep][price] = output_vector.at[timestep, 'grid_electric_power']

            der_bids[der_name][timestep].iloc[:-1] -= der_bids[der_name][timestep].min() # Convert to block bids
            der_bids[der_name][timestep] = -der_bids[der_name][timestep] # Convert to negative power

        # Pass bids to MCE and obtain clearing price and power dispatch in return
        (
            cleared_price,
            active_power_dispatch
        ) = market_model.clear_market(der_bids, timestep)

        print(f'Clearing price for timestep {timestep}: {cleared_price}')

        # Update power dispatch
        for der_name in der_model_set.der_names:
            actual_dispatch[der_name][timestep] = -active_power_dispatch[der_name] # Convert back to positive to follow CoBMo convention

        # TODO: Update forecast model with new market clearing price
        # if timestep == timesteps[-1]: # Skip forecast model update if at the last timestep
        #     continue
        # new_timesteps = timesteps[timesteps > timestep]
        # forecast_timestep += dt.timedelta(minutes=30)
        # forecast_model.update_model(clearing_prices[timestep], forecast_timestep)
        # price_forecast = forecast_model.forecast_prices(steps=len(new_timesteps))
        # price_forecast.index = new_timesteps

    # # Define abritrary DER bids.
    # der_bids = dict.fromkeys(ders)
    # for der_index, der in enumerate(ders):
    #     der_bids[der] = (
    #         pd.Series(
    #             [der_active_power_vector[der_index] / 2, der_active_power_vector[der_index] / 2],
    #             index=[0.0, 1.0]
    #         )
    #     )

    # # Define arbitrary clearing price.
    # cleared_price = 0.5
    #
    # # Obtain dispatch power.
    # der_active_power_vector_dispatch = np.zeros(der_active_power_vector.shape, dtype=np.float)
    # for der_index, der in enumerate(ders):
    #     if der_active_power_vector[der_index] < 0.0:
    #         der_active_power_vector_dispatch[der_index] += (
    #             der_bids[der].loc[der_bids[der].index > cleared_price].sum()
    #         )
    #     elif der_active_power_vector[der_index] > 0.0:
    #         der_active_power_vector_dispatch[der_index] += (
    #             der_bids[der].loc[der_bids[der].index < cleared_price].sum()
    #         )

    # (
    #     cleared_prices,
    #     der_active_power_vector_dispatch
    # ) = market_model.clear_market(
    #     der_bids
    # )

    # Print results.
    # print(f"der_bids = \n{der_bids}")
    # print(f"der_active_power_vector_dispatch = \n{active_power_vector_dispatch}")

    # Store results
    for der_name in der_model_set.der_names:
        actual_dispatch[der_name].to_csv(
            os.path.join(results_path, f'actual_dispatch_{der_name}.csv')
        )

#
# def obtain_dispatch(
#         clearing_price: float,
#         bids_series: pd.Series
# ) -> float :
#     lowest_price = bids_series.index[0]
#     highest_price = bids_series.index[-1]
#     if clearing_price < lowest_price:
#         dispatch_quantity = bids_series.iloc[0]
#     elif clearing_price > highest_price:
#         dispatch_quantity = bids_series.iloc[-1]
#     else:
#         price_intervals = (
#             pd.arrays.IntervalArray(
#                 [pd.Interval(bids_series.index[j], bids_series.index[j+1]) for j in range(len(bids_series)-1)]
#                 )
#         )
#         selected_index = price_intervals[price_intervals.contains(clearing_price)].right
#         dispatch_quantity = bids_series[selected_index]
#     return dispatch_quantity.values


if __name__ == "__main__":
    main()
