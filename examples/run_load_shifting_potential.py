"""Run script for peak load minimization example."""

import numpy as np
import pandas as pd
import os
import pyomo.environ as pyo
import datetime as dt
from matplotlib import pyplot as plt

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
    results_path = fledge.utils.get_results_path('run_load_shifting_potential', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps

    # Create DataFrame to store results
    electric_power = pd.DataFrame(0.0, timesteps, ['baseline', 'load_shifting'])
    electric_power_baseline_per_building = pd.DataFrame(0.0, timesteps, der_model_set.der_names)
    electric_power_load_shifting_per_building_lower = pd.DataFrame(0.0, timesteps, der_model_set.der_names)
    electric_power_load_shifting_per_building_upper = pd.DataFrame(0.0, timesteps, der_model_set.der_names)

    # Build initial forecast model
    # forecast_model = forecast.forecast_model.forecastModel()
    # price_forecast = forecast_model.forecast_prices(steps=len(timesteps))
    # price_forecast.index = timesteps
    # price_forecast.rename({'expected_price': 'price_value'}, axis=1, inplace=True)
    # Use artificial price time series
    price_forecast = pd.DataFrame(0.0, timesteps, ['price_value', 'lower_limit', 'upper_limit'])
    price_forecast.loc[:, 'price_value'] = 1
    fluctuation_parameter = 0.4
    price_forecast.loc[:, 'lower_limit'] = price_forecast['price_value'] * (1-fluctuation_parameter)
    price_forecast.loc[:, 'upper_limit'] = price_forecast['price_value'] * (1+fluctuation_parameter)

    # Baseline planning
    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    der_model_set.define_optimization_objective(optimization_problem, price_forecast)

    # Solve optimization problem
    results = der_model_set.get_optimization_results(optimization_problem)
    (
        state_vector,
        control_vector,
        output_vector
    ) = results['state_vector'], results['control_vector'], results['output_vector']

    output_vector.to_csv(os.path.join(results_path, 'output_baseline.csv'))
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            electric_power.loc[timestep, 'baseline'] += output_vector.loc[timestep, (der_name, 'grid_electric_power')] / 1e6
            electric_power_baseline_per_building.loc[timestep, der_name] += output_vector.loc[timestep, (der_name, 'grid_electric_power')] / 1e6

    # Load shifting potential quantification
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            for scenario in ['lower', 'upper']:
                der_model = der_model_set.flexible_der_models[der_name]
                optimization_problem = pyo.ConcreteModel()
                der_model.define_optimization_variables(optimization_problem)
                der_model.define_optimization_constraints(optimization_problem)
                der_model.define_optimization_objective(optimization_problem, price_forecast, timestep, scenario)

                # Solve optimization problem
                results = der_model.get_optimization_results(optimization_problem)
                (
                    state_vector,
                    control_vector,
                    output_vector
                ) = results['state_vector'], results['control_vector'], results['output_vector']

                output_vector.to_csv(
                    os.path.join(results_path,
                                 f'output_load_shifting_{der_name}_{scenario}_{timestep}.csv'.replace(':', '_')
                    )
                )
                electric_power.loc[timestep, 'load_shifting'] += \
                    output_vector.loc[timestep, 'grid_electric_power'] / 1e6
                if scenario == 'lower':
                    electric_power_load_shifting_per_building_lower.loc[timestep, der_name] += \
                        output_vector.loc[timestep, 'grid_electric_power'] / 1e6
                elif scenario == 'upper':
                    electric_power_load_shifting_per_building_upper.loc[timestep, der_name] += \
                        output_vector.loc[timestep, 'grid_electric_power'] / 1e6

    # Store consumption profile
    electric_power.to_csv(os.path.join(results_path, 'electric_power.csv'))
    electric_power_baseline_per_building.to_csv(os.path.join(results_path, 'electric_power_baseline.csv'))
    electric_power_load_shifting_per_building_lower.to_csv(
        os.path.join(results_path, 'electric_power_load_shifting_lower.csv')
    )
    electric_power_load_shifting_per_building_upper.to_csv(
        os.path.join(results_path, 'electric_power_load_shifting_upper.csv')
    )

    # Plot and store figures
    for der_name in der_model_set.der_names:
        fig, ax = plt.subplots()
        ax.plot(electric_power_baseline_per_building.loc[:, der_name], label='Baseline')
        ax.plot(electric_power_load_shifting_per_building_lower.loc[:, der_name], label='Load shifting (lower)')
        ax.plot(electric_power_load_shifting_per_building_upper.loc[:, der_name], label='Load shifting (upper)')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Electric power [MW]')
        fig.savefig(os.path.join(results_path, f'electric_power_{der_name}.png'))


if __name__ == "__main__":
    main()
