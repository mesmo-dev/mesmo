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
    scenario_name = 'singapore_downtowncore'
    results_path = fledge.utils.get_results_path('run_peak_load_reduction', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid model.
    # electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps

    # Create DataFrame to store results
    electric_power = pd.DataFrame(0.0, timesteps, ['baseline', 'peak_shaving'])

    # Baseline planning - flat price
    flat_prices = pd.DataFrame(10.0, timesteps, ['price_value'])
    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    der_model_set.define_optimization_objective(optimization_problem, flat_prices)

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
    peak_load_baseline = electric_power.loc[:, 'baseline'].max() # Convert to MW
    print(f'Baseline peak: {peak_load_baseline} MW')

    # Peak load minimization planning
    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    der_model_set.define_optimization_objective(optimization_problem)

    # Solve optimization problem
    results = der_model_set.get_optimization_results(optimization_problem)
    (
        state_vector,
        control_vector,
        output_vector
    ) = results['state_vector'], results['control_vector'], results['output_vector']

    output_vector.to_csv(os.path.join(results_path, 'output_peak_shaving.csv'))
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            electric_power.loc[timestep, 'peak_shaving'] += output_vector.loc[
                timestep, (der_name, 'grid_electric_power')] / 1e6
    peak_load_reduced = electric_power.loc[:, 'peak_shaving'].max() # Convert to MW
    print(f'Reduced peak: {peak_load_reduced} MW')
    peak_load_reduction = (peak_load_baseline-peak_load_reduced)/peak_load_baseline*100
    print(f'Peak load reduction: {peak_load_reduction}%')
    baseline_consumption = electric_power.loc[:, 'baseline'].sum()*0.5
    print(f'\nBaseline consumption: {baseline_consumption} MWh')
    adjusted_consumption = electric_power.loc[:, 'peak_shaving'].sum()*0.5
    print(f'Adjusted consumption: {adjusted_consumption} MWh')
    additional_energy = (adjusted_consumption-baseline_consumption)/baseline_consumption*100
    print(f'Additional energy use: {additional_energy} %')

    # Store consumption profile
    electric_power.to_csv(os.path.join(results_path, 'electric_power.csv'))

    # Plot and store figure
    fig, ax = plt.subplots()
    ax.plot(electric_power.loc[:, 'baseline'], label='Baseline')
    ax.plot(electric_power.loc[:, 'peak_shaving'], label='Reduced peak')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Electric power [MW]')
    fig.savefig(os.path.join(results_path, 'electric_power.png'))

if __name__ == "__main__":
    main()
