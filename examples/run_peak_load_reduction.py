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
    total_electric_power_baseline = pd.Series(0.0, timesteps)
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            total_electric_power_baseline.loc[timestep] += output_vector.loc[timestep, (der_name, 'grid_electric_power')]
    peak_load_baseline = total_electric_power_baseline.max() / 1e6 # Convert to MW
    print(f'Baseline peak: {peak_load_baseline} MW')

    # Formulate optimization problem - iterate over buildings
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
    total_electric_power_reduced = pd.Series(0.0, timesteps)
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            total_electric_power_reduced.loc[timestep] += output_vector.loc[
                timestep, (der_name, 'grid_electric_power')]
    peak_load_reduced = total_electric_power_reduced.max() / 1e6  # Convert to MW
    print(f'Reduced peak: {peak_load_reduced} MW')


if __name__ == "__main__":
    main()
