"""Run script for peak load reduction evaluation."""

import matplotlib.pyplot as plt
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


def main():

    # Settings.
    scenario_name = 'singapore_downtowncore'
    results_path = fledge.utils.get_results_path('run_peak_load_reduction', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps
    timestep_interval_hours = (der_model_set.timesteps[1] - der_model_set.timesteps[0]) / pd.Timedelta('1h')

    # Obtain price data.
    # - Using `price_type=None` to get flat price.
    price_data = fledge.data_interface.PriceData(scenario_name, price_type=None)

    # Instantiate results variables.
    electric_power = pd.DataFrame(0.0, timesteps, ['baseline', 'peak_shaving'])

    # Setup & solve baseline operation problem.
    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    der_model_set.define_optimization_objective(optimization_problem, price_data)
    fledge.utils.solve_optimization(optimization_problem)

    # Obtain results.
    results = der_model_set.get_optimization_results(optimization_problem)
    output_vector = results['output_vector']
    output_vector.to_csv(os.path.join(results_path, 'output_baseline.csv'))
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            electric_power.loc[timestep, 'baseline'] += (
                output_vector.loc[timestep, (der_name, 'grid_electric_power')]
                / 1e6  # in MW.
            )

    # Setup peak load minimization problem.
    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)

    # Additional peak load variable / constraint / objective.
    optimization_problem.peak_load = pyo.Var(domain=pyo.NonNegativeReals)
    for timestep in der_model_set.timesteps:
        optimization_problem.der_model_constraints.add(
            sum(
                optimization_problem.output_vector[timestep, der_name, 'grid_electric_power']
                for der_name in der_model_set.der_names
            )
            <=
            optimization_problem.peak_load
        )
    optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    optimization_problem.objective.expr += optimization_problem.peak_load
    optimization_problem.objective.expr += (
        1e-2
        * sum(
            optimization_problem.output_vector[timestep, der_name, 'grid_electric_power']
            for der_name in der_model_set.der_names
            for timestep in der_model_set.timesteps
        )
    )

    # Solve peak load minimization problem.
    fledge.utils.solve_optimization(optimization_problem)

    # Obtain results.
    results = der_model_set.get_optimization_results(optimization_problem)
    output_vector = results['output_vector']
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            electric_power.loc[timestep, 'peak_shaving'] += (
                output_vector.loc[timestep, (der_name, 'grid_electric_power')]
                / 1e6  # in MW.
            )

    # Print results.
    peak_load_baseline = electric_power.loc[:, 'baseline'].max()
    print(f'Baseline peak: {peak_load_baseline} MW')
    peak_load_reduced = electric_power.loc[:, 'peak_shaving'].max()
    print(f'Reduced peak: {peak_load_reduced} MW')
    peak_load_reduction = (peak_load_baseline - peak_load_reduced) / peak_load_baseline * 100
    print(f'Peak load reduction: {peak_load_reduction} %')
    baseline_consumption = electric_power.loc[:, 'baseline'].sum() * timestep_interval_hours
    print(f'\nBaseline consumption: {baseline_consumption} MWh')
    adjusted_consumption = electric_power.loc[:, 'peak_shaving'].sum() * timestep_interval_hours
    print(f'Adjusted consumption: {adjusted_consumption} MWh')
    additional_energy = (adjusted_consumption - baseline_consumption) / baseline_consumption * 100
    print(f'Additional energy use: {additional_energy} %')

    # Store results.
    output_vector.to_csv(os.path.join(results_path, 'output_peak_shaving.csv'))
    electric_power.to_csv(os.path.join(results_path, 'electric_power.csv'))

    # Plot and store figure.
    fig, ax = plt.subplots()
    ax.plot(electric_power.loc[:, 'baseline'], label='Baseline')
    ax.plot(electric_power.loc[:, 'peak_shaving'], label='Reduced peak')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Electric power [MW]')
    fig.savefig(os.path.join(results_path, 'electric_power.png'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
