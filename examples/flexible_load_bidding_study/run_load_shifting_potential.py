"""Run script for load shifting potential quantification."""

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
    scenario_name = 'singapore_benchmark'
    results_path = fledge.utils.get_results_path('run_load_shifting_potential', scenario_name)

    # Parameters.
    price_fluctuation_parameter = 0.4

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain DER model set.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    timesteps = der_model_set.timesteps

    # Obtain price data.
    # - Using `price_type=None` to get flat price.
    price_data = fledge.data_interface.PriceData(scenario_name, price_type=None)

    # Instantiate results variables.
    electric_power_baseline = pd.DataFrame(0.0, timesteps, der_model_set.der_names)
    electric_power_load_shifting_lower = pd.DataFrame(0.0, timesteps, der_model_set.der_names)
    electric_power_load_shifting_upper = pd.DataFrame(0.0, timesteps, der_model_set.der_names)

    # Setup & solve baseline operation problem.
    optimization_problem = pyo.ConcreteModel()
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_constraints(optimization_problem)
    der_model_set.define_optimization_objective(optimization_problem, price_data)
    fledge.utils.solve_optimization(optimization_problem)

    # Obtain results.
    results = der_model_set.get_optimization_results(optimization_problem)
    output_vector = results['output_vector']

    # Store results.
    output_vector.to_csv(os.path.join(results_path, 'output_baseline.csv'))
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            electric_power_baseline.loc[timestep, der_name] += (
                output_vector.loc[timestep, (der_name, 'grid_electric_power')]
            )

    # Load shifting potential quantification.
    for timestep in timesteps:
        for der_name in der_model_set.der_names:
            for scenario in ['lower', 'upper']:

                # Modify price data.
                price_data_modified = price_data.copy()
                if scenario == 'lower':
                    price_data_modified.price_timeseries.loc[timestep, :] *= (1 - price_fluctuation_parameter)
                elif scenario == 'upper':
                    price_data_modified.price_timeseries.loc[timestep, :] *= (1 + price_fluctuation_parameter)

                # Setup & solve load shifting operation problem.
                der_model = der_model_set.flexible_der_models[der_name]
                optimization_problem = pyo.ConcreteModel()
                der_model.define_optimization_variables(optimization_problem)
                der_model.define_optimization_constraints(optimization_problem)
                der_model.define_optimization_objective(optimization_problem, price_data_modified)
                fledge.utils.solve_optimization(optimization_problem)

                # Obtain results.
                results = der_model.get_optimization_results(optimization_problem)
                output_vector = results['output_vector']

                # Store results.
                output_vector.to_csv(
                    os.path.join(
                        results_path,
                        fledge.utils.get_alphanumeric_string(
                            f'output_load_shifting_{der_name}_{scenario}_{timestep}.csv'
                        )
                    )
                )
                if scenario == 'lower':
                    electric_power_load_shifting_lower.loc[timestep, der_name] += (
                        output_vector.loc[timestep, 'grid_electric_power']
                    )
                elif scenario == 'upper':
                    electric_power_load_shifting_upper.loc[timestep, der_name] += (
                        output_vector.loc[timestep, 'grid_electric_power']
                    )

    # Store results.
    electric_power_baseline.to_csv(
        os.path.join(results_path, 'electric_power_baseline.csv')
    )
    electric_power_load_shifting_lower.to_csv(
        os.path.join(results_path, 'electric_power_load_shifting_lower.csv')
    )
    electric_power_load_shifting_upper.to_csv(
        os.path.join(results_path, 'electric_power_load_shifting_upper.csv')
    )

    # Plot and store figures.
    for der_name in der_model_set.der_names:
        fig, ax = plt.subplots()
        ax.plot(
            electric_power_baseline.loc[:, der_name] / 1e6,
            label='Baseline'
        )
        ax.plot(
            electric_power_load_shifting_lower.loc[:, der_name] / 1e6,
            label='Load shifting (lower)'
        )
        ax.plot(
            electric_power_load_shifting_upper.loc[:, der_name] / 1e6,
            label='Load shifting (upper)'
        )
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Electric power [MW]')
        fig.savefig(os.path.join(results_path, f'electric_power_{der_name}.png'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
