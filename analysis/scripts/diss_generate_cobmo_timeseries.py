"""Script that generates load profiles from cobmo models based on a decentral optimization problem.
It creates plots that show the coincidence
"""

import matplotlib.pyplot as plt
import matplotlib.dates
import pandas as pd
import numpy as np
import random

import analysis.simulation
import analysis.input
import fledge.data_interface

plots = True  # if True, script will execute plots section

# scenario_name = 'nl_zuidermeer_manual_flat'
scenario_name = 'nl_zuidermeer_manual'
generate_new_data = False

# TODO: simulate 1-2 weeks more in the beginning and end and cut them off

if generate_new_data:
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    print('Loading data...', end='\r')
    fledge.data_interface.recreate_database()

    _, results = analysis.simulation.SolutionEngine.run_optimal_operation_problem(
        scenario_name=scenario_name,
        problem_type='decentral',
        results_path=results_path
    )
else:
    file_name = '/Users/tomschelo/PycharmProjects/fledge/results/diss_generate_cobmo_timeseries_nl_zuidermeer_manual_2021-01-15_11-52-47/output_vector.csv'
    results = fledge.problems.Results
    results.output_vector = pd.read_csv(
        file_name,
        header=[0, 1],
        index_col=0
    )
    # convert index to datetime
    results.output_vector.index = pd.to_datetime(results.output_vector.index)

if plots:
    timesteps = pd.date_range('2017-01-01', periods=24, freq='1H')

    # Generate array with colors for plots
    colors = list(color['color'] for color in matplotlib.rcParams['axes.prop_cycle'])

    # Get all DER names
    der_names = results.output_vector.columns.get_level_values(0).unique().to_list()

    # for der_name in der_names:
    #
    #     active_power_timeseries = results.output_vector.loc[:, (der_name, 'grid_electric_power')]
    #     # plot_data = pd.DataFrame(columns=['min', 'mean', 'max'], index=timesteps)
    #     #
    #     # for timestep in timesteps:
    #     #     plot_data.loc[timestep, 'min'] = np.min((-1)*der_model.active_power_nominal_timeseries.at_time(timestep.time()))
    #     #     plot_data.loc[timestep, 'max'] = np.max((-1)*der_model.active_power_nominal_timeseries.at_time(timestep.time()))
    #     #     plot_data.loc[timestep, 'mean'] = np.mean((-1)*der_model.active_power_nominal_timeseries.at_time(timestep.time()))
    #
    #     plot_data = pd.DataFrame(0, columns=['Total in kW'], index=timesteps)
    #
    #     for timestep in timesteps:
    #         plot_data.loc[timestep, 'Total in kW'] += 1000 * np.mean(active_power_timeseries.at_time(timestep.time()))
    #
    # plot_data.plot(kind='line')
    # plt.show()

    # Generate coincidence plot
    # Assign season to every timestep in results
    calendar = analysis.input.Calendar()
    results.output_vector['timestep'] = results.output_vector.index
    results.output_vector['season'] = results.output_vector['timestep'].apply(calendar.get_season)
    results.output_vector.drop(columns=['timestep'])

    # TODO: when drawing a load profile from base population, it should not be put back
    plt_index = 0
    seasons = ['winter', 'spring', 'summer', 'fall']
    coincidences_per_season = {}
    for season in seasons:
        print(f'calculating season {season}')
        load_profiles_base_population = (
            results.output_vector.loc[results.output_vector['season'] == season, (slice(None), 'grid_electric_power')]
        )
        iter_max = len(load_profiles_base_population)
        load_profile_names = load_profiles_base_population.columns.get_level_values(0).unique().to_list()
        # TODO: should be 100 instead of num of DERs
        n_households = range(1, len(der_names)+1)
        monte_carlo_iters = range(1, 1001)
        peak_load_coincidence = pd.DataFrame(0, columns=monte_carlo_iters, index=n_households)
        for i in n_households:
            print(f'household number {i}', end='\r')
            # TODO: Draw monte_carlo_iters * n load profiles from base population (don't have enough currently)
            for j in monte_carlo_iters:
                load_profile_names_iter = random.sample(load_profile_names, k=i)
                load_profiles_iter = load_profiles_base_population.loc[:, load_profile_names_iter]
                peak_load_per_timestep = load_profiles_iter.sum(axis=1)
                peak_load_coincidence.loc[i, j] = np.max(peak_load_per_timestep) / i

        coincidences_per_season[season] = peak_load_coincidence
        plot_data = peak_load_coincidence.mean(axis=1) / peak_load_coincidence.mean().max() * 100
        plot_data.plot(kind='line', label=season, color=colors[plt_index])
        plt_index += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.ylim([0, 110])
    plt.legend(handles, labels)
    plt.show()









