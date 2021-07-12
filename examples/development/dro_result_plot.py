"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dro_data_interface import DRO_data, DRO_ambiguity_set
import fledge
import statistics
import glob

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def main():
    scenario_name = 'singapore_6node_custom'

    energy_dro = pd.read_csv(
             "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\dro_std\\energy_dro.csv")

    energy_det = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\dro_std\\energy_det.csv")

    up_reserve_det = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\dro_std\\up_reserve_det.csv")

    up_reserve_dro = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\dro_std\\up_reserve_dro.csv")

    down_reserve_det = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\dro_std\\down_reserve_det.csv")

    down_reserve_dro = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\dro_std\\down_reserve_dro.csv")

    print()

    number_of_time_steps = len(energy_dro['timestep'])
    X = np.arange(number_of_time_steps)

    # offer figure
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.bar(X + 0.00, energy_dro['total'], color='blue', width=0.25)
    ax.bar(X + 0.00, up_reserve_dro['total'], bottom=energy_dro['total'], color='r', width=0.25)
    ax.bar(X + 0.00, down_reserve_dro['total'], bottom=up_reserve_dro['total'], color='y', width=0.25)

    ax.bar(X + 0.25, energy_det['total'], color='royalblue', width=0.25)
    ax.bar(X + 0.25, up_reserve_det['total'], bottom=energy_det['total'], color='indianred', width=0.25)
    ax.bar(X + 0.25, down_reserve_det['total'], bottom=up_reserve_det['total'], color='gold', width=0.25)

    ax.legend(labels=['Energy - DRO', 'Up reserve - DRO', 'Down reserve - DRO', 'Energy - det.', 'Up reserve - det.', 'Down reserve - det.'], loc='right')

    ax.set_ylabel('Offers kWh')
    ax.set_xlabel('time step')
    ax.set_title('Comparison of DRO solution and deterministic solution')
    ax.set_xticks(X)
    ax.set_xticklabels(energy_dro['timestep'].values.tolist())
    plt.grid()
    fig.savefig('C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\final_plots\\offer_result.pdf')
    plt.show()


    dro_data_set = DRO_data("C:\\Users\\kai.zhang\\Desktop\\local_fledge_data\\dro_data\\")

    price_timeseries_energy = dro_data_set.energy_price[0:4].to_numpy()
    price_timeseries_reserve = dro_data_set.contingency_reserve_price[0:4].to_numpy()

    # price figure
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.bar(X + 0.00, price_timeseries_energy, color='blue', width=0.25)

    ax.bar(X + 0.25, price_timeseries_reserve, color='gold', width=0.25)

    ax.legend(labels=['Energy price forecast', 'Contingency reserve price forecast'], loc='right')
    plt.grid()
    ax.set_ylabel('price $/kWh')
    ax.set_xlabel('time step')
    ax.set_title('Price')
    ax.set_xticks(X)
    ax.set_xticklabels(energy_dro['timestep'].values.tolist())
    fig.savefig('C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\final_plots\\price.pdf')
    plt.show()


    fig, ax = plt.subplots()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.bar(X + 0.00, 100*price_timeseries_energy, color='blue', width=0.25)
    ax.set_xlabel("time step")
    ax.set_ylabel("energy price $/MWh")
    ax.set_xticks(X)
    ax.set_xticklabels(energy_dro['timestep'].values.tolist())
    plt.grid()
    ax.legend(labels=['Energy price forecast'], loc='right')

    ax2 = ax.twinx()
    ax2.bar(X + 0.25, 100*price_timeseries_reserve, color='gold', width=0.25)
    ax2.set_ylabel("contingency reserve price $/MWh")
    ax2.set_xticks(X)
    ax2.set_xticklabels(energy_dro['timestep'].values.tolist())
    fig.savefig('C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\final_plots\\price.pdf')
    ax2.legend(labels=['Energy price forecast', 'Contingency reserve price forecast'], loc='lower right')
    plt.show()


    so_normal = pd.DataFrame()
    for file_name in glob.glob('C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\SO_normal_distribution\\' + '*.csv'):
        x = pd.read_csv(file_name, low_memory=False)
        so_normal = pd.concat([so_normal, x], axis=0)

    so_gamma = pd.DataFrame()
    for file_name in glob.glob('C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\SO_gamma_distribution\\' + '*.csv'):
        x = pd.read_csv(file_name, low_memory=False)
        so_gamma = pd.concat([so_gamma, x], axis=0)

    dro = pd.read_csv(
        "C:\\Users\\kai.zhang\\Desktop\\fledge\\results\\dro_std\\objective_dro.csv")


    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.set_title('Default violin plot')
    ax.set_ylabel('Observed values')
    ax.violinplot([so_normal['objective_value_so'], so_gamma['objective_value_so'], dro['objective_value_dro']])


    plt.show()

    # # create test data
    # np.random.seed(19680801)
    # data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]
    #
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    #
    # ax1.set_title('Default violin plot')
    # ax1.set_ylabel('Observed values')
    # ax1.violinplot(data)
    #
    # ax2.set_title('Customized violin plot')
    # parts = ax2.violinplot(
    #         data, showmeans=False, showmedians=False,
    #         showextrema=False)
    #
    # for pc in parts['bodies']:
    #     pc.set_facecolor('#D43F3A')
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(1)
    #
    # quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    # whiskers = np.array([
    #     adjacent_values(sorted_array, q1, q3)
    #     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    #
    # inds = np.arange(1, len(medians) + 1)
    # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    #
    # # set style for the axes
    # labels = ['A', 'B', 'C', 'D']
    # for ax in [ax1, ax2]:
    #     set_axis_style(ax, labels)
    #
    # plt.subplots_adjust(bottom=0.15, wspace=0.05)
    # plt.show()

if __name__ == '__main__':
    main()
