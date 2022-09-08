import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mesmo

from bscs_data_interface import data_bscs, data_ev_swapping_demand_simulation
from bscs_models import bscs_wep_optimization_model
import numpy as np

def main():

    # regulation signal time step
    reg_time_constant = 0.02
    # Settings.
    scenario_name = 'bscs_modelling'
    mesmo.data_interface.recreate_database()

    # get time steps for market
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)
    time_step = der_model_set.timesteps

    # EV swapping demand simulation
    data_set_swapping_demand = data_ev_swapping_demand_simulation(time_step)

    # Obtain data.
    data_set = data_bscs(os.path.join(os.path.dirname(os.path.normpath(__file__)), 'Dataset'))

    # plot reg D signals
    samples_to_plot = data_set.reg_d_data_40min_sample.iloc[0:-1]
    fig = px.line(samples_to_plot['RegDTest'], labels=dict(x="time step (0.2s)", value="CRS", variable="Day index"))
    fig.show()

    dfs = {"day_1_CRS": data_set.reg_d_data_whole_day[pd.datetime(2020, 1, 1, 0, 0)].cumsum().values * reg_time_constant}

    for i in range(1, 11):
        dfs.update({"day_{}_CRS".format(i): data_set.reg_d_data_whole_day[pd.datetime(2020, 1, i, 0, 0)].cumsum().values*reg_time_constant})

    dfs = pd.DataFrame(dfs)

    # plot the data
    fig = go.Figure()

    fig = px.line(dfs, x=dfs.index.values, y=["day_1_CRS", "day_2_CRS", "day_3_CRS", "day_4_CRS", "day_5_CRS",
                                              "day_6_CRS", "day_7_CRS", "day_8_CRS", "day_9_CRS", "day_10_CRS"],
                  labels=dict(x="time step (0.2s)", value="CRS", variable="Day index"))
    fig.show()

    # Get results path.
    optimal_sizing_problem = bscs_wep_optimization_model(scenario_name, data_set)

    results_path = mesmo.utils.get_results_path(__file__, scenario_name)




if __name__ == '__main__':
    main()
