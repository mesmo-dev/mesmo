import mesmo
import pandas as pd
import plotly.express as px
import numpy as np
import os


def main():
    # Settings.

    scenario_name = "uncertainty_sensitivity_analysis"

    # Settings.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()
    price_data = mesmo.data_interface.PriceData(scenario_name)
    wholesale_electricity_price_2017 = price_data.price_timeseries_raw
    wholesale_electricity_price_2017.to_csv(str(os.getcwd())+'/electricity_price_data_2017')

    fig = px.line(wholesale_electricity_price_2017)
    fig.show()
    # TODO: neural network for price prediction
    #print("TensorFlow version:", tf.__version__)




if __name__ == "__main__":
    main()