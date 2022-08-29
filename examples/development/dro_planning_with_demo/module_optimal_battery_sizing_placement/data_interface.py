"""DRO data interface."""

import numpy as np
import os
import pandas as pd

import mesmo


class data_battery_sizing_placement(object):

    energy_price_annual_average: pd.DataFrame
    battery_base_data: pd.DataFrame

    def __init__(
            self,
            data_path: str,
    ):

        self.price_raw = pd.read_csv(os.path.join(data_path, 'typical_market_price_curve.csv'))

        self.annual_average_energy_price = self.price_raw['USEP($/MWh)'] / 100

        self.battery_data = pd.read_csv(os.path.join(data_path, 'battery_cell_base_data.csv'))



def main():
    test = data_battery_sizing_placement(os.path.join(os.path.dirname(os.path.normpath(__file__)),'test_case_customized'))
    print('pause')

if __name__ == '__main__':
    main()
