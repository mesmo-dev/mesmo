"""BSCS modelling data interface."""
import numpy as np
import os
import pandas as pd
import glob
import mesmo

class data_bscs(object):

    reg_d_data_40min_sample: pd.DataFrame
    wholesale_electricity_price_data: pd.DataFrame
    reserve_price_data: pd.DataFrame
    regulation_price_data: pd.DataFrame
    reg_d_data_whole_day: pd.DataFrame

    def __init__(
            self,
            data_path: str,
    ):

        # RegD data
        self.reg_d_data_40min_sample = pd.read_csv(os.path.join(data_path, 'PJM_Data', 'reg_d_40min_sample.csv'))

        # RegD data wholeday
        self.reg_d_data_whole_day = pd.read_excel(os.path.join(data_path, 'PJM_Data', '2020', '01 2020.xlsx'))

        # Wholesale market price data
        all_files = glob.glob(os.path.join(data_path, 'EMC_data', 'WEP_from_01-Jan-2021_to_31-Dec-2021', "*.csv"))
        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        self.wholesale_electricity_price_data = pd.concat(li, axis=0, ignore_index=True)

        # Reserve price data
        all_files = glob.glob(os.path.join(data_path, 'EMC_data', 'MRP_from_01-Jan-2021_to_31-Dec-2021', "*.csv"))
        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        self.reserve_price_data = pd.concat(li, axis=0, ignore_index=True)

        # Regulation price data
        all_files = glob.glob(os.path.join(data_path, 'EMC_data', 'MFP_from_01-Jan-2021_to_31-Dec-2021', "*.csv"))
        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        self.regulation_price_data = pd.concat(li, axis=0, ignore_index=True)

def main():

    data_bscs = data_bscs(os.path.join(os.path.dirname(os.path.normpath(__file__)),'test_case_customized'))

    print('pause')

if __name__ == '__main__':
    main()
