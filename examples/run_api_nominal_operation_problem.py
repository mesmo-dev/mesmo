"""Example script for setting up and solving an nominal operation problem.

- The nominal operation problem (alias: power flow problem, electric grid simulation problem)
  formulates the steady-state power flow problem for all timesteps of the given scenario
  subject to the nominal operation schedule of all DERs.
"""
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

pd.options.plotting.backend = "matplotlib"
import mesmo


def main():

    # Settings.
    scenario_name = 'ieee_34node'

    # High-level API call.
    nominal_operation = mesmo.api.run_nominal_operation_problem(scenario_name, store_results=False)
    branch_power_1 = nominal_operation.branch_power_magnitude_vector_1_per_unit
    branch_power_2 = nominal_operation.branch_power_magnitude_vector_2_per_unit
    voltage = nominal_operation.node_voltage_magnitude_vector_per_unit



    print(3)
if __name__ == '__main__':
    main()
