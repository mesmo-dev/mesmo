"""Configuration."""

import datetime
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd

# Path definitions.
fledge_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
data_path = os.path.join(fledge_path, 'data')
database_path = os.path.join(data_path, 'database.sqlite')
# database_path = 'file:database?mode=memory&cache=shared'
results_path = os.path.join(fledge_path, 'results')

# Generate timestamp (for saving results with timestamp).
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Optimization solver settings.
solver_name = 'gurobi'  # Must be valid input string for Pyomo's `SolverFactory`.
solver_output = True  # If True, activate verbose solver output.

# Plotting settings.
plt.style.use('seaborn-colorblind')
pd.plotting.register_matplotlib_converters()

# Test settings.
test_scenario_name = 'singapore_6node'
test_data_path = os.path.join(fledge_path, 'tests', 'data')
test_plots = True  # If True, tests may produce plots.

# Physical constants.
water_density = 998.31  # [kg/m^3]
water_kinematic_viscosity = 1.3504e-6  # [m^2/s]
gravitational_acceleration = 9.81  # [m^2/s]

# Pandas settings.
# - These settings ensure that that data frames are always printed in full, rather than cropped.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
try:
    pd.set_option('display.max_colwidth', None)
except ValueError:
    # This is provided for compatibility with older versions of Pandas.
    pd.set_option('display.max_colwidth', 0)

# Logger settings.
logging_level = logging.INFO
logging_handler = logging.StreamHandler()
logging_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(message)s'))
# logging_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))


def get_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging_handler)
    logger.setLevel(logging_level)
    return logger
