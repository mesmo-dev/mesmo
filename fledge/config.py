"""Configuration."""

import datetime
import logging
import matplotlib.pyplot as plt
import os

# Path definitions.
fledge_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
data_path = os.path.join(fledge_path, 'data')
database_path = os.path.join(data_path, 'database.sqlite')
# database_path = 'file:database?mode=memory&cache=shared'
results_path = os.path.join(fledge_path, 'results')

# Generate timestamp (for saving results with timestamp).
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Optimization solver settings.
solver_name = 'glpk'  # Must be valid input string for Pyomo's `SolverFactory`.
solver_output = False  # If True, activate verbose solver output.

# Plotting settings.
plt.style.use('seaborn-colorblind')

# Test settings.
test_scenario_name = 'singapore_6node'
test_data_path = os.path.join(fledge_path, 'tests', 'data')
test_plots = True  # If True, tests may produce plots.

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
