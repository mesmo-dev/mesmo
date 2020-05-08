"""Configuration module."""

import datetime
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd


# Obtain repository base path.
base_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))

# Instantiate config dictionary.
# TODO: Enable input of config parameters via config file. Therefore already structured as dictionary.
config = dict()

# Path settings.
config['paths'] = dict()
config['paths']['data'] = os.path.join(base_path, 'data')
config['paths']['additional_data'] = []
config['paths']['database'] = os.path.join(base_path, 'data', 'database.sqlite')
# config['paths']['database'] = 'file:database?mode=memory&cache=shared'
config['paths']['results'] = os.path.join(base_path, 'results')

# Optimization settings.
config['optimization'] = dict()
config['optimization']['solver_name'] = 'gurobi'  # Must be valid input string for Pyomo's `SolverFactory`.
config['optimization']['show_solver_output'] = True  # If True, activate verbose solver output.

# Test settings.
config['testing'] = dict()
config['testing']['scenario_name'] = 'singapore_6node'  # Defines scenario which is considered in tests.
config['testing']['show_plots'] = True  # If True, tests may produce plots.

# Logging settings.
config['logging'] = dict()
config['logging']['level'] = 'debug'  # Choices: `debug`, `info`, `warn`, `error`.

# Plotting settings.
config['plotting'] = dict()
config['plotting']['matplotlib_style'] = 'seaborn-colorblind'

# Physical constants.
# TODO: Move physical constants to model definition.
water_density = 998.31  # [kg/m^3]
water_kinematic_viscosity = 1.3504e-6  # [m^2/s]
gravitational_acceleration = 9.81  # [m^2/s]


# Modify matplotlib settings.
plt.style.use(config['plotting']['matplotlib_style'])
pd.plotting.register_matplotlib_converters()  # Remove warning when plotting with pandas.

# Modify pandas settings.
# - These settings ensure that that data frames are always printed in full, rather than cropped.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
try:
    pd.set_option('display.max_colwidth', None)
except ValueError:
    # For compatibility with older versions of pandas.
    pd.set_option('display.max_colwidth', 0)


def get_logger(
        name: str
) -> logging.Logger:
    """Generate logger with given name."""

    logger = logging.getLogger(name)

    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(message)s'))
    # logging_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
    logger.addHandler(logging_handler)

    if config['logging']['level'] == 'debug':
        logger.setLevel(logging.DEBUG)
    elif config['logging']['level'] == 'info':
        logger.setLevel(logging.INFO)
    elif config['logging']['level'] == 'warn':
        logger.setLevel(logging.WARN)
    elif config['logging']['level'] == 'error':
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError(f"Unknown logging level: {config['logging']['level']}")

    return logger


def get_timestamp(
        time: datetime.datetime = None
) -> str:
    """Generate formatted timestamp string, e.g., for saving results with timestamp."""

    if time is None:
        time = datetime.datetime.now()

    return time.strftime('%Y-%m-%d_%H-%M-%S')
