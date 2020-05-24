"""Configuration module."""

import logging
import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd
import typing
import yaml


def get_config() -> dict:
    """Load the configuration dictionary.

    - Default configuration is obtained from `./fledge/config_default.yml`.
    - Custom configuration is obtained from `./config.yml` and overwrites the respective default configuration.
    - `./` denotes the repository base directory.
    """

    # Load default configuration values.
    with open(os.path.join(base_path, 'fledge', 'config_default.yml'), 'r') as file:
        default_config = yaml.safe_load(file)

    # Create local `config.yml` for custom configuration in base directory, if not existing.
    # - The additional data paths setting is added for reference.
    if not os.path.isfile(os.path.join(base_path, 'config.yml')):
        with open(os.path.join(base_path, 'config.yml'), 'w') as file:
            file.write(
                "# Local configuration values.\n"
                "# - Default values can be found in `fledge/config_default.yml`\n"
                "paths:\n"
                "  additional_data: []\n"
            )

    # Load custom configuration values, overwriting the default values.
    with open(os.path.join(base_path, 'config.yml'), 'r') as file:
        custom_config = yaml.safe_load(file)

    # Define utility function to recursively merge default and custom configuration.
    def merge_config(default_values: dict, custom_values: dict) -> dict:
        full_values = default_values.copy()
        full_values.update({
            key: (
                merge_config(default_values[key], custom_values[key])
                if (
                    (key in default_values)
                    and isinstance(default_values[key], dict)
                    and isinstance(custom_values[key], dict)
                )
                else custom_values[key]
            )
            for key in custom_values.keys()
        })
        return full_values

    # Obtain complete configuration.
    if custom_config is not None:
        complete_config = merge_config(default_config, custom_config)
    else:
        complete_config = default_config

    # Define utility function to obtain full paths.
    # - Replace `./` with the base path and normalize paths.
    def get_full_path(path: str) -> str:
        return os.path.normpath(path.replace('./', base_path + os.path.sep))

    # Obtain full paths.
    complete_config['paths']['data'] = get_full_path(complete_config['paths']['data'])
    complete_config['paths']['additional_data'] = (
        [get_full_path(path) for path in complete_config['paths']['additional_data']]
    )
    complete_config['paths']['database'] = get_full_path(complete_config['paths']['database'])
    complete_config['paths']['results'] = get_full_path(complete_config['paths']['results'])

    # If not running as main process, set `run_parallel` to False.
    # - Workaround to avoid that subprocesses / workers infinitely spawn further subprocesses / workers.
    if multiprocessing.current_process().name != 'MainProcess':
        complete_config['multiprocessing']['run_parallel'] = False

    return complete_config


def get_logger(
        name: str
) -> logging.Logger:
    """Generate logger with given name."""

    logger = logging.getLogger(name)

    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(logging.Formatter(config['logs']['format']))
    logger.addHandler(logging_handler)

    if config['logs']['level'] == 'debug':
        logger.setLevel(logging.DEBUG)
    elif config['logs']['level'] == 'info':
        logger.setLevel(logging.INFO)
    elif config['logs']['level'] == 'warn':
        logger.setLevel(logging.WARN)
    elif config['logs']['level'] == 'error':
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError(f"Unknown logging level: {config['logs']['level']}")

    return logger


# Obtain repository base directory path.
base_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))

# Obtain configuration dictionary.
config = get_config()

# Physical constants.
# TODO: Move physical constants to model definition.
water_density = 998.31  # [kg/m^3]
water_kinematic_viscosity = 1.3504e-6  # [m^2/s]
gravitational_acceleration = 9.81  # [m^2/s]

# Setup multiprocessing / parallel computing pool.
# - Number of parallel processes defaults to number of CPU threads as returned by `os.cpu_count()`.
if config['multiprocessing']['run_parallel']:
    parallel_pool = multiprocessing.Pool()

# Modify matplotlib default settings.
plt.style.use(config['plots']['matplotlib_style'])
pd.plotting.register_matplotlib_converters()  # Remove warning when plotting with pandas.

# Modify pandas default settings.
# - These settings ensure that that data frames are always printed in full, rather than cropped.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
try:
    pd.set_option('display.max_colwidth', None)
except ValueError:
    # For compatibility with older versions of pandas.
    pd.set_option('display.max_colwidth', 0)
