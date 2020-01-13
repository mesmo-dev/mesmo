"""Database interface."""

import glob
import os
import pandas as pd
import sqlite3

import fledge.config

logger = fledge.config.get_logger(__name__)


def create_database(
        database_path,
        database_schema_path=os.path.join(fledge.config.fledge_path, 'fledge', 'database_schema.sql'),
        csv_path=fledge.config.data_path
):
    """Create SQLITE database from SQL schema file and CSV files."""

    # Connect SQLITE database (creates file, if none).
    database_connection = sqlite3.connect(database_path)
    cursor = database_connection.cursor()

    # Remove old data, if any.
    cursor.executescript(
        """ 
        PRAGMA writable_schema = 1; 
        DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger'); 
        PRAGMA writable_schema = 0; 
        VACUUM; 
        """
    )

    # Recreate SQLITE database (schema) from SQL file.
    with open(database_schema_path, 'r') as database_schema_file:
        cursor.executescript(database_schema_file.read())
    database_connection.commit()

    # Import CSV files into SQLITE database.
    database_connection.text_factory = str  # Allows utf-8 data to be stored.
    cursor = database_connection.cursor()
    for file in glob.glob(os.path.join(csv_path, '*.csv')):
        # Obtain table name.
        table_name = os.path.splitext(os.path.basename(file))[0]

        # Delete existing table content.
        cursor.execute("DELETE FROM {}".format(table_name))
        database_connection.commit()

        # Write new table content.
        logger.debug(f"Loading {file} into database.")
        table = pd.read_csv(file)
        table.to_sql(
            table_name,
            con=database_connection,
            if_exists='append',
            index=False
        )
    cursor.close()
    database_connection.close()


def connect_database(
        database_path=os.path.join(fledge.config.data_path, 'database.sqlite'),
        overwrite_database=False
):
    """Connect to the database at given `data_path` and return connection handle."""

    # Create database, if `overwrite_database` or no database exists.
    if overwrite_database or not os.path.isfile(database_path):
        create_database(
            database_path=database_path
        )

    # Obtain connection.
    database_connection = sqlite3.connect(database_path)
    return database_connection


class ElectricGridData(object):
    """Electric grid data object."""

    electric_grids: pd.DataFrame
    electric_grid_nodes: pd.DataFrame
    electric_grid_loads: pd.DataFrame
    electric_grid_lines: pd.DataFrame
    electric_grid_line_types: pd.DataFrame
    electric_grid_line_types_matrices: pd.DataFrame
    electric_grid_transformers: pd.DataFrame
    electric_grid_transformer_reactances: pd.DataFrame
    electric_grid_transformer_taps: pd.DataFrame

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load electric grid data from database for given `scenario_name`."""

        # TODO: Define indexes & convert to series where appropriate.

        self.electric_grids = (
            pd.read_sql(
                """
                SELECT * FROM electric_grids
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_nodes = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_nodes
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_nodes.index = self.electric_grid_nodes['node_name']
        self.electric_grid_loads = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_loads
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_loads.index = self.electric_grid_loads['load_name']
        self.electric_grid_lines = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_lines
                JOIN electric_grid_line_types USING (line_type)
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_line_types = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_line_types
                WHERE line_type IN (
                    SELECT line_type FROM electric_grid_lines
                    WHERE electric_grid_name = (
                        SELECT electric_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_line_types_matrices = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_line_types_matrices
                WHERE line_type IN (
                    SELECT line_type FROM electric_grid_lines
                    WHERE electric_grid_name = (
                        SELECT electric_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                )
                ORDER BY line_type ASC, row ASC, col ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_transformers = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_transformers
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY transformer_name ASC, winding ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_transformer_reactances = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_transformer_reactances
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY transformer_name ASC, row ASC, col ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_transformer_taps = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_transformer_taps
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY transformer_name ASC, winding ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )


class FixedLoadData(object):
    """Fixed load data object."""

    fixed_loads: pd.DataFrame
    fixed_load_timeseries_dict: dict

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load fixed load data from database for given `scenario_name`."""

        self.fixed_loads = (
            pd.read_sql(
                """
                SELECT * FROM fixed_loads
                JOIN electric_grid_loads USING (model_name)
                WHERE model_type = 'fixed_load'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                AND case_name = (
                    SELECT case_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[
                    scenario_name,
                    scenario_name
                ]
            )
        )
        self.fixed_loads.index = self.fixed_loads['load_name']

        # Instantiate dictionary for unique `timeseries_name`.
        self.fixed_load_timeseries_dict = dict.fromkeys(self.fixed_loads['timeseries_name'].unique())

        # Load timeseries for each `timeseries_name`.
        # TODO: Resample / interpolate timeseries depending on timestep interval.
        for timeseries_name in self.fixed_load_timeseries_dict:
            self.fixed_load_timeseries_dict[timeseries_name] = (
                pd.read_sql(
                    """
                    SELECT * FROM fixed_load_timeseries
                    WHERE timeseries_name = ?
                    AND time >= (
                        SELECT timestep_start FROM scenarios
                        WHERE scenario_name = ?
                    )
                    AND time <= (
                        SELECT timestep_end FROM scenarios
                        WHERE scenario_name = ?
                    )
                    """,
                    con=database_connection,
                    params=[
                        timeseries_name,
                        scenario_name,
                        scenario_name
                    ],
                    parse_dates=['time']
                )
            )


class EVChargerData(object):
    """EV charger data object."""

    ev_chargers: pd.DataFrame
    ev_charger_timeseries_dict: dict

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load EV charger data from database for given `scenario_name`."""

        self.ev_chargers = (
            pd.read_sql(
                """
                SELECT * FROM ev_chargers
                JOIN electric_grid_loads USING (model_name)
                WHERE model_type = 'ev_charger'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                AND case_name = (
                    SELECT case_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[
                    scenario_name,
                    scenario_name
                ]
            )
        )
        self.ev_chargers.index = self.ev_chargers['load_name']

        # Instantiate dictionary for unique `timeseries_name`.
        self.ev_charger_timeseries_dict = dict.fromkeys(self.ev_chargers['timeseries_name'].unique())

        # Load timeseries for each `timeseries_name`.
        # TODO: Resample / interpolate timeseries depending on timestep interval.
        for timeseries_name in self.ev_charger_timeseries_dict:
            self.ev_charger_timeseries_dict[timeseries_name] = (
                pd.read_sql(
                    """
                    SELECT * FROM ev_charger_timeseries
                    WHERE timeseries_name = ?
                    AND time >= (
                        SELECT timestep_start FROM scenarios
                        WHERE scenario_name = ?
                    )
                    AND time <= (
                        SELECT timestep_end FROM scenarios
                        WHERE scenario_name = ?
                    )
                    """,
                    con=database_connection,
                    params=[
                        timeseries_name,
                        scenario_name,
                        scenario_name
                    ],
                    parse_dates=['time']
                )
            )
