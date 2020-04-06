"""Database interface."""

import glob
from multimethod import multimethod
import os
import pandas as pd
import sqlite3
import typing

import cobmo.building_model
import fledge.config

logger = fledge.config.get_logger(__name__)


def recreate_database(
        database_path: str = fledge.config.database_path,
        database_schema_path: str = os.path.join(fledge.config.fledge_path, 'fledge', 'database_schema.sql'),
        csv_path: str = fledge.config.data_path
) -> None:
    """Recreate SQLITE database from SQL schema file and CSV files."""

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
        database_path: str = fledge.config.database_path
) -> sqlite3.Connection:
    """Connect to the database at given `data_path` and return connection handle."""

    # Recreate database, if no database exists.
    if not os.path.isfile(database_path):
        logger.debug(f"Database does not exist and is recreated at: {database_path}")
        recreate_database(
            database_path=database_path
        )

    # Obtain connection.
    database_connection = sqlite3.connect(database_path)
    return database_connection


class ScenarioData(object):
    """scenario data data object."""

    scenario: pd.Series
    timesteps: pd.Index

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load scenario data for given `scenario_name`."""

        self.scenario = (
            pd.read_sql(
                """
                SELECT * FROM scenarios
                WHERE scenario_name = ?
                """,
                con=database_connection,
                params=[scenario_name],
                parse_dates=[
                    'timestep_start',
                    'timestep_end'
                ]
            ).iloc[0]  # TODO: Check needed for redundant `scenario_name` in database?
        )
        # TODO: Refactor `timestep_interval_seconds` to `timestep_interval`.
        self.scenario['timestep_interval'] = (
             pd.to_timedelta(int(self.scenario['timestep_interval_seconds']), unit='second')
        )

        # Instantiate timestep series.
        self.timesteps = (
            pd.Index(
                pd.date_range(
                    start=self.scenario['timestep_start'],
                    end=self.scenario['timestep_end'],
                    freq=self.scenario['timestep_interval']
                ),
                name='timestep'
            )
        )


class ElectricGridData(object):
    """Electric grid data object."""

    electric_grid: pd.DataFrame
    electric_grid_nodes: pd.DataFrame
    electric_grid_ders: pd.DataFrame
    electric_grid_lines: pd.DataFrame
    electric_grid_line_types: pd.DataFrame
    electric_grid_line_types_matrices: pd.DataFrame
    electric_grid_transformers: pd.DataFrame

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load electric grid data from database for given `scenario_name`."""

        # TODO: Define indexes & convert to series where appropriate.

        self.electric_grid = (
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
            ).iloc[0]  # TODO: Check needed for redundant `electric_grid_name` in database?
        )
        self.electric_grid_nodes = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_nodes
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY node_name ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_nodes.index = self.electric_grid_nodes['node_name']
        self.electric_grid_ders = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_ders
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY der_name ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_ders.index = self.electric_grid_ders['der_name']
        self.electric_grid_lines = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_lines
                JOIN electric_grid_line_types USING (line_type)
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY line_name ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_lines.index = self.electric_grid_lines['line_name']
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
                ORDER BY line_type ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_line_types.index = self.electric_grid_line_types['line_type']
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
                JOIN electric_grid_transformer_types USING (transformer_type)
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY transformer_name ASC
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.electric_grid_transformers.index = self.electric_grid_transformers['transformer_name']


class ThermalGridData(object):
    """Thermal grid data object."""

    thermal_grid: pd.DataFrame
    thermal_grid_nodes: pd.DataFrame
    thermal_grid_ders: pd.DataFrame
    thermal_grid_lines: pd.DataFrame

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load thermal grid data from database for given `scenario_name`."""

        self.thermal_grid = (
            pd.read_sql(
                """
                SELECT * FROM thermal_grids
                JOIN thermal_grid_cooling_plant_types USING (cooling_plant_type)
                WHERE thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            ).iloc[0]  # TODO: Check needed for redundant `thermal_grid_name` in database?
        )
        self.thermal_grid_nodes = (
            pd.read_sql(
                """
                SELECT * FROM thermal_grid_nodes
                WHERE thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.thermal_grid_nodes.index = self.thermal_grid_nodes['node_name']
        self.thermal_grid_ders = (
            pd.read_sql(
                """
                SELECT * FROM thermal_grid_ders
                WHERE thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.thermal_grid_ders.index = self.thermal_grid_ders['der_name']
        self.thermal_grid_lines = (
            pd.read_sql(
                """
                SELECT * FROM thermal_grid_lines
                WHERE thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.thermal_grid_lines.index = self.thermal_grid_lines['line_name']


class DERData(object):
    """DER data object."""

    fixed_loads: pd.DataFrame
    fixed_load_timeseries_dict: typing.Dict[str, pd.DataFrame]
    ev_chargers: pd.DataFrame
    ev_charger_timeseries_dict: typing.Dict[str, pd.DataFrame]
    flexible_loads: pd.DataFrame
    flexible_load_timeseries_dict: typing.Dict[str, pd.DataFrame]
    flexible_buildings: pd.DataFrame
    flexible_building_model_dict: typing.Dict[str, cobmo.building_model.BuildingModel]

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load fixed load data from database for given `scenario_name`."""

        # Obtain scenario data.
        scenario_data = ScenarioData(scenario_name)

        self.__init__(
            scenario_data,
            database_connection=database_connection
        )

    @multimethod
    def __init__(
            self,
            scenario_data: ScenarioData,
            database_connection=connect_database()
    ):
        # Obtain shorthand for `scenario_name`.
        scenario_name = scenario_data.scenario['scenario_name']

        # Obtain fixed load data.
        self.fixed_loads = (
            pd.read_sql(
                """
                SELECT * FROM fixed_loads
                JOIN electric_grid_ders USING (model_name)
                WHERE der_type = 'fixed_load'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[
                    scenario_name
                ]
            )
        )
        self.fixed_loads.index = self.fixed_loads['der_name']

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
                    parse_dates=['time'],
                    index_col=['time']
                ).reindex(
                    scenario_data.timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(  # Backward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                ).ffill(  # Forward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                )
            )

        # Obtain EV charger data.
        self.ev_chargers = (
            pd.read_sql(
                """
                SELECT * FROM ev_chargers
                JOIN electric_grid_ders USING (model_name)
                WHERE der_type = 'ev_charger'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[
                    scenario_name
                ]
            )
        )
        self.ev_chargers.index = self.ev_chargers['der_name']

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
                    parse_dates=['time'],
                    index_col=['time']
                ).reindex(
                    scenario_data.timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(  # Backward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                ).ffill(  # Forward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                )
            )

        # Obtain flexible load data.
        self.flexible_loads = (
            pd.read_sql(
                """
                SELECT * FROM flexible_loads
                JOIN electric_grid_ders USING (model_name)
                WHERE der_type = 'flexible_load'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[
                    scenario_name
                ]
            )
        )
        self.flexible_loads.index = self.flexible_loads['der_name']

        # Instantiate dictionary for unique `timeseries_name`.
        self.flexible_load_timeseries_dict = dict.fromkeys(self.flexible_loads['timeseries_name'].unique())

        # Load timeseries for each `timeseries_name`.
        # TODO: Resample / interpolate timeseries depending on timestep interval.
        for timeseries_name in self.flexible_load_timeseries_dict:
            self.flexible_load_timeseries_dict[timeseries_name] = (
                pd.read_sql(
                    """
                    SELECT * FROM flexible_load_timeseries
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
                    parse_dates=['time'],
                    index_col=['time']
                ).reindex(
                    scenario_data.timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(  # Backward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                ).ffill(  # Forward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                )
            )

        # Obtain flexible building data.
        # - Obtain DERs for electric grid / thermal grid separately and perform full outer join via `pandas.merge()`,
        #   due to SQLITE missing full outer join syntax.
        flexible_buildings_electric_grid = (
            pd.read_sql(
                """
                SELECT * FROM electric_grid_ders
                WHERE der_type = 'flexible_building'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        flexible_buildings_thermal_grid = (
            pd.read_sql(
                """
                SELECT * FROM thermal_grid_ders
                WHERE der_type = 'flexible_building'
                AND thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.flexible_buildings = (
            pd.merge(
                flexible_buildings_electric_grid,
                flexible_buildings_thermal_grid,
                how='outer',
                on=['der_name', 'der_type', 'model_name'],
                suffixes=('_electric_grid', '_thermal_grid')
            )
        )
        self.flexible_buildings.index = self.flexible_buildings['der_name']

        # Instantiate dictionary for unique `timeseries_name`.
        self.flexible_building_model_dict = dict.fromkeys(self.flexible_buildings['model_name'].unique())

        # Obtain flexible building model.
        for model_name in self.flexible_building_model_dict:
            self.flexible_building_model_dict[model_name] = (
                cobmo.building_model.BuildingModel(
                    model_name,
                    timestep_start=scenario_data.scenario['timestep_start'],
                    timestep_end=scenario_data.scenario['timestep_end'],
                    timestep_delta=scenario_data.scenario['timestep_interval'],
                    connect_electric_grid=True,
                    connect_thermal_grid_cooling=True
                )
            )


class PriceData(object):
    """Price data object."""

    price_timeseries_dict: dict

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load price data object from database for a given `scenario_name`."""

        # Obtain scenario data.
        scenario_data = ScenarioData(scenario_name)

        self.__init__(
            scenario_data,
            database_connection=database_connection
        )

    @multimethod
    def __init__(
            self,
            scenario_data: ScenarioData,
            database_connection=connect_database()
    ):
        """Load price data object from database for a given `scenario_data`."""

        # Obtain shorthand for `scenario_name`.
        scenario_name = scenario_data.scenario['scenario_name']

        # Instantiate dictionary for unique `price_name`.
        price_names = (
            pd.read_sql(
                """
                SELECT DISTINCT price_name FROM price_timeseries
                """,
                con=database_connection,
            )
        )
        self.price_timeseries_dict = dict.fromkeys(price_names.values.flatten())

        # Load timeseries for each `price_name`.
        # TODO: Resample / interpolate timeseries depending on timestep interval.
        for price_name in self.price_timeseries_dict:
            self.price_timeseries_dict[price_name] = (
                pd.read_sql(
                    """
                    SELECT * FROM price_timeseries
                    WHERE price_name = ?
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
                        price_name,
                        scenario_name,
                        scenario_name
                    ],
                    parse_dates=['time'],
                    index_col=['time']
                ).reindex(
                    scenario_data.timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(  # Backward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                ).ffill(  # Forward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / scenario_data.scenario['timestep_interval'])
                )
            )
