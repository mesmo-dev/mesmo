"""Database interface."""

import glob
from multimethod import multimethod
import numpy as np
import os
import pandas as pd
import sqlite3
import typing

import cobmo.database_interface
import fledge.config

logger = fledge.config.get_logger(__name__)


def recreate_database(
        additional_data_paths: typing.List[str] = fledge.config.config['paths']['additional_data']
) -> None:
    """Recreate SQLITE database from SQL schema file and CSV files in the data path / additional data paths."""

    # Connect SQLITE database (creates file, if none).
    database_connection = sqlite3.connect(fledge.config.config['paths']['database'])
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

    # Recreate SQLITE database schema from SQL schema file.
    with open(os.path.join(fledge.config.base_path, 'fledge', 'data_schema.sql'), 'r') as database_schema_file:
        cursor.executescript(database_schema_file.read())
    database_connection.commit()

    # Import CSV files into SQLITE database.
    # - Import only from data path, if no additional data paths are specified.
    data_paths = (
        [fledge.config.config['paths']['data']] + additional_data_paths
        if additional_data_paths is not None
        else [fledge.config.config['paths']['data']]
    )
    for data_path in data_paths:
        for csv_file in glob.glob(os.path.join(data_path, '**', '*.csv'), recursive=True):

            # Exclude CSV files from CoBMo data folders.
            if os.path.join('data', 'cobmo_data') not in csv_file:

                # Obtain table name.
                table_name = os.path.splitext(os.path.basename(csv_file))[0]

                # Write new table content.
                logger.debug(f"Loading {csv_file} into database.")
                table = pd.read_csv(csv_file)
                table.to_sql(
                    table_name,
                    con=database_connection,
                    if_exists='append',
                    index=False
                )

    cursor.close()
    database_connection.close()

    # Recreate CoBMo database to include FLEDGE's CoBMo definitions.
    # TODO: Modify CoBMo config instead.
    cobmo.database_interface.recreate_database(
        additional_data_paths=[os.path.join(fledge.config.config['paths']['data'], 'cobmo_data')]
    )


def connect_database() -> sqlite3.Connection:
    """Connect to the database and return connection handle."""

    # Recreate database, if no database exists.
    if not os.path.isfile(fledge.config.config['paths']['database']):
        logger.debug(f"Database does not exist and is recreated at: {fledge.config.config['paths']['database']}")
        recreate_database()

    # Obtain connection handle.
    database_connection = sqlite3.connect(fledge.config.config['paths']['database'])
    return database_connection


class ScenarioData(object):
    """Scenario data object."""

    scenario: pd.Series
    timesteps: pd.Index
    parameters: pd.Series

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):

        # Obtain parameters.
        self.parameters = (
            pd.read_sql(
                """
                SELECT * FROM parameters
                JOIN scenarios USING (parameter_set)
                WHERE scenario_name = ?
                """,
                con=database_connection,
                params=[scenario_name],
                index_col='parameter_name'
            ).loc[:, 'parameter_value']
        )

        # Obtain scenario data.
        self.scenario = (
            self.parse_parameters_dataframe(pd.read_sql(
                """
                SELECT * FROM scenarios
                LEFT JOIN electric_grid_operation_limit_types USING (electric_grid_operation_limit_type)
                LEFT JOIN thermal_grid_operation_limit_types USING (thermal_grid_operation_limit_type)
                WHERE scenario_name = ?
                """,
                con=database_connection,
                params=[scenario_name]
            )).iloc[0]
        )

        # Parse time definitions.
        self.scenario['timestep_start'] = (
            pd.Timestamp(self.scenario['timestep_start'])
        )
        self.scenario['timestep_end'] = (
            pd.Timestamp(self.scenario['timestep_end'])
        )
        self.scenario['timestep_interval'] = (
            pd.Timedelta(self.scenario['timestep_interval'])
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

    def parse_parameters_column(
            self,
            column: np.ndarray
    ):
        """Parse parameters into one column of a dataframe.
        - Replace strings that match `parameter_name` with `parameter_value`.
        - Other strings are are directly parsed into numbers.
        - If a string doesn't match any match `parameter_name` and cannot be parsed, it is replaced with NaN.
        - Expects `column` to be passed as `np.ndarray` rather than directly as `pd.Series` (for performance reasons).
        """

        if column.dtype == object:  # `object` represents string type.
            if any(np.isin(column, self.parameters.index)):
                column_values = (
                    self.parameters.reindex(column).values
                )
                column_values[pd.isnull(column_values)] = (
                    pd.to_numeric(column[pd.isnull(column_values)])
                )
                column = column_values
            else:
                column = pd.to_numeric(column)

        # Explicitly parse to float, for consistent behavior independent of specific values.
        column = column.astype(np.float)

        return column

    def parse_parameters_dataframe(
            self,
            dataframe: pd.DataFrame,
            excluded_columns: list = None
    ):
        """Parse parameters into a dataframe.
        - Applies `parse_parameters_column` for all string columns.
        - Columns in `excluded_columns` are not parsed. By default this includes `_name`, `_type`, `connection` columns.
        """

        # Define excluded columns. By default, all columns containing the following strings are excluded:
        # `_name`, `_type`, `connection`
        if excluded_columns is None:
            excluded_columns = []
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_name')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('parameter_set')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_type')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('connection')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('timestep')])

        # Select non-excluded, string columns and apply `parse_parameters_column`.
        selected_columns = (
            dataframe.columns[
                ~dataframe.columns.isin(excluded_columns)
                & (dataframe.dtypes == object)  # `object` represents string type.
            ]
        )
        for column in selected_columns:
            dataframe[column] = self.parse_parameters_column(dataframe[column].values)

        return dataframe


class ElectricGridData(object):
    """Electric grid data object."""

    scenario_data: ScenarioData
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

        # Obtain scenario data.
        self.scenario_data = ScenarioData(scenario_name)

        # Obtain electric grid data.
        self.electric_grid = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                """
                SELECT * FROM electric_grids
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )).iloc[0]
        )
        self.electric_grid_nodes = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.electric_grid_nodes.index = self.electric_grid_nodes['node_name']
        self.electric_grid_ders = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.electric_grid_ders.index = self.electric_grid_ders['der_name']
        self.electric_grid_lines = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.electric_grid_lines.index = self.electric_grid_lines['line_name']
        self.electric_grid_line_types = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.electric_grid_line_types.index = self.electric_grid_line_types['line_type']
        self.electric_grid_line_types_matrices = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.electric_grid_transformers = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.electric_grid_transformers.index = self.electric_grid_transformers['transformer_name']


class ThermalGridData(object):
    """Thermal grid data object."""

    scenario_data: ScenarioData
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

        # Obtain scenario data.
        self.scenario_data = ScenarioData(scenario_name)

        self.thermal_grid = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
            )).iloc[0]
        )
        self.thermal_grid_nodes = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                """
                SELECT * FROM thermal_grid_nodes
                WHERE thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            ))
        )
        self.thermal_grid_nodes.index = self.thermal_grid_nodes['node_name']
        self.thermal_grid_ders = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                """
                SELECT * FROM thermal_grid_ders
                WHERE thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            ))
        )
        self.thermal_grid_ders.index = self.thermal_grid_ders['der_name']
        self.thermal_grid_lines = (
            self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                """
                SELECT * FROM thermal_grid_lines
                JOIN thermal_grid_line_types USING (line_type)
                WHERE thermal_grid_name = (
                    SELECT thermal_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            ))
        )
        self.thermal_grid_lines.index = self.thermal_grid_lines['line_name']


class DERData(object):
    """DER data object."""

    scenario_data: ScenarioData
    fixed_loads: pd.DataFrame
    fixed_load_timeseries_dict: typing.Dict[str, pd.DataFrame]
    ev_chargers: pd.DataFrame
    ev_charger_timeseries_dict: typing.Dict[str, pd.DataFrame]
    flexible_loads: pd.DataFrame
    flexible_load_timeseries_dict: typing.Dict[str, pd.DataFrame]
    cooling_plants: pd.DataFrame
    fixed_generators: pd.DataFrame
    flexible_buildings: pd.DataFrame

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):
        """Load fixed load data from database for given `scenario_name`."""

        # Obtain scenario data.
        self.scenario_data = ScenarioData(scenario_name)

        # Obtain timeseries data. Shorthand for SQL commands.
        timestep_start_string = self.scenario_data.scenario.at['timestep_start'].strftime('%Y-%m-%dT%H:%M:%S')
        timestep_end_string = self.scenario_data.scenario.at['timestep_end'].strftime('%Y-%m-%dT%H:%M:%S')

        # Define utility function for DER data.
        # TODO: Move to method?
        # TODO: Add thermal grid DER type support.
        def get_der_type_data(der_type: str) -> typing.Tuple[pd.DataFrame, typing.Dict[str, pd.DataFrame]]:
            """Obtain DER data for given DER type."""

            # Check validity of DER type.
            try:
                assert der_type in ['fixed_load', 'ev_charger', 'flexible_load']
            except AssertionError:
                logger.error(f"Invalid DER type: {der_type}")
                raise

            # Obtain fixed load data.
            der_models = (
                self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                    f"""
                    SELECT * FROM {der_type}s
                    JOIN electric_grid_ders USING (model_name)
                    WHERE der_type = ?
                    AND electric_grid_name = (
                        SELECT electric_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                    """,
                    con=database_connection,
                    params=[
                        der_type,
                        scenario_name
                    ]
                ))
            )
            der_models.index = der_models['der_name']

            # Instantiate dictionary for unique `model_name`.
            der_models_unique = der_models.loc[:, ['model_name', 'definition_type']].drop_duplicates()
            der_models_unique.index = der_models_unique.loc[:, 'model_name']
            der_model_timeseries_dict = dict.fromkeys(der_models_unique.loc[:, 'model_name'])

            # Load timeseries for each `model_name`.
            for model_name in der_model_timeseries_dict:

                if 'timeseries' in der_models_unique.at[model_name, 'definition_type']:
                    der_model_timeseries_dict[model_name] = (
                        pd.read_sql(
                            f"""
                            SELECT * FROM {der_type}_timeseries
                            WHERE model_name = ?
                            AND time between ? AND ?
                            """,
                            con=database_connection,
                            params=[
                                model_name,
                                timestep_start_string,
                                timestep_end_string
                            ],
                            parse_dates=['time'],
                            index_col=['time']
                        ).reindex(
                            self.scenario_data.timesteps
                        ).interpolate(
                            'quadratic'
                        ).bfill(  # Backward fill to handle edge definition gaps.
                            limit=int(pd.to_timedelta('1h') / self.scenario_data.scenario['timestep_interval'])
                        ).ffill(  # Forward fill to handle edge definition gaps.
                            limit=int(pd.to_timedelta('1h') / self.scenario_data.scenario['timestep_interval'])
                        )
                    )

                if 'schedule' in der_models_unique.at[model_name, 'definition_type']:
                    der_model_schedule = (
                        pd.read_sql(
                            f"""
                            SELECT * FROM {der_type}_schedules
                            WHERE model_name = ?
                            """,
                            con=database_connection,
                            params=[
                                model_name
                            ],
                            index_col='time_period'
                        )
                    )

                    # Parse time period index.
                    der_model_schedule.index = np.vectorize(pd.Period)(der_model_schedule.index)

                    # Obtain complete schedule for all weekdays.
                    # TODO: Check if '01T00:00:00' is defined for each schedule.
                    der_model_schedule_complete = []
                    for day in range(1, 8):
                        if day in der_model_schedule.index.day.unique():
                            der_model_schedule_complete.append(
                                der_model_schedule.loc[der_model_schedule.index.day == day, :]
                            )
                        else:
                            der_model_schedule_previous = der_model_schedule_complete[-1].copy()
                            der_model_schedule_previous.index += pd.Timedelta('1 day')
                            der_model_schedule_complete.append(der_model_schedule_previous)
                    der_model_schedule_complete = pd.concat(der_model_schedule_complete)

                    # Obtain complete schedule for each minute of the week.
                    der_model_schedule_complete = (
                        der_model_schedule_complete.reindex(
                            pd.period_range(start='01T00:00', end='07T23:59', freq='T')
                        ).fillna(method='ffill')
                    )

                    # Reindex / fill internal gain schedule for given timesteps.
                    der_model_schedule_complete.index = (
                        pd.MultiIndex.from_arrays([
                            der_model_schedule_complete.index.day - 1,
                            der_model_schedule_complete.index.hour,
                            der_model_schedule_complete.index.minute
                        ])
                    )
                    der_model_schedule = (
                        pd.DataFrame(
                            index=pd.MultiIndex.from_arrays([
                                self.scenario_data.timesteps.weekday,
                                self.scenario_data.timesteps.hour,
                                self.scenario_data.timesteps.minute
                            ]),
                            columns=der_model_schedule_complete.columns
                        )
                    )
                    for column in der_model_schedule.columns:
                        der_model_schedule[column] = (
                            der_model_schedule[column].fillna(der_model_schedule_complete[column])
                        )
                    der_model_schedule.index = self.scenario_data.timesteps

                    der_model_timeseries_dict[model_name] = der_model_schedule

            return (
                der_models,
                der_model_timeseries_dict
            )

        # Obtain fixed load / EV charger / flexible load data.
        self.fixed_loads, self.fixed_load_timeseries_dict = get_der_type_data('fixed_load')
        self.ev_chargers, self.ev_charger_timeseries_dict = get_der_type_data('ev_charger')
        self.flexible_loads, self.flexible_load_timeseries_dict = get_der_type_data('flexible_load')

        # Obtain cooling plant data.
        # - Obtain DERs for electric grid / thermal grid separately and perform full outer join via `pandas.merge()`,
        #   due to SQLITE missing full outer join syntax.
        self.cooling_plants = (
            pd.merge(
                self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                    """
                    SELECT * FROM electric_grid_ders
                    WHERE der_type = 'cooling_plant'
                    AND electric_grid_name = (
                        SELECT electric_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                    """,
                    con=database_connection,
                    params=[scenario_name]
                )),
                self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                    """
                    SELECT * FROM thermal_grid_ders
                    JOIN cooling_plants USING (model_name)
                    WHERE der_type = 'cooling_plant'
                    AND thermal_grid_name = (
                        SELECT thermal_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                    """,
                    con=database_connection,
                    params=[scenario_name]
                )),
                how='outer',
                on=['der_name', 'der_type', 'model_name'],
                suffixes=('_electric_grid', '_thermal_grid')
            )
        )
        self.cooling_plants.index = self.cooling_plants['der_name']

        # Obtain fixed generator data.
        # - Obtain DERs for electric grid / thermal grid separately and perform full outer join via `pandas.merge()`,
        #   due to SQLITE missing full outer join syntax.
        self.fixed_generators = (
            pd.merge(
                self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                    """
                    SELECT * FROM electric_grid_ders
                    WHERE der_type = 'fixed_generator'
                    AND electric_grid_name = (
                        SELECT electric_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                    """,
                    con=database_connection,
                    params=[scenario_name]
                )),
                self.scenario_data.parse_parameters_dataframe(pd.read_sql(
                    """
                    SELECT * FROM thermal_grid_ders
                    WHERE der_type = 'fixed_generator'
                    AND thermal_grid_name = (
                        SELECT thermal_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                    """,
                    con=database_connection,
                    params=[scenario_name]
                )),
                how='outer',
                on=['der_name', 'der_type', 'model_name'],
                suffixes=('_electric_grid', '_thermal_grid')
            )
        )
        self.fixed_generators.index = self.fixed_generators['der_name']

        # Obtain flexible building data.
        # - Obtain DERs for electric grid / thermal grid separately and perform full outer join via `pandas.merge()`,
        #   due to SQLITE missing full outer join syntax.
        self.flexible_buildings = (
            pd.merge(
                self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
                )),
                self.scenario_data.parse_parameters_dataframe(pd.read_sql(
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
                )),
                how='outer',
                on=['der_name', 'der_type', 'model_name'],
                suffixes=('_electric_grid', '_thermal_grid')
            )
        )
        self.flexible_buildings.index = self.flexible_buildings['der_name']


class PriceData(object):
    """Price data object."""

    scenario_data: ScenarioData
    price_timeseries_dict: dict

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database()
    ):

        # Obtain scenario data.
        self.scenario_data = ScenarioData(scenario_name)

        # Instantiate dictionary for unique `price_type`.
        price_types = (
            pd.read_sql(
                """
                SELECT DISTINCT price_type FROM price_timeseries
                """,
                con=database_connection,
            )
        )
        self.price_timeseries_dict = dict.fromkeys(price_types.values.flatten())

        # Load timeseries for each `price_type`.
        # TODO: Resample / interpolate timeseries depending on timestep interval.
        for price_type in self.price_timeseries_dict:
            self.price_timeseries_dict[price_type] = (
                pd.read_sql(
                    """
                    SELECT * FROM price_timeseries
                    WHERE price_type = ?
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
                        price_type,
                        scenario_name,
                        scenario_name
                    ],
                    parse_dates=['time'],
                    index_col=['time']
                ).reindex(
                    self.scenario_data.timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(  # Backward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / self.scenario_data.scenario['timestep_interval'])
                ).ffill(  # Forward fill to handle edge definition gaps.
                    limit=int(pd.to_timedelta('1h') / self.scenario_data.scenario['timestep_interval'])
                )
            )


class ResultsDict(typing.Dict[str, pd.DataFrame]):
    """Results dictionary object, i.e., a modified dictionary object with strings as keys and dataframes as values.

    - When printed or represented as string, all dataframes are printed in full.
    - Provides a method for storing all results dataframes to CSV files.
    """

    def __repr__(self) -> str:
        """Obtain string representation of results."""

        repr_string = ""
        for key in self:
            repr_string += f"{key} = \n{self[key]}\n"

        return repr_string

    def to_csv(self, path: str) -> None:
        """Store results to CSV files at given `path`."""

        for key in self:
            self[key].to_csv(os.path.join(path, f'{key}.csv'))
