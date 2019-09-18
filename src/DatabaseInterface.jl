"Database interface."
module DatabaseInterface

include("config.jl")

import CSV
import DataFrames
import Dates
import SQLite
import TimeSeries

"""
Get database connection handle.

- If the SQLITE file `database_path` does not exist, database is created.
- If `overwrite_database=true`, the database is re-created from scratch.
"""
function connect_database(;
    database_path=joinpath(_config["data_path"], "database.sqlite"),
    overwrite_database=false
)
    if (
        overwrite_database
        || !isfile(database_path)
    )
        # Debugging output.
        Memento.info(
            _logger,
            "Creating / overwriting SQLITE file at `$database_path`."
        )

        create_database(database_path)
    end

    database_connection = SQLite.DB(database_path)
    return database_connection
end

"""
Create or overwrite SQLITE database from SQL schema file and CSV files.

- TODO: Investigate database locking behavior.
- TODO: Check behaviour with redundant values in CSV.
"""
function create_database(
    database_path;
    database_schema_path=(
        joinpath(_config["fledge_path"], "database_schema.sql")
    ),
    csv_path=_config["data_path"]
)
    # Connect to SQLITE database file (creates file, if none).
    database_connection = SQLite.DB(database_path)

    # Remove any existing data.
    sql_commands = [
        "PRAGMA writable_schema = 1"
        "DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')"
        "PRAGMA writable_schema = 0"
        "VACUUM"
    ]
    for sql_command in sql_commands
        SQLite.execute!(database_connection, sql_command)
    end

    # Create database schema from SQL schema file.
    open(database_schema_path) do database_schema_file
        # Read SQL commands and remove special characters.
        sql_commands = (
            split(join(split(join(split(join(
                readlines(database_schema_file)
            ),"\t"), " "), "\""), "'"), ";")
        )
        for sql_command in sql_commands
            # Execute sql commands, except if empty.
            if sql_command != ""
                SQLite.execute!(database_connection, sql_command)
            end
        end
    end

    # Import CSV files into database.
    for file_name in readdir(csv_path)
        if endswith(file_name, ".csv")
            Memento.info(_logger, "Loading $file_name into database.")
            SQLite.load!(
                CSV.read(
                    joinpath(csv_path, file_name);
                    dateformat="" # Keep date / time columns as strings.
                ),
                database_connection,
                file_name[1:end-4] # Table name.
            )
        end
    end
end

"Timestep data object."
struct TimestepData
    timestep_start::Dates.DateTime
    timestep_end::Dates.DateTime
    timestep_interval_seconds::Dates.Second
    timestep_count::Int
    timestep_vector::Array{Dates.DateTime,1}
end

"Load timestep data and generate timestep vector for given `scenario_name`."
function TimestepData(scenario_name::String)
    database_connection = DatabaseInterface.connect_database()

    scenarios = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM scenarios
                WHERE scenario_name = ?
                """;
                values=[scenario_name]
            )
        )
    )

    # Parse strings into `Dates.DateTime`.
    # - Strings must be in the format "yyyy-mm-ddTHH:MM:SS",
    #   e.g., "2017-12-31T01:30:45", for this parsing to work.
    timestep_start = Dates.DateTime(scenarios[:timestep_start][1])
    timestep_end = Dates.DateTime(scenarios[:timestep_end][1])

    # Timestep interval is converted to `Dates.Second`.
    timestep_interval_seconds = (
        Dates.Second(scenarios[:timestep_interval_seconds][1])
    )

    # Construct vector of timesteps for the scenario.
    timestep_vector = (
        Array(timestep_start:timestep_interval_seconds:timestep_end)
    )

    timestep_count = length(timestep_vector)

    TimestepData(
        timestep_start,
        timestep_end,
        timestep_interval_seconds,
        timestep_count,
        timestep_vector
    )
end

"Electric grid data object."
struct ElectricGridData
    electric_grids::DataFrames.DataFrame
    electric_grid_nodes::DataFrames.DataFrame
    electric_grid_loads::DataFrames.DataFrame
    electric_grid_lines::DataFrames.DataFrame
    electric_grid_line_types::DataFrames.DataFrame
    electric_grid_line_types_matrices::DataFrames.DataFrame
    electric_grid_transformers::DataFrames.DataFrame
    electric_grid_transformer_reactances::DataFrames.DataFrame
    electric_grid_transformer_taps::DataFrames.DataFrame
end

"Load electric grid data from database for given `scenario_name`."
function ElectricGridData(scenario_name::String)
    database_connection = DatabaseInterface.connect_database()

    electric_grids = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grids
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_nodes = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grid_nodes
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_loads = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grid_loads
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_lines = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grid_lines
                JOIN electric_grid_line_types USING (line_type)
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_line_types = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grid_line_types
                WHERE line_type IN (
                    SELECT line_type FROM electric_grid_lines
                    WHERE electric_grid_name = (
                        SELECT electric_grid_name FROM scenarios
                        WHERE scenario_name = ?
                    )
                )
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_line_types_matrices = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
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
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_transformers = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grid_transformers
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY transformer_name ASC, winding ASC
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_transformer_reactances = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grid_transformer_reactances
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY transformer_name ASC, row ASC, col ASC
                """;
                values=[scenario_name]
            )
        )
    )
    electric_grid_transformer_taps = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM electric_grid_transformer_taps
                WHERE electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                ORDER BY transformer_name ASC, winding ASC
                """;
                values=[scenario_name]
            )
        )
    )

    ElectricGridData(
        electric_grids,
        electric_grid_nodes,
        electric_grid_loads,
        electric_grid_lines,
        electric_grid_line_types,
        electric_grid_line_types_matrices,
        electric_grid_transformers,
        electric_grid_transformer_reactances,
        electric_grid_transformer_taps
    )
end

"Fixed load data object."
struct FixedLoadData
    fixed_loads::DataFrames.DataFrame
    fixed_load_timeseries_dict::Dict{String,TimeSeries.TimeArray}
end

"Load fixed load data from database for given `scenario_name`."
function FixedLoadData(scenario_name::String)
    database_connection = DatabaseInterface.connect_database()

    fixed_loads = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
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
                """;
                values=[
                    scenario_name,
                    scenario_name
                ]
            )
        )
    )

    # Instantiate dictionary for unique `timeseries_name`.
    fixed_load_timeseries_dict = (
        Dict(
            key => TimeSeries.TimeArray(Vector{Dates.DateTime}(), Array{Any}(undef, 0, 0))
            for key in unique(fixed_loads[:timeseries_name])
        )
    )

    # Load timeseries for each `timeseries_name`.
    # TODO: Resample / interpolate timeseries depending on timestep interval.
    for timeseries_name in keys(fixed_load_timeseries_dict)
        fixed_load_timeseries = (
            DataFrames.DataFrame(
                SQLite.Query(
                    database_connection,
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
                    """;
                    values=[
                        timeseries_name,
                        scenario_name,
                        scenario_name
                    ]
                )
            )
        )

        # Parse strings into `Dates.DateTime`.
        # - Strings must be in the format "yyyy-mm-ddTHH:MM:SS",
        #   e.g., "2017-12-31T01:30:45", for this parsing to work.
        fixed_load_timeseries[:time] = (
            Dates.DateTime.(fixed_load_timeseries[:time])
        )

        # Convert to `TimeSeries.TimeArray` and store into dictionary.
        fixed_load_timeseries_dict[timeseries_name] = (
            TimeSeries.TimeArray(fixed_load_timeseries; timestamp=:time)
        )
    end

    FixedLoadData(
        fixed_loads,
        fixed_load_timeseries_dict
    )
end

"EV charger data object."
struct EVChargerData
    ev_chargers::DataFrames.DataFrame
    ev_charger_timeseries_dict::Dict{String,TimeSeries.TimeArray}
end

"Load EV charger data from database for given `scenario_name`."
function EVChargerData(scenario_name::String)
    database_connection = DatabaseInterface.connect_database()

    ev_chargers = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
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
                """;
                values=[
                    scenario_name,
                    scenario_name
                ]
            )
        )
    )

    # Instantiate dictionary for unique `timeseries_name`.
    ev_charger_timeseries_dict = (
        Dict(
            key => TimeSeries.TimeArray(Vector{Dates.DateTime}(), Array{Any}(undef, 0, 0))
            for key in unique(ev_chargers[:timeseries_name])
        )
    )

    # Load timeseries for each `timeseries_name`.
    # TODO: Resample / interpolate timeseries depending on timestep interval.
    for timeseries_name in keys(ev_charger_timeseries_dict)
        ev_charger_timeseries = (
            DataFrames.DataFrame(
                SQLite.Query(
                    database_connection,
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
                    """;
                    values=[
                        timeseries_name,
                        scenario_name,
                        scenario_name
                    ]
                )
            )
        )

        # Parse strings into `Dates.DateTime`.
        # - Strings must be in the format "yyyy-mm-ddTHH:MM:SS",
        #   e.g., "2017-12-31T01:30:45", for this parsing to work.
        ev_charger_timeseries[:time] = (
            Dates.DateTime.(ev_charger_timeseries[:time])
        )

        # Convert to `TimeSeries.TimeArray` and store into dictionary.
        ev_charger_timeseries_dict[timeseries_name] = (
            TimeSeries.TimeArray(ev_charger_timeseries; timestamp=:time)
        )
    end

    EVChargerData(
        ev_chargers,
        ev_charger_timeseries_dict
    )
end

"Flexible load data object."
struct FlexibleLoadData
    flexible_loads::DataFrames.DataFrame
    flexible_load_timeseries_dict::Dict{String,TimeSeries.TimeArray}
end

"Flexible load data from database for given `scenario_name`."
function FlexibleLoadData(scenario_name::String)
    database_connection = DatabaseInterface.connect_database()

    flexible_loads = (
        DataFrames.DataFrame(
            SQLite.Query(
                database_connection,
                """
                SELECT * FROM flexible_loads
                JOIN electric_grid_loads USING (model_name)
                WHERE model_type = 'flexible_load'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
                AND case_name = (
                    SELECT case_name FROM scenarios
                    WHERE scenario_name = ?
                )
                """;
                values=[
                    scenario_name,
                    scenario_name
                ]
            )
        )
    )

    # Instantiate dictionary for unique `timeseries_name`.
    flexible_load_timeseries_dict = (
        Dict(
            key => TimeSeries.TimeArray(Vector{Dates.DateTime}(), Array{Any}(undef, 0, 0))
            for key in unique(flexible_loads[:timeseries_name])
        )
    )

    # Load timeseries for each `timeseries_name`.
    # TODO: Resample / interpolate timeseries depending on timestep interval.
    for timeseries_name in keys(flexible_load_timeseries_dict)
        flexible_load_timeseries = (
            DataFrames.DataFrame(
                SQLite.Query(
                    database_connection,
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
                    """;
                    values=[
                        timeseries_name,
                        scenario_name,
                        scenario_name
                    ]
                )
            )
        )

        # Parse strings into `Dates.DateTime`.
        # - Strings must be in the format "yyyy-mm-ddTHH:MM:SS",
        #   e.g., "2017-12-31T01:30:45", for this parsing to work.
        flexible_load_timeseries[:time] = (
            Dates.DateTime.(flexible_load_timeseries[:time])
        )

        # Convert to `TimeSeries.TimeArray` and store into dictionary.
        flexible_load_timeseries_dict[timeseries_name] = (
            TimeSeries.TimeArray(flexible_load_timeseries; timestamp=:time)
        )
    end

    FlexibleLoadData(
        flexible_loads,
        flexible_load_timeseries_dict
    )
end

end
