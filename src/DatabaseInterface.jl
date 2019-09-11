"Database interface."
module DatabaseInterface

include("config.jl")

import CSV
import DataFrames
import SQLite

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
                CSV.read(joinpath(csv_path, file_name)),
                database_connection,
                file_name[1:end-4]
            )
        end
    end
end

"""
Electric grid data object.

- Gets electric grid data from database
- Data is stored in pandas dataframes.
"""
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
function ElectricGridData(scenario_name::String)
    database_connection = DatabaseInterface.connect_database()

    electric_grids = DataFrames.DataFrame(SQLite.Query(
        database_connection,
        """
        SELECT * FROM electric_grids
        WHERE electric_grid_name = (
            SELECT electric_grid_name FROM scenarios
            WHERE scenario_name = ?
        )
        """;
        values=[scenario_name]
    ))
    electric_grid_nodes = DataFrames.DataFrame(SQLite.Query(
        database_connection,
        """
        SELECT * FROM electric_grid_nodes
        WHERE electric_grid_name = (
            SELECT electric_grid_name FROM scenarios
            WHERE scenario_name = ?
        )
        """;
        values=[scenario_name]
    ))
    electric_grid_loads = DataFrames.DataFrame(SQLite.Query(
        database_connection,
        """
        SELECT * FROM electric_grid_loads
        WHERE electric_grid_name = (
            SELECT electric_grid_name FROM scenarios
            WHERE scenario_name = ?
        )
        """;
        values=[scenario_name]
    ))
    electric_grid_lines = DataFrames.DataFrame(SQLite.Query(
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
    ))
    electric_grid_line_types = DataFrames.DataFrame(SQLite.Query(
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
    ))
    electric_grid_line_types_matrices = DataFrames.DataFrame(SQLite.Query(
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
    ))
    electric_grid_transformers = DataFrames.DataFrame(SQLite.Query(
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
    ))
    electric_grid_transformer_reactances = DataFrames.DataFrame(SQLite.Query(
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
    ))
    electric_grid_transformer_taps = DataFrames.DataFrame(SQLite.Query(
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
    ))

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
    fixed_load_timeseries::DataFrames.DataFrame
end

"Load fixed load data from database for give scenario name."
function FixedLoadData(scenario_name::String)
    database_connection = DatabaseInterface.connect_database()

    fixed_loads = DataFrames.DataFrame(SQLite.Query(
        database_connection,
        """
        SELECT * FROM fixed_loads
        WHERE model_name IN (
            SELECT DISTINCT model_name FROM electric_grid_loads
            WHERE model_type = 'fixed_load'
            AND electric_grid_name = (
				SELECT electric_grid_name FROM scenarios
				WHERE scenario_name = ?
			)
        )
        """;
        values=[scenario_name]
    ))
    fixed_load_timeseries = DataFrames.DataFrame(SQLite.Query(
        database_connection,
        """
        SELECT * FROM fixed_load_timeseries
        WHERE timeseries_name IN (
            SELECT DISTINCT timeseries_name FROM fixed_loads
            WHERE model_name IN (
                SELECT DISTINCT model_name FROM electric_grid_loads
                WHERE model_type = 'fixed_load'
                AND electric_grid_name = (
                    SELECT electric_grid_name FROM scenarios
                    WHERE scenario_name = ?
                )
            )
        )
        """;
        values=[scenario_name]
    ))

    FixedLoadData(
        fixed_loads,
        fixed_load_timeseries
    )
end

end
