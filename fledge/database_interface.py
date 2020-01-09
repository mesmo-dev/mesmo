"""Database interface."""

import glob
import os
import pandas as pd
import sqlite3

import fledge.config


def create_database(
        database_path,
        database_schema_path=os.path.join(fledge.config.fledge_path, 'fledge', 'database_schema.sql'),
        csv_path=fledge.config.data_path
):
    """Create SQLITE database from SQL schema file and CSV files."""

    # Connect SQLITE database (creates file, if none).
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

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
    conn.commit()

    # Import CSV files into SQLITE database.
    conn.text_factory = str  # Allows utf-8 data to be stored.
    cursor = conn.cursor()
    for file in glob.glob(os.path.join(csv_path, '*.csv')):
        # Obtain table name.
        table_name = os.path.splitext(os.path.basename(file))[0]

        # Delete existing table content.
        cursor.execute("DELETE FROM {}".format(table_name))
        conn.commit()

        # Write new table content.
        table = pd.read_csv(file)
        table.to_sql(
            table_name,
            con=conn,
            if_exists='append',
            index=False
        )
    cursor.close()
    conn.close()


def connect_database(
        database_path=os.path.join(fledge.config.data_path, 'database.sqlite'),
        overwrite_database=False
):
    """Connect to the database at given `data_path` and return connection handle."""

    # Create database, if `overwrite_database` or no database exists.
    if overwrite_database or not os.path.isfile(database_path):
        fledge.database_interface.create_database(
            database_path=database_path
        )

    # Obtain connection.
    database_connection = sqlite3.connect(database_path)
    return database_connection
