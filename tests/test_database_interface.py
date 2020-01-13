"""Test database interface."""

import os
import sqlite3
import time
import unittest

import fledge.config
import fledge.database_interface

logger = fledge.config.get_logger(__name__)


class TestDatabaseInterface(unittest.TestCase):

    def test_create_database(self):
        # Get result.
        time_start = time.time()
        fledge.database_interface.create_database(
            database_path=os.path.join(fledge.config.data_path, 'database.sqlite')
        )
        time_end = time.time()
        logger.info(f"Test create_database: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_connect_database(self):
        # Define expected result.
        expected = sqlite3.dbapi2.Connection

        # Get actual result.
        time_start = time.time()
        actual = type(fledge.database_interface.connect_database())
        time_end = time.time()
        logger.info(f"Test connect_database: Completed in {round(time_end - time_start, 6)} seconds.")

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_scenario_data(self):
        # Get result.
        time_start = time.time()
        fledge.database_interface.ScenarioData(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test ScenarioData: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_electric_grid_data(self):
        # Get result.
        time_start = time.time()
        fledge.database_interface.ElectricGridData(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test ElectricGridData: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_fixed_load_data(self):
        # Get result.
        time_start = time.time()
        fledge.database_interface.FixedLoadData(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test FixedLoadData: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_ev_charger_data(self):
        # Get result.
        time_start = time.time()
        fledge.database_interface.EVChargerData(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test EVChargerData: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_flexible_load_data(self):
        # Get result.
        time_start = time.time()
        fledge.database_interface.FlexibleLoadData(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test FlexibleLoadData: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_price_data(self):
        # Get result.
        time_start = time.time()
        fledge.database_interface.PriceData(fledge.config.test_scenario_name)
        time_end = time.time()
        logger.info(f"Test PriceData: Completed in {round(time_end - time_start, 6)} seconds.")


if __name__ == '__main__':
    unittest.main()
