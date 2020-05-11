"""Test database interface."""

import os
import sqlite3
import time
import unittest

import fledge.config
import fledge.data_interface

logger = fledge.config.get_logger(__name__)


class TestDatabaseInterface(unittest.TestCase):

    def test_recreate_database(self):
        # Get result.
        time_start = time.time()
        fledge.data_interface.recreate_database()
        time_duration = time.time() - time_start
        logger.info(f"Test recreate_database: Completed in {time_duration:.6f} seconds.")

    def test_connect_database(self):
        # Define expected result.
        expected = sqlite3.dbapi2.Connection

        # Get actual result.
        time_start = time.time()
        actual = type(fledge.data_interface.connect_database())
        time_duration = time.time() - time_start
        logger.info(f"Test connect_database: Completed in {time_duration:.6f} seconds.")

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_scenario_data(self):
        # Get result.
        time_start = time.time()
        fledge.data_interface.ScenarioData(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test ScenarioData: Completed in {time_duration:.6f} seconds.")

    def test_electric_grid_data(self):
        # Get result.
        time_start = time.time()
        fledge.data_interface.ElectricGridData(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridData: Completed in {time_duration:.6f} seconds.")

    def test_thermal_grid_data(self):
        # Get result.
        time_start = time.time()
        fledge.data_interface.ThermalGridData('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test ThermalGridData: Completed in {time_duration:.6f} seconds.")

    def test_electric_grid_der_data(self):
        # Get result.
        time_start = time.time()
        fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test DERData: Completed in {time_duration:.6f} seconds.")

    def test_price_data(self):
        # Get result.
        time_start = time.time()
        fledge.data_interface.PriceData(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test PriceData: Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
