"""Test electric grid models."""

import time
import unittest

import fledge.config
import fledge.database_interface
import fledge.der_models

logger = fledge.config.get_logger(__name__)


class TestDERModels(unittest.TestCase):

    def test_fixed_load_model(self):
        # Obtain test data.
        fixed_load_data = fledge.database_interface.FixedLoadData(fledge.config.test_scenario_name)

        # Get result.
        time_start = time.time()
        fledge.der_models.FixedLoadModel(
            fixed_load_data,
            fixed_load_data.fixed_loads.index[0]  # Take `load_name` of first row.
        )
        time_end = time.time()
        logger.info(f"Test FixedLoadModel: Completed in {round(time_end - time_start, 6)} seconds.")

    def test_ev_charger_model(self):
        # Obtain test data.
        ev_charger_data = fledge.database_interface.EVChargerData(fledge.config.test_scenario_name)

        # Get result.
        time_start = time.time()
        fledge.der_models.EVChargerModel(
            ev_charger_data,
            ev_charger_data.ev_chargers.index[0]  # Take `load_name` of first row.
        )
        time_end = time.time()
        logger.info(f"Test EVChargerModel: Completed in {round(time_end - time_start, 6)} seconds.")


if __name__ == '__main__':
    unittest.main()