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
        der_data = fledge.database_interface.ElectricGridDERData(fledge.config.test_scenario_name)

        # Get result.
        time_start = time.time()
        fledge.der_models.FixedLoadModel(
            der_data,
            der_data.fixed_loads.index[0]  # Take `der_name` of first row.
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FixedLoadModel: Completed in {time_duration:.6f} seconds.")

    def test_ev_charger_model(self):
        # Obtain test data.
        der_data = fledge.database_interface.ElectricGridDERData(fledge.config.test_scenario_name)

        # Get result.
        time_start = time.time()
        fledge.der_models.EVChargerModel(
            der_data,
            der_data.ev_chargers.index[0]  # Take `der_name` of first row.
        )
        time_duration = time.time() - time_start
        logger.info(f"Test EVChargerModel: Completed in {time_duration:.6f} seconds.")

    def test_flexible_load_model(self):
        # Obtain test data.
        der_data = fledge.database_interface.ElectricGridDERData(fledge.config.test_scenario_name)

        # Get result.
        time_start = time.time()
        fledge.der_models.FlexibleLoadModel(
            der_data,
            der_data.flexible_loads.index[0]  # Take `der_name` of first row.
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FlexibleLoadModel: Completed in {time_duration:.6f} seconds.")

    def test_der_model_set(self):
        # Get result.
        time_start = time.time()
        fledge.der_models.DERModelSet(fledge.config.test_scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test DERModelSet: Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()