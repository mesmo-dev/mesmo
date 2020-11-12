"""Test DER models."""

import time
import unittest

import fledge.config
import fledge.data_interface
import fledge.der_models

logger = fledge.config.get_logger(__name__)


class TestDERModels(unittest.TestCase):

    def test_fixed_load_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        time_start = time.time()
        fledge.der_models.FixedLoadModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'fixed_load', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FixedLoadModel: Completed in {time_duration:.6f} seconds.")

    def test_fixed_ev_charger_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        time_start = time.time()
        fledge.der_models.FixedEVChargerModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'fixed_ev_charger', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FixedEVChargerModel: Completed in {time_duration:.6f} seconds.")

    def test_fixed_generator_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        time_start = time.time()
        fledge.der_models.FixedGeneratorModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'fixed_generator', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FlexibleLoadModel: Completed in {time_duration:.6f} seconds.")

    def test_flexible_load_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        time_start = time.time()
        fledge.der_models.FlexibleLoadModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'flexible_load', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FlexibleLoadModel: Completed in {time_duration:.6f} seconds.")

    def test_flexible_generator_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        time_start = time.time()
        fledge.der_models.FlexibleGeneratorModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'flexible_generator', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FlexibleGeneratorModel: Completed in {time_duration:.6f} seconds.")

    def test_storage_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        time_start = time.time()
        fledge.der_models.StorageModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'storage', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test StorageModel: Completed in {time_duration:.6f} seconds.")

    def test_flexible_building_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        time_start = time.time()
        fledge.der_models.FlexibleBuildingModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'flexible_building', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test FlexibleBuildingModel: Completed in {time_duration:.6f} seconds.")

    def test_cooling_plant_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData('paper_2020_2_scenario_6_7_8')

        # Get result.
        time_start = time.time()
        fledge.der_models.CoolingPlantModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'cooling_plant', 'der_name'].iat[0]
        )
        time_duration = time.time() - time_start
        logger.info(f"Test CoolingPlantModel: Completed in {time_duration:.6f} seconds.")

    def test_der_model_set(self):
        # Get result.
        time_start = time.time()
        fledge.der_models.DERModelSet(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test DERModelSet: Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
