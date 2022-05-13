"""Test electric grid models."""

import inspect
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestElectricGridModels(unittest.TestCase):
    def test_electric_grid_models(self):

        # Get module members.
        for object_name, object_handle in inspect.getmembers(mesmo.electric_grid_models):
            # Get classes, excluding base classes.
            if inspect.isclass(object_handle) and not object_name.endswith("Base"):
                # Exclude results classes.
                if not issubclass(object_handle, mesmo.utils.ResultsBase):

                    # Test initialization routines.
                    mesmo.utils.log_time(f"test `{object_name}`", log_level="info", logger_object=logger)
                    if object_handle is mesmo.electric_grid_models.PowerFlowSolutionSet:
                        # `PowerFlowSolutionSet` requires additional arguments.
                        with self.subTest(object_name=object_name):
                            electric_grid_model = mesmo.electric_grid_models.ElectricGridModel(
                                mesmo.config.config["tests"]["scenario_name"]
                            )
                            der_model_set = mesmo.der_models.DERModelSet(mesmo.config.config["tests"]["scenario_name"])
                            der_operation_results = mesmo.electric_grid_models.ElectricGridDEROperationResults(
                                der_active_power_vector=der_model_set.der_active_power_nominal_timeseries,
                                der_reactive_power_vector=der_model_set.der_reactive_power_nominal_timeseries,
                            )
                            object_handle(electric_grid_model, der_operation_results)
                    else:
                        # Other object accept scenario name string.
                        with self.subTest(object_name=object_name):
                            object_handle(mesmo.config.config["tests"]["scenario_name"])
                    mesmo.utils.log_time(f"test `{object_name}`", log_level="info", logger_object=logger)


if __name__ == "__main__":
    unittest.main()
