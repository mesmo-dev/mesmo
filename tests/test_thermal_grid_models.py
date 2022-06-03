"""Test thermal grid models."""

import inspect
from parameterized import parameterized
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)

# Get module members.
objects = [
    (object_name, object_handle)
    for object_name, object_handle in inspect.getmembers(mesmo.thermal_grid_models)
    # Get classes, excluding base classes.
    if inspect.isclass(object_handle) and not object_name.endswith("Base")
    # Exclude results classes.
    and not issubclass(object_handle, mesmo.utils.ResultsBase)
    # Exclude multimethod wrapper.
    and not object_name == "multimethod"
]


class TestThermalGridModels(unittest.TestCase):
    @parameterized.expand(objects)
    def test_thermal_grid_models(self, object_name, object_handle):
        # Test initialization routines.
        mesmo.utils.log_time(f"test `{object_name}`", log_level="info", logger_object=logger)
        if object_handle is mesmo.thermal_grid_models.ThermalPowerFlowSolutionSet:
            # `ThermalPowerFlowSolutionSet` requires additional arguments.
            thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(
                mesmo.config.config["tests"]["thermal_grid_scenario_name"]
            )
            der_model_set = mesmo.der_models.DERModelSet(mesmo.config.config["tests"]["thermal_grid_scenario_name"])
            der_operation_results = mesmo.thermal_grid_models.ThermalGridDEROperationResults(
                der_thermal_power_vector=der_model_set.der_thermal_power_nominal_timeseries,
            )
            object_handle(thermal_grid_model, der_operation_results)
        else:
            # Other object accept scenario name string.
            object_handle(mesmo.config.config["tests"]["thermal_grid_scenario_name"])
        mesmo.utils.log_time(f"test `{object_name}`", log_level="info", logger_object=logger)


if __name__ == "__main__":
    unittest.main()
