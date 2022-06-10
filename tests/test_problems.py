"""Test problems."""

import inspect
from parameterized import parameterized
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)

# Get module members.
objects = [
    (object_name, object_handle)
    for object_name, object_handle in inspect.getmembers(mesmo.problems)
    # Get classes, excluding base classes.
    if inspect.isclass(object_handle) and not object_name.endswith("Base")
    # Exclude results classes.
    and not issubclass(object_handle, mesmo.utils.ResultsBase)
    # Exclude multimethod wrapper.
    and not object_name == "multimethod"
    # TODO: Exclude dict classes for now.
    and not object_name.endswith("Dict")
]


class TestProblems(unittest.TestCase):
    @parameterized.expand(objects)
    def test_problems(self, object_name, object_handle):
        mesmo.utils.log_time(f"test `{object_name}`", log_level="info", logger_object=logger)
        # Test initialization routines.
        object_instance = object_handle(mesmo.config.config["tests"]["scenario_name"])
        # Test solve(), get_results() routines.
        object_instance.solve()
        object_instance.get_results()
        mesmo.utils.log_time(f"test `{object_name}`", log_level="info", logger_object=logger)


if __name__ == "__main__":
    unittest.main()
