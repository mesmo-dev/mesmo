"""Test example scripts."""

import importlib.util
import pathlib
import sys
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestExamples(unittest.TestCase):
    def test_example(self):
        mesmo.utils.log_time("test_example", log_level="info", logger_object=logger)
        # Find example scripts.
        example_files = [
            *((pathlib.Path(mesmo.config.base_path) / "examples").glob("*.py")),
            # TODO: Excluded publication scripts from tests due to HiGHS solver errors.
            # *((pathlib.Path(mesmo.config.base_path) / 'examples' / 'publications').glob('*.py'))
        ]
        logger.info(f"Found example script files:\n{example_files}")
        for example_file in example_files:
            with self.subTest(example=example_file.stem):
                # Add directory to path, to enable sibling imports in examples scripts.
                if str(example_file.parent) not in sys.path:
                    sys.path.append(str(example_file.parent))
                # Import example script as module.
                # - Reference: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
                spec = importlib.util.spec_from_file_location(example_file.stem, str(example_file))
                module = importlib.util.module_from_spec(spec)
                sys.modules[example_file.stem] = module
                spec.loader.exec_module(module)
                # Run main(), which will fail if it doesn't exist.
                module.main()
        mesmo.utils.log_time("test_example", log_level="info", logger_object=logger)
