"""Test example scripts."""

import gurobipy as gp
import importlib.machinery
import pathlib
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestExamples(unittest.TestCase):

    def test_example(self):
        mesmo.utils.log_time("test_example", log_level='info', logger_object=logger)
        # Find example scripts.
        example_files = [
            *((pathlib.Path(mesmo.config.base_path) / 'examples').glob('*.py')),
            *((pathlib.Path(mesmo.config.base_path) / 'examples' / 'publications').glob('*.py'))
        ]
        logger.info(f"Found example script files:\n{example_files}")
        for example_file in example_files:
            with self.subTest(example=example_file.stem):
                try:
                    # Import example script as module.
                    example_module = (
                        importlib.machinery.SourceFileLoader(example_file.stem, str(example_file)).load_module()
                    )
                    # Run main(), which will fail if it doesn't exist.
                    example_module.main()
                except gp.GurobiError as exception:
                    # Soft fail: Only raise warning on selected errors, since it may be due to solver not installed.
                    logger.warning(f"Test for example '{example_file.stem}' failed due to solver error.", exc_info=True)
        mesmo.utils.log_time("test_example", log_level='info', logger_object=logger)
