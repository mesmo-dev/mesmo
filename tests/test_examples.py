"""Test run scripts in the `examples` directory."""

import glob
import os
from parameterized import parameterized
import pyomo.environ as pyo
import time
import unittest

import fledge.config

logger = fledge.config.get_logger(__name__)

# Check availability of optimization solver.
try:
    optimization_solver_available = pyo.SolverFactory(fledge.config.solver_name).available()
except Exception:
    optimization_solver_available = False

# Run tests, if optimization solver is available.
if optimization_solver_available:

    # Obtain example scripts.
    examples_path = os.path.join(fledge.config.fledge_path, 'examples')
    example_scripts = glob.glob(os.path.join(examples_path, '*.py'))


    class TestExamples(unittest.TestCase):

        @parameterized.expand(
            [(os.path.basename(example_script),) for example_script in example_scripts])
        def test_examples(self, example_script):
            # Get result.
            time_start = time.time()
            os.system(f'python {os.path.join(examples_path, example_script)}')
            time_duration = time.time() - time_start
            logger.info(f"Test {example_script}: Completed in {time_duration:.6f} seconds.")


    if __name__ == '__main__':
        unittest.main()
