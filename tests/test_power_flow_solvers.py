"""Test power flow solvers."""

import numpy as np
import os
import pandas as pd
from parameterized import parameterized
import scipy.sparse
import time
import unittest

import fledge.config
import fledge.power_flow_solvers

logger = fledge.config.get_logger(__name__)

as_complex = np.vectorize(np.complex)  # Utility function to convert strings in numpy array to complex numbers.


class TestPowerFlowSolvers(unittest.TestCase):

    @parameterized.expand([
        (0,),
        (1,),
        (2,),
    ])
    def test_get_voltage_fixed_point_1(self, test_number):
        # Obtain test data.
        path = os.path.join(fledge.config.test_data_path, 'test_get_voltage_vector_' + str(test_number))
        admittance_matrix = scipy.sparse.csr_matrix(as_complex(
            pd.read_csv(os.path.join(path, 'admittance_matrix.csv'), header=None, dtype=str).values
        ))
        transformation_matrix = scipy.sparse.csr_matrix(as_complex(
            pd.read_csv(os.path.join(path, 'transformation_matrix.csv'), header=None).values
        ))
        power_vector_wye = as_complex(
            pd.read_csv(os.path.join(path, 'power_vector_wye.csv'), header=None, dtype=str).values
        )
        power_vector_delta = as_complex(
            pd.read_csv(os.path.join(path, 'power_vector_delta.csv'), header=None, dtype=str).values
        )
        voltage_vector_no_load = as_complex(
            pd.read_csv(os.path.join(path, 'voltage_vector_no_load.csv'), header=None, dtype=str).values
        )
        voltage_vector_solution = as_complex(
            pd.read_csv(os.path.join(path, 'voltage_vector_solution.csv'), header=None, dtype=str).values
        )

        # Define expected result.
        expected = abs(voltage_vector_solution[3:])

        # Get actual result.
        time_start = time.time()
        actual = abs(fledge.power_flow_solvers.get_voltage_fixed_point(
            admittance_matrix[3:, 3:],
            transformation_matrix[3:, 3:],
            power_vector_wye[3:],
            power_vector_delta[3:],
            np.zeros(power_vector_wye[3:].shape),
            np.zeros(power_vector_delta[3:].shape),
            voltage_vector_no_load[3:],
            voltage_vector_no_load[3:]
        ))
        time_end = time.time()
        logger.info(f"Test get_voltage_fixed_point #1: Solved in {round(time_end - time_start, 6)} seconds.")

        # Compare expected and actual.
        np.testing.assert_array_almost_equal(actual, expected, decimal=0)


if __name__ == '__main__':
    unittest.main()
