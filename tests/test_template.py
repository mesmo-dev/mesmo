"""Test template."""

import time
import unittest

import fledge.config

logger = fledge.config.get_logger(__name__)


class TestTemplate(unittest.TestCase):

    def test_equal(self):
        # Define expected result.
        expected = 4

        # Get actual result.
        time_start = time.time()
        actual = 2 + 2
        time_duration = time.time() - time_start
        logger.info(f"Test equal: Completed in {time_duration:.6f} seconds.")

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_not_equal(self):
        # Define expected result.
        expected = 4

        # Get actual result.
        time_start = time.time()
        actual = 2 + 1
        time_duration = time.time() - time_start
        logger.info(f"Test not equal: Completed in {time_duration:.6f} seconds.")

        # Compare expected and actual.
        self.assertNotEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
