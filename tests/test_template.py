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
        time_end = time.time()
        logger.info("Test equal: Completed in {} seconds.".format(round(time_end - time_start, 6)))

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_not_equal(self):
        # Define expected result.
        expected = 4

        # Get actual result.
        time_start = time.time()
        actual = 2 + 1
        time_end = time.time()
        logger.info("Test not equal: Completed in {} seconds.".format(round(time_end - time_start, 6)))

        # Compare expected and actual.
        self.assertNotEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
