"""Run script which executes all example scripts, for testing."""

import subprocess
import sys

import mesmo


def main():

    # Obtain example scripts.
    examples_path = mesmo.config.base_path / "examples"
    example_scripts = list(examples_path.glob("*.py"))

    # Run all example scripts.
    for example_script in example_scripts:
        mesmo.utils.log_time(example_script, log_level="info")
        subprocess.check_call([sys.executable, (examples_path / example_script)])
        mesmo.utils.log_time(example_script, log_level="info")


if __name__ == "__main__":
    main()
