"""Development dependencies setup script."""

import argparse
import pathlib
import subprocess
import sys

submodules = [
    "cobmo",
]

# Get repository base path.
base_path = pathlib.Path(__file__).parent.absolute()

# Get command line arguments.
parser = argparse.ArgumentParser(description="Development dependencies setup script for SITEM Middleware")
parser.add_argument("--highs", action="store_true", help="Only (re)install HiGHS solver and skip other steps")
args = parser.parse_args()


def main():
    # Check which steps to be run.
    run_highs = args.highs
    run_all = not run_highs

    # Check if submodules are loaded.
    if run_all:
        print("Checking if submodules are loaded.")
        for submodule in submodules:
            if not (base_path / submodule / "setup.py").is_file():
                try:
                    subprocess.check_call(["git", "-C", f"{base_path}", "submodule", "update", "--init", "--recursive"])
                except FileNotFoundError as exception:
                    raise FileNotFoundError(
                        f"No setup file found for submodule `{submodule}`. "
                        f"Please check if the submodule is loaded correctly."
                    ) from exception

    # Install MESMO and submodules in develop mode.
    if run_all:
        print("Installing poetry package management tool.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "poetry"])
        print("Installing MESMO in development mode.")
        subprocess.check_call([sys.executable, "-m", "poetry", "install"], cwd=base_path)

    # Install HiGHS solver.
    if run_all or run_highs:
        print("Installing HiGHS solver.")
        subprocess.check_call([sys.executable, "-m", "poetry", "run", "python", f"{base_path / 'highs_setup.py'}"])


if __name__ == "__main__":
    main()
