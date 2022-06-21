"""Development dependencies setup script."""

import argparse
import pathlib
import subprocess
import sys
import tarfile

import requests

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
                        f"ERROR: No setup file found for submodule `{submodule}`. "
                        f"Please check if the submodule is loaded correctly."
                    ) from exception

    # Install submodules in develop mode.
    if run_all:
        print("Installing submodules in development mode.")
        for submodule in submodules:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{base_path / submodule}"])

    # Install MESMO.
    if run_all:
        print("Installing MESMO in development mode.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{base_path}[tests]"])

    # Install HiGHS solver.
    if run_all or run_highs:
        print("Installing HiGHS solver.")
        # Make HiGHS directory.
        (base_path / "highs").mkdir(exist_ok=True)
        # Construct HiGHS binary download URL.
        base_url = "https://github.com/JuliaBinaryWrappers/HiGHSstatic_jll.jl/releases/download/"
        version_string = "HiGHSstatic-v1.2.2%2B0/HiGHSstatic.v1.2.2"
        if sys.platform == "win32":
            architecture_string = "x86_64-w64-mingw32"
        elif sys.platform == "darwin":
            architecture_string = "x86_64-apple-darwin"
        else:
            architecture_string = "x86_64-linux-gnu-cxx11"
        url = f"{base_url}{version_string}.{architecture_string}.tar.gz"
        # Download and unpack HiGHS binary files.
        try:
            with requests.get(url, stream=True) as request:
                request.raise_for_status()
                with open(base_path / "highs" / "highs.tar.gz", "wb") as file:
                    for chunk in request.iter_content(chunk_size=10240):
                        file.write(chunk)
                with tarfile.open(base_path / "highs" / "highs.tar.gz") as file:
                    file.extractall(base_path / "highs")
            # Remove downloaded archive file.
            (base_path / "highs" / "highs.tar.gz").unlink()
        except requests.ConnectionError:
            # Soft-fail on download connection errors.
            print(
                "WARNING: HiGHS solver could not be installed automatically. "
                "Please configure optimization solver for MESMO manually."
            )
        else:
            print("Successfully installed HiGHS solver.")


if __name__ == "__main__":
    main()
