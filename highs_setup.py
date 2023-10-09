"""HiGHS solver setup script."""

import pathlib
import sys
import tarfile

import requests

# Get repository base path.
base_path = pathlib.Path(__file__).parent.absolute()


def main():
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
