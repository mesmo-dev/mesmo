"""Setup script."""

import pathlib
import requests
import setuptools
import setuptools.command.develop
import setuptools.command.install
import subprocess
import sys
import tarfile

submodules = [
    "cobmo",
]

# Check if submodules are loaded.
base_path = pathlib.Path(__file__).parent.absolute()
for submodule in submodules:
    if not (base_path / submodule / "setup.py").is_file():
        try:
            subprocess.check_call(["git", "-C", str(base_path), "submodule", "update", "--init", "--recursive"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No setup file found for submodule `{submodule}`. Please check if the submodule is loaded correctly."
            )

# Add post-installation routine to install submodules in develop mode.
class develop_submodules(setuptools.command.develop.develop):
    def run(self):
        super().run()
        # Install submodules. Use `pip -v` to see subprocess outputs.
        for submodule in submodules:
            subprocess.check_call([sys.executable, "-m" "pip", "install", "-v", "-e", submodule])
        # Install HiGHS.
        install_highs()


# Add post-installation routine to install submodules in normal mode.
class install_submodules(setuptools.command.install.install):
    def run(self):
        super().run()
        # Install submodules. Use `pip -v` to see subprocess outputs.
        for submodule in submodules:
            subprocess.check_call([sys.executable, "-m" "pip", "install", "-v", submodule])
        # Install HiGHS solver.
        install_highs()


def install_highs():
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


setuptools.setup(
    name="mesmo",
    version="0.5.0",
    py_modules=setuptools.find_packages(),
    cmdclass={"install": install_submodules, "develop": develop_submodules},
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        "cvxpy",
        "dill",
        "dynaconf",
        "gurobipy",
        "kaleido",  # For static plot output with plotly.
        "matplotlib",
        "multimethod",
        "networkx",
        "natsort",
        "numpy",
        "opencv-python",
        "OpenDSSDirect.py",
        "pandas",
        "parameterized",  # For tests.
        "plotly",
        "pyyaml",
        "ray[default]",
        "redis",  # Temporary fix for ray import error. See: https://github.com/ray-project/ray/issues/24169
        "requests",  # For HiGHS installation.
        "scipy",
        "tqdm",
    ],
)
