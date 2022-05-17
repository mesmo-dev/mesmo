"""Setup script."""

import setuptools

setuptools.setup(
    name="mesmo",
    version="0.5.0",
    py_modules=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        "cvxpy",
        "cobmo",  # Not on package index. Needs to be manually installed from https://github.com/mesmo-dev/cobmo
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
