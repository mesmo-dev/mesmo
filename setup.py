"""Setup script."""

import os
import setuptools
import subprocess
import sys

# Check if submodule is loaded.
if not os.path.exists(os.path.join('cobmo', 'setup.py')):
    raise FileNotFoundError(
        "No setup file found for submodule `cobmo`. "
        "Please check if the submodule is loaded correctly."
    )

# Install submodules. (This will run without command line outputs.)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', 'cobmo'])

# Install Gurobi interface. (This will run without command line outputs.)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-i', 'https://pypi.gurobi.com', 'gurobipy'])

setuptools.setup(
    name='fledge',
    version='0.3.0',
    py_modules=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        'cvxpy',
        'diskcache',
        'kaleido',  # For static plot output with plotly.
        'matplotlib',
        'multimethod',
        'multiprocess',
        'networkx',
        'natsort',
        'numpy',
        'opencv-python',
        'OpenDSSDirect.py',
        'pandas',
        'parameterized',  # For tests.
        'plotly',
        'pyyaml',
        'scipy',
    ]
)
