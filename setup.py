"""Setup script."""

import setuptools
import subprocess
import sys

# Install submodules.
# - Note that this will run without command line outputs.
# - TODO: Update documentation.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', 'cobmo'])

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
