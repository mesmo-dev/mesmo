"""Setup script."""

import pathlib
import setuptools
import setuptools.command.develop
import setuptools.command.install
import subprocess
import sys

submodules = [
    'cobmo',
]

# Check if submodules are loaded.
for submodule in submodules:
    if not (pathlib.Path(__file__).parent.absolute() / submodule / 'setup.py').is_file():
        raise FileNotFoundError(
            f"No setup file found for submodule `{submodule}`. Please check if the submodule is loaded correctly."
        )

# Add post-installation routine to install submodules.
class develop_submodules(setuptools.command.develop.develop):
    def run(self):
        super().run()
        # Install submodules. Use `pip -v` to see subprocess outputs.
        for submodule in submodules:
            subprocess.check_call([sys.executable, '-m' 'pip', 'install', '-e', submodule])

# Add post-installation routine to install submodules.
class install_submodules(setuptools.command.install.install):
    def run(self):
        super().run()
        # Install submodules. Use `pip -v` to see subprocess outputs.
        for submodule in submodules:
            subprocess.check_call([sys.executable, '-m' 'pip', 'install', '-e', submodule])

setuptools.setup(
    name='mesmo',
    version='0.4.1',
    py_modules=setuptools.find_packages(),
    cmdclass={'install': install_submodules, 'develop': develop_submodules},
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        'cvxpy',
        'dill',
        'gurobipy',
        'kaleido',  # For static plot output with plotly.
        'matplotlib',
        'multimethod',
        'networkx',
        'natsort',
        'numpy',
        'opencv-python',
        'OpenDSSDirect.py',
        'pandas',
        'parameterized',  # For tests.
        'plotly',
        'pyyaml',
        'ray[default]',
        'scipy',
        'tqdm',
    ]
)
