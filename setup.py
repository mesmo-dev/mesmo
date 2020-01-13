"""Setup script."""

import setuptools

setuptools.setup(
    name='fledge',
    version='0.3.0',
    py_modules=setuptools.find_packages(),
    install_requires=[
        'hvplot',
        'multimethod',
        'natsort',
        'numpy',
        'OpenDSSDirect.py',
        'pandas',
        'parameterized',  # For tests.
        'pyomo',
        'scipy',
    ],
)
