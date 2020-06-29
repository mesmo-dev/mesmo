"""Setup script."""

import setuptools

setuptools.setup(
    name='fledge',
    version='0.3.0',
    py_modules=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        # 'hvplot',  # TODO: Currently causes error on Github Actions.
        'matplotlib',
        'multimethod',
        'networkx',
        'natsort',
        'numpy',
        'OpenDSSDirect.py',
        'pandas',
        'parameterized',  # For tests.
        'pyomo',
        'pyyaml',
        'scipy',
    ],
)
