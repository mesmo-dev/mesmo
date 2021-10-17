# MESMO - Multi-Energy System Modeling and Optimization

[![DOI](https://zenodo.org/badge/201130660.svg)](https://zenodo.org/badge/latestdoi/201130660)

> **Looking for FLEDGE?** - The Flexible Distribution Grid Demonstrator (FLEDGE) is now called Multi-Energy System Modeling and Optimization (MESMO) and has moved to this shiny new repository.

Multi-Energy System Modeling and Optimization (MESMO) is a Python tool for optimal operation problems of electric and thermal distribution grids along with distributed energy resources (DERs), such as flexible building loads, electric vehicle (EV) chargers, distributed generators (DGs) and energy storage systems (ESS). To this end, it implements 1) electric grid models, 2) thermal grid models, 3) DER models, and 4) optimal operation problems.

> Work in progress: Please note that the repository is under active development and the interface may change without notice. Create an [issue](https://github.com/mesmo-dev/mesmo/issues) if you have ideas / comments / criticism that may help to make the tool more useful.

## Features

- Electric grid models:
    - Obtain nodal / branch admittance and incidence matrices¹.
    - Obtain steady state power flow solution for nodal voltage / branch flows / losses via fixed-point algorithm / [OpenDSS](https://github.com/dss-extensions/OpenDSSDirect.py)¹.
    - Obtain sensitivity matrices of global linear approximate grid model¹.
    - ¹Fully enabled for unbalanced / multiphase grid configuration.
- Thermal grid models:
    - Obtain nodal / branch incidence matrices.
    - Obtain thermal power flow solution for nodal head / branch flows / pumping losses.
    - Obtain sensitivity matrices of global linear approximate grid model.
- Distributed energy resource (DER) models:
    - Time series models for fixed loads.
    - Time series models for EV charging.
    - Linear models for flexible building loads.
- Optimal operation problems:
    - Obtain numerical optimization problem for combined optimal operation for electric / thermal grids with DERs.
    - Obtain electric / thermal optimal power flow solution.
    - Obtain distribution locational marginal prices (DLMPs) for the electric / thermal grids.

## Documentation

The documentation is located at [mesmo-dev.github.io/mesmo](https://mesmo-dev.github.io/mesmo).

## Installation

MESMO has not yet been deployed to Python `pip` / `conda` package indexes, but can be installed in a local development environment as follows:

1. Check requirements:
    - Python distribution¹: [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
    - Optimization solver²: [Gurobi](http://www.gurobi.com/) or [CVXPY-supported solver](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver).
2. Clone or download the repository. Ensure that the `cobmo` submodule directory is loaded as well.
3. In `conda`-enabled shell (e.g. Anaconda Prompt), run:
    - `cd path_to_mesmo_repository`
    - `conda create -n mesmo -c conda-forge python=3.8 contextily cvxpy numpy pandas scipy; conda activate mesmo`
    - `pip install -v -e .`
    - On Intel CPUs³: `conda install -c conda-forge "libblas=*=*mkl"`

For notes ¹/²/³ and alternative installation guide, see [docs/getting_started/installation.md](docs/getting_started/installation.md).

## Contributing

If you are keen to contribute to this project, please see [docs/contributing.md](./docs/contributing.md).

## Publications

Information on citing MESMO and a list of related publications is available at [docs/publications.md](docs/publications.md).

## Acknowledgements

- This work was financially supported by the Singapore National Research Foundation under its Campus for Research Excellence And Technological Enterprise (CREATE) programme.
- Sebastian Troitzsch implemented the initial version of MESMO and maintains this repository.
- Sarmad Hanif and Kai Zhang developed the underlying electric grid modelling, fixed-point power flow solution and electric grid approximation methodologies.
- Arif Ahmed implemented the implicit Z-bus power flow solution method & overhead line type definitions.
- Mischa Grussmann developed the thermal grid modelling and approximation methodologies.
