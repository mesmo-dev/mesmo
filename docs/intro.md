# Getting Started

## Installation

### Quick installation

1. Check requirements:
    - Python 3.7
2. Clone or download repository.
3. In your Python environment, run `pip install -e path_to_fledge_repository`.

### Recommended installation

The following installation procedure contains additional steps and requirements which can improve the numerical performance when running FLEDGE.

1. Check requirements:
    - [Anaconda Distribution](https://www.anaconda.com/distribution/) (Python 3.x version)
    - [Gurobi Optimizer](http://www.gurobi.com/) or [CPLEX Optimizer](https://www.ibm.com/analytics/cplex-optimizer)
2. Clone or download repository.
3. In Anaconda Prompt, run:
    1. `conda create -n fledge python=3.7`
    2. `conda activate fledge`
    3. `conda install pandas`
    4. `pip install -e path_to_fledge_repository`.
4. In `fledge/config.py`, change `solver_name` to `'gurobi'` or `'cplex'`.

### Alternative installation

If you are running into errors when installing or running FLEDGE, this may be due to incompatibility with new versions of package dependencies, which have yet to be discovered and fixed. As a workaround, try installing FLEDGE in an tested Anaconda environment via the the provided `environment.yml`, which represents the latest Anaconda Python environment in which FLEDGE was tested and is expected to work.

1. Check requirements:
    - Windows 10
    - [Anaconda Distribution](https://www.anaconda.com/distribution/) (Python 3.x version)
2. Clone or download repository.
4. In Anaconda Prompt, run:
    1. `conda env create -f path_to_fledge_repository/environment.yml`
    2. `conda activate fledge`
    3. `pip install -e path_to_fledge_repository`.

``` important::
    Please also create an issue on Github if you run into problems with the normal installation procedure.
```

### Optional steps



## Examples

The `examples` directory contains run scripts which demonstrate possible usages of FLEDGE. You may also check the `test` directory for further examples.

## Contributing

If you are keen to contribute to this project, please see [Contributing](contributing.md).
