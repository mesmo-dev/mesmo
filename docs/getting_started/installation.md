# Installation

## Recommended installation

MESMO has not yet been deployed to Python `pip` / `conda` package indexes, but can be installed in a local development environment as follows:

1. Check requirements:
   - Python distribution: [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
   - Optimization solver: [Gurobi Optimizer](http://www.gurobi.com/) or other solver [via manual configuration](optimization-solver).
2. Clone or download the repository. Ensure that the `cobmo` submodule directory is loaded as well.
3. In `conda`-enabled shell (e.g. Anaconda Prompt), run:
   - `cd path_to_mesmo_repository`
   - On Intel CPUs: `conda create -n mesmo -c conda-forge python=3.8 contextily cvxpy numpy pandas scipy "libblas=*=*mkl"; conda activate mesmo; pip install -v -e .`
   - On other CPUs: `conda create -n mesmo -c conda-forge python=3.8 contextily cvxpy numpy pandas scipy; conda activate mesmo; pip install -v -e .`

The installation via `conda` is recommended, because it sets up a dedicated Python environment and supports the performant MKL math library on Intel CPUs for `numpy` / `scipy`. It also allows installing the `contextily` package on Windows, which is required for some geographical plots. The direct installation via `pip` in a non-`conda` environment is also possible, but is currently not tested.

## Alternative installation

If you are running into errors when installing or running MESMO, this may be due to incompatibility with new versions of package dependencies, which have yet to be discovered and fixed. As a workaround, try installing MESMO via the provided `environment-windows-latest.yml` / `environment-macos-latest.yml` / `environment-ubuntu-latest.yml`, which represent the latest Python environment in which MESMO was tested and is expected to work for the respective OS.

```{important}
Please create an issue on Github if you run into problems with the normal installation procedure.
```

1. Check requirements:
   - Python distribution: [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
   - Optimization solver: [Gurobi Optimizer](http://www.gurobi.com/) or other optimization solver [via manual configuration](configuration.md).
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
4. In `conda`-enabled shell (e.g. Anaconda Prompt), run:
   - `cd path_to_mesmo_repository`
   - On Windows: `conda env create -f path_to_mesmo_repository/environment-windows-latest.yml; conda activate mesmo; pip install -v -e .`
   - On macOS: `conda env create -f path_to_mesmo_repository/environment-macos-latest.yml; conda activate mesmo; pip install -v -e .`
   - On Ubuntu: `conda env create -f path_to_mesmo_repository/environment-ubuntu-latest.yml; conda activate mesmo; pip install -v -e .`
