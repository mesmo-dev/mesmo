# Installation

## Recommended installation

MESMO has not yet been deployed to Python `pip` / `conda` package indexes, but can be installed in a local development environment as follows:

1. Install `conda`-based Python distribution¹ such as [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
2. Clone or download the repository. Ensure that the `cobmo` submodule directory is loaded as well.
3. In `conda`-enabled shell (e.g. Anaconda Prompt), run:
    - `cd path_to_mesmo_repository`
    - `conda create -n mesmo -c conda-forge python=3.8 contextily cvxpy numpy pandas scipy`
    - `conda activate mesmo`
    - `python development_setup.py`
    - On Intel CPUs²: `conda install -c conda-forge "libblas=*=*mkl"`

MESMO ships with [HiGHS](https://highs.dev/) as default optimization solver³, but also supports [Gurobi](http://www.gurobi.com/) and [any CVXPY-supported solvers](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver).

### Notes

¹ The installation via `conda` is recommended, because it sets up a dedicated Python environment and supports the performant Intel MKL math library on Intel CPUs for `numpy` / `scipy`. It also allows installing the `contextily` package on Windows, which is required for some geographical plots. The direct installation via `pip` in a non-`conda` environment is also possible, but is currently not tested.

² On Intel CPUs, using the Intel MKL math library enables better performance in `numpy` / `scipy` over the default OpenBLAS library. With conda-forge, these libraries can be configured via [`conda install` command](https://conda-forge.org/docs/maintainer/knowledge_base.html#switching-blas-implementation).

³ HiGHS is currently MESMO's default optimization solver and the HiGHS binaries are automatically fetched during MESMO setup. Other solvers can be selected via [MESMO configuration](configuration_reference.md#optimization-solver-configuration). For better performance, MESMO implements direct solver interfaces to HiGHS and Gurobi. Other solvers are indirectly supported [via CVXPY](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver). CVXPY comes bundled with several open-source solvers and supports additional solvers via manual installation (see "Install with ... support" sections in [CVPXY installation guide](https://www.cvxpy.org/install/index.html)). Note that interfacing solvers via CVXPY currently has [performance limitations](https://github.com/cvxpy/cvxpy/issues/704) for large-scale scenarios.

## Alternative installation

If you are running into errors when installing or running MESMO, this may be due to incompatibility with new versions of package dependencies, which have yet to be discovered and fixed. As a workaround, try installing MESMO via the provided `environment-windows-latest.yml` / `environment-macos-latest.yml` / `environment-ubuntu-latest.yml`, which represent the latest Python environment in which MESMO was tested and is expected to work for the respective OS.

```{important}
Please create an issue on Github if you run into problems with the recommended installation procedure.
```

1. Install `conda`-based Python distribution such as [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
4. In `conda`-enabled shell (e.g. Anaconda Prompt), run:
    - `cd path_to_mesmo_repository`
    - On Windows: `conda env create -f environment-windows-latest.yml`
    - On macOS: `conda env create -f environment-macos-latest.yml`
    - On Ubuntu: `conda env create -f environment-ubuntu-latest.yml`
    - `conda activate mesmo`
    - `python development_setup.py`
