# Getting Started

## Installation

### Quick installation

1. Check requirements:
   - Python 3.7
   - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
3. In your Python environment, run:
   1. `pip install -v -e path_to_repository`

Note that the `fledge.plots` module optionally allows adding contextual basemaps to grid plots for orientation. This requires installation of `contextily`, which is an optional dependency, because it needs to be installed through `conda` on Windows. If you need it, please follow the reccomended installation procedure below.

### Recommended installation

The following installation procedure requires additional steps, but can improve performance and includes optional dependencies. For example, the numpy and pandas packages are installed through Anaconda, which ensures the use of more performant math libraries. Additionally, the contextily package is installed, which is required for some geographical plots.

1. Check requirements:
   - [Anaconda Distribution](https://www.anaconda.com/distribution/) (Python 3.x version)
   - [Gurobi Optimizer](http://www.gurobi.com/) or [CPLEX Optimizer](https://www.ibm.com/analytics/cplex-optimizer)
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
3. In Anaconda Prompt, run:
   1. `conda create -n fledge python=3.7`
   2. `conda activate fledge`
   3. `conda install -c conda-forge contextily cvxpy`
   4. `conda install numpy pandas`
   5. `pip install -v -e path_to_repository`
4. If you want to use CPLEX:
   1. Install CPLEX Python interface (see latest CPLEX documentation).
   2. Create or modify `config.yml` (see below in "Configuration with `config.yml`").

### Alternative installation

If you are running into errors when installing or running FLEDGE, this may be due to incompatibility with new versions of package dependencies, which have yet to be discovered and fixed. As a workaround, try installing FLEDGE in an tested Anaconda environment via the the provided `environment.yml`, which represents the latest Anaconda Python environment in which FLEDGE was tested and is expected to work.

1. Check requirements:
   - Windows 10
   - [Anaconda Distribution](https://www.anaconda.com/distribution/) (Python 3.x version)
   - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
4. In Anaconda Prompt, run:
   1. `conda env create -f path_to_fledge_repository/environment.yml`
   2. `conda activate fledge`
   3. `pip install -v -e path_to_repository`.

``` important::
    Please create an issue on Github if you run into problems with the normal installation procedure.
```

## Examples

The `examples` directory contains run scripts which demonstrate possible usages of FLEDGE. You may also check the `test` directory for a better understanding of the API.

## Configuration with `config.yml`

FLEDGE configuration parameters (e.g. the output format of plots) can be set in `config.yml`. As an initial user, you most likely will not need to modify the configuration.

If you want to change the configuration, you can create or modify `config.yml` in the FLEDGE repository main directory. FLEDGE will automatically create `config.yml` if it does not exist. Initially, `config.yml` will be empty. You can copy configuration parameters from `fledge/config_default.yml` to `config.yml` and modify their value to define your local configuration. To define nested configuration parameters, you need to replicate the nested structure in `config.yml`. For example, to define CPLEX as the optimization solver, use:

```
optimization:
  solver_name: cplex
```

The configuration parameters which are defined in `config.yml` will take precedence over those defined in `fledge/config_default.yml`. If you would like to revert a parameter to its default value, just delete the parameter from `config.yml`. Please do not modify `fledge/config_default.yml` directly.

## Papers

The following papers have been prepared in relation to FLEDGE:

- [Preprint] Troitzsch, S., Grussmann, M., Zhang, K., & Hamacher, T., **Distribution Locational Marginal Pricing for Combined Thermal and Electric Grid Operation**, 2020. [`doi: 10.36227/techrxiv.11918712`](https://doi.org/10.36227/techrxiv.11918712)
    - FLEDGE [v0.3.0](https://github.com/TUMCREATE-ESTL/fledge/releases/tag/v0.3.0) was used to preare the results for this paper.
    - The related script is [`examples/paper_2020_dlmp_combined_thermal_electric.py`](https://github.com/TUMCREATE-ESTL/fledge/blob/v0.3.0/examples/paper_2020_dlmp_combined_thermal_electric.py).
- Troitzsch, S., Hanif, S., Zhang, K., Trpovski, A., & Hamacher, T., **Flexible Distribution Grid Demonstrator (FLEDGE): Requirements and Software Architecture**, in IEEE PES General Meeting, Atlanta, GA, USA, 2019. [`doi: 10.1109/PESGM40551.2019.8973567`](https://doi.org/10.1109/PESGM40551.2019.8973567).
    - The paper served as an outline for the [software architecture](architecture.md) of FLEDGE.
- D. Recalde, A. Trpovski, S. Troitzsch, K. Zhang, S. Hanif, and T. Hamacher, **A Review of Operation Methods and Simulation Requirements for Future Smart Distribution Grids**, in IEEE PES Innovative Smart Grid Technologies Conference Asia, Singapore, 2018. [`doi:10.1109/ISGT-Asia.2018.8467850`](https://doi.org/10.1109/ISGT-Asia.2018.8467850).
    - The review paper initiated the development of FLEDGE.
    
The following papers served as the methodological basis for the implementation of FLEDGE as well as for the development of test cases:
- S. Hanif, K. Zhang, C. Hackl, M. Barati, H. B. Gooi, and T. Hamacher, **Decomposition and Equilibrium Achieving Distribution Locational Marginal Prices using Trust-Region Method**, IEEE Transactions on Smart Grid, 2018. [`doi:10.1109/TSG.2018.2822766`](https://doi.org/10.1109/TSG.2018.2822766).
- K. Zhang, S. Hanif, C. M. Hackl, and T. Hamacher, **A Framework for Multi-Regional Real-Time Pricing in Distribution Grids**, IEEE Transactions Smart Grid, vol. 10, no. 6, pp. 6826–6838, 2019. [`doi:10.1109/TSG.2019.2911996`](https://doi.org/10.1109/TSG.2019.2911996).
- A. Trpovski, D. Recalde, and T. Hamacher, **Synthetic Distribution Grid Generation Using Power System Planning: Case Study of Singapore**, in UPEC International Universities Power Engineering Conference, 2018. [`doi:10.1109/UPEC.2018.8542054`](https://doi.org/10.1109/UPEC.2018.8542054).

## Contributing

If you are keen to contribute to this project, please see [Contributing](contributing.md).
