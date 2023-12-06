# Change log

Version numbering follows the [Semantic Versioning principle](https://semver.org/).

## Next release

### New features

- Added constant power DER model, which is intended as the most simplistic placeholder DER type. The constant power DER model now acts as default value if no DER model is defined in the scenario definition.
- Work-in-progress: Added trust-region algorithm as solve method for the optimal operation problem.
- Work-in-progress: Added support for meshed thermal grid models.

### Changes

- Revised "Examples" section in the documentation and added three tutorials.
- Work-in-progress: Revised `thermal_grid_models` module structure:
    - Revised `ThermalGridModel` attributes and added method `get_branch_loss_coefficient_vector()` method.
    - The implementation of `ThermalPowerFlowSolution` is split into `ThermalPowerFlowSolutionBase` as base class and `ThermalPowerFlowSolutionExplicit` for the power flow solution algorithm.
    - `ThermalPowerFlowSolution` now serves as high-level interface which passes calls to `ThermalPowerFlowSolutionExplicit` for radial grids and raises `NotImplementedError` for meshed grids.
- Added aggregated attributes in `DERModels`: `der_active_power_nominal_timeseries`, `der_reactive_power_nominal_timeseries`, `der_thermal_power_nominal_timeseries`
- Refactored `data` directory, to separate data items for default-type library definitions, template definitions, test-case scenario definitions and cobmo-related definitions.

### Fixes

- `OptimizationProblem.solve()` does not try to retrieve duals for non-convex problems anymore.

## [0.5.0](https://github.com/mesmo-dev/mesmo/releases/tag/0.5.0)

With this release, the Flexible Distribution Grid Demonstrator (FLEDGE) was renamed to Multi-Energy System Modeling and Optimization (MESMO).

### New features

- Added new optimization problem object (`utils.OptimizationProblem`) as main interface for defining optimization problems with functionality to 1) export the standard form for LP / QP, 2) directly interface Gurobi for better performance with large problems.
- Overhead line types can now be defined in terms of conductor data and geometric arrangement (Arif Ahmed).
- Added local-approximation variant for linear electric grid model.
- Added linear model set for electric grid model, which enables defining separate linear models for each time step.
- Added power flow solution set, to obtain power flow solutions more conveniently for multiple time steps.
- Added pre-solve method for DER model set, to obtain baseline nominal power time series for flexible DERs.

### Changes

- Improved / simplified `define_optimization...()` methods for most use cases.
- Revised `define_optimization...()` methods for new optimization problem object.
- Switched from `multiprocess` to `ray` for parallel processing for performance reasons.
- Revised documentation structure, overhauled the architecture documentation and added the configuration reference.

## [v0.4.1](https://github.com/mesmo-dev/mesmo/releases/tag/v0.4.1)

### Fixes

- Updated `environment.yml`.
- Updated version indicators.

## [v0.4.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.4.0)

### New features

- Added problems module with definitions for nominal operation problem (simulation) and optimal operation problem (optimization).
- Added high-level API for executing optimal & nominal operation problems.
- Added various DER models.
- Enabled most DERs for thermal grids (Verena Kleinschmidt).
- Added ability to define electric grid model as single-phase-approximate.
- Added Z-Bus power flow solution method (Arif Ahmed).
- Added plots module (work-in-progress).
- Added ability to set local configuration with `config.yml`.
- Added ability to set base units for apparent power, voltage and thermal power for in scenario definition.

### Changes

- Moved implementation of optimization problems from Pyomo to CVXPY for performance improvements.
- Reformulated optimization constraints to use normalized values for improved numerical performance.
- Improved MESMO definition data format documentation.
- Refactored DER model data definition format.
- Refactored price data object.
- Various fixes in linear electric grid model model and DLMP calculations.
- Introduced various error messages for common issues.

## [v0.3.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.3.0)

### New features

- Moved to Python as main implementation language.
- Extended linear electric models with methods for defining optimization variables / constraints.
- Added thermal grid model.
- Added linear thermal grid model with methods for defining optimization variables / constraints.
- Added DER models and integrated CoBMo for flexible building models.
- Added methods for defining operation limits and obtaining DLMPs for electric and thermal grids.
- Provided various example scripts for running optimal operation problems for DERs / electric grids / thermal grids / multi-energy grids.

## [v0.2.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.2.0)

### Auxiliary Release

- Snapshot before moving to Python as main implementation language.

## [v0.1.0](https://github.com/mesmo-dev/mesmo/releases/tag/v0.1.0)

### Initial release

- Initial set of modules with Julia as main implementation language.
