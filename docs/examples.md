# Examples

The `examples` directory contains run scripts which demonstrate the usage of MESMO.

## API examples

These examples demonstrate the usage of the high-level API to execute predefined problem types.

- `run_api_nominal_operation_problem.py`: Example script for setting up and solving an nominal operation problem. The nominal operation problem (alias: power flow problem, electric grid simulation problem) formulates the steady-state power flow problem for all timesteps of the given scenario subject to the nominal operation schedule of all DERs.
- `run_api_optimal_operation_problem.py`: Example script for setting up and solving an optimal operation problem. The optimal operation problem (alias: optimal dispatch problem, optimal power flow problem) formulates the optimization problem for minimizing the objective functions of DERs and grid operators subject to the model constraints of all DERs and grids.

## Advanced examples

For advanced usage of MESMO, the following examples demonstrate in a step-by-step manner how energy system models and optimization problems can be defined and solved with MESMO. These example scripts serve as a reference for setting up custom work flows.

- `run_electric_grid_optimal_operation.py`: Example script for setting up and solving an electric grid optimal operation problem.
- `run_thermal_grid_optimal_operation.py`: Example script for setting up and solving a thermal grid optimal operation problem.
- `run_multi_grid_optimal_operation.py`: Example script for setting up and solving a multi-grid optimal operation problem.
- `run_flexible_der_optimal_operation.py`: Example script for setting up and solving a flexible DER optimal operation problem.
- `run_electric_grid_power_flow_single_step.py`: Example script for setting up and solving an single step electric grid power flow problem.

## Validation scripts

Since the model implementations in MESMO are not infallible, these validation scripts are provided for model testing.

- `validation_electric_grid_power_flow.py`: Example script for testing / validating the electric grid power flow solution.
- `validation_linear_electric_grid_model.py`: Example script for testing / validating the linear electric grid model.
- `validation_electric_grid_dlmp_solution.py`: Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem.

## Other scripts

- The directory `examples/development` contains example scripts which are under development and scripts related to the development of new features for MESMO.
- The directory `examples/publications` contains scripts related to publications which based on MESMO.
