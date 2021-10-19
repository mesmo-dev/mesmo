# Configuration

The MESMO configuration is defined via `config.yml`. As an initial user, you likely do not need to modify the configuration.

## Configuration workflow

An empty `config.yml` is created during the first runtime of MESMO. To define the local configuration, simply insert and modify configuration parameters from the references below.

![](../assets/configuration_file_structure.png)

**Figure: Configuration file location.**

MESMO distinguishes 1) local configuration in `config.yml` and 2) default configuration in `config_default.yml`. The local configuration takes precedence over the default configuration. That means, during initialization, `get_config()` first reads the default configuration from `config_default.yml` and then redefines any parameters that have been modified in `config.yml`.

![](../assets/configuration_workflow.png)

**Figure: Configuration initialization workflow.**

The following sections serve as a reference for configuration parameters. Please note that key / value pairs are defined in a nested structure for each configuration type. This nested structure needs to be replicated for key / value pairs in `config.yml`.

```{important}
Please only modify `config.yml`, but do not make changes to `config_default.yml`.
```

## Path configuration

### Default values

```yaml
paths:
  data: ./data
  database: ./data/database.sqlite
  results: ./results
  additional_data: []
  ignore_data_folders: []
  cobmo_additional_data: []
```

### Configuration keys

 - `data`: Defines the main data directory, i.e. the location of the CSV input files which are imported by {func}`mesmo.data_interface.recreate_database()`. Can be given as absolute path or relative path to `./`¹. Defaults to the data directory that is included with the MESMO repository. If you want to include additional data from other directories, please see `additional_data` below.
 - `database`: Defines the file path for the internal SQLITE database. Can be given as absolute path or relative path to `./`¹. This file will be created by {func}`mesmo.data_interface.recreate_database()`, if it does not exist.
 - `results`: Defines the main results directory, i.e. the directory where results outputs are stored. This parameter is used as base path in {func}`mesmo.utils.get_results_path()`. Defaults to the results directory in the MESMO repository.
 - `additional_data`: Defines list of supplementary data directories, which are imported in addition to the main data directory by {func}`mesmo.data_interface.recreate_database()`. Should be defined as list of absolute or relative paths to `./`¹.
 - `ignore_data_folders`: Defines a list of directory names that are excluded during import by {func}`mesmo.data_interface.recreate_database()`. Should be defined as list of folder names to exclude, but does accept full paths.
 - `cobmo_additional_data`: Defines list of supplementary data directories for the `cobmo` submodule, similar to `additional_data` above.

¹ In the path parameters, `./` denotes the MESMO repository base directory as reference for relative path definitions.

### Using additional data directories

As an example, additional data from folders `supplementary_data` and `project_data` adjacent to the MESMO repository base directory can be defined with the following configuration snippet:

```yaml
paths:
  additional_data: [
    ./../supplementary_data,
    ./../project_data
  ]
```

## Optimization solver configuration

### Default values

```yaml
optimization:
  solver_name: gurobi
  solver_interface: direct
  show_solver_output: true
  time_limit:
```

### Configuration keys

 - `solver_name`: Defines the optimization solver. Choices are 'gurobi' or any valid solver name [for CVXPY](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver). Solver name should be defined in lower caps.
 - `solver_interface`: Defines the interface for sending the optimization problem to the solver. Choices are 'direct' or 'cvxpy'. If 'direct', MESMO will use a direct solver interface, which is currently only implemented for Gurobi. If no direct solver interface is available for the selected solver, MESMO will automatically fall back to CVXPY. If 'cvxpy', MESMO will always use CVXPY without checking for a direct solver interface. If not defined, will use 'direct'.
 - `time_limit`: Solver time limit in seconds. If not defined, the value is set to infinite. Currently only implemented for Gurobi and CPLEX.
 - `show_solver_output`: Choices are 'true' or 'false'. If 'true', activate verbose solver output. If 'false', silence any solver outputs.

### Setting CPLEX as optimization solver

As an example, [CPLEX](https://www.ibm.com/analytics/cplex-optimizer) can be defined as the optimization solver with the following configuration snippet:

```yaml
optimization:
  solver_name: cplex
  solver_interface: cvxpy
```

Note that CPLEX is interfaced through CVXPY, because there is currently no direct interface implemented in MESMO. This requires CPLEX to be installed [as per CVXPY instructions](https://www.cvxpy.org/install/index.html#install-with-cplex-support).
