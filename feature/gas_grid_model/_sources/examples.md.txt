# Examples

## Overview

The following introduces basic examples for using MESMO in the form of tutorials with step-by-step guidance. We address the following simple use cases:

- [Tutorial 1](#tutorial-1): Setting up and solving a multi-period electric grid optimal power flow problem. (This tutorial assumes that a test case scenario has already been defined.)
- [Tutorial 2](#tutorial-2): Defining a test case scenario for an electric distribution grid with several lines, transformers and DERs.
- [Tutorial 3](#tutorial-3): Utilizing the optimization problem interface to define a simple optimization problem.

More examples are in the `examples` directory of the repository and are listed [at the bottom of this section](#more-examples).

## Tutorial 1

### Outline

This example demonstrates the usage of MESMO for setting up and solving a multi-period optimal power flow (OPF) problem for an electric grid with several DERs. We will utilize the test case `sinagpore_6node`, which is shipped with MESMO for this tutorial. Skip to [tutorial 2](#tutorial-2) for learning how to define your own test case scenario.

The multi-period OPF problem is an optimization problem for the maximization of overall social welfare, i.e. minimization of overall costs, for the modeled energy system. The decision variables are the dispatch decisions of DERs, subject to the operational constraints of the DERs and the electric grid.

The tutorial will work through the following steps:

1. Imports and settings: Getting started and selecting a test case scenario.
2. Load data & models: Selecting the MESMO objects that are needed for this use case.
3. Defining and solving the optimization problem: Utilizing API methods for composing the OPF problem.
4. Retrieving and evaluating results: How to access the results and what to do with them.

### Imports and settings

```python
import os
import plotly.graph_objects as go
import mesmo
```

- We will use `plotly` for plotting and `os` for path operations later on.
- Most importantly, we need `import mesmo` to load MESMO. This always loads all submodules and no explicit submodule imports are needed, except if you want to be specific about the MESMO components that you are using.

```python
scenario_name = 'singapore_6node'
results_path = mesmo.utils.get_results_path(__file__, scenario_name)
```

- We use `scenario_name` as an identifier throughout MESMO for selecting a test case scenario. Therefore each test case scenario has a unique `scenario_name`. See [tutorial 2](#tutorial-2) for more information on defining a test case scenario.
- The `get_results_path()` utility function provides a unique timestamped results path, i.e., it creates new folder in the `results` directory of the repository and returns the path. This is intended for convenience, but is entirely optional. Of course, you can also store your results to any other folder.

### Loading data and models

```python
price_data = mesmo.data_interface.PriceData(scenario_name)
linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)
der_model_set = mesmo.der_models.DERModelSet(scenario_name)
```

- We load data & model objects, which are containers for the mathematical models and parameters of the problem. The `scenario_name` is passed to the initialization of each object to identify the test case that should be loaded. For the multi-period OPF problem, we need the following objects:
  - {class}`mesmo.data_interface.PriceData`: Contains the energy price time series and price sensitivity parameter. In `singapore_6node`, this is the Singapore wholesale market price for a representative time period.
  - {class}`mesmo.electric_grid_models.LinearElectricGridModelSet`: Contains the linear approximate electric grid model. Optimization problems in MESMO are convex optimization problems by default, such that we use the linear approximate electric grid model rather than the non-linear default electric grid model {class}`mesmo.electric_grid_models.ElectricGridModelDefault` for this example. The linear approximate electric grid model is essentially a collection of sensitivity matrices that describe the electric grid state variables with respect to the DER power dispatch values. These models are defined as set containing one model per time step, which accommodates different linearization points for each time steps. However, this is not utilized in this tutorial.
  - {class}`mesmo.der_models.DERModelSet`: The DER model set contains all DER models for the test case scenario. These are either 1) times series models for fixed DERs or 2) state space models for flexible DERs.
  - Please refer to the [architecture documentation](architecture.md) for more information on the MESMO module structure.
- Data & model classes are structured into thematic submodules and objects can generally be instantiated for the current test case by simply passing `scenario_name`.

### Defining and solving the optimization problem

```python
optimization_problem = mesmo.solutions.OptimizationProblem()
```

- We first initialize the {class}`mesmo.solutions.OptimizationProblem` object, which serves as container for variables / parameters / constraints / objective of the optimization problem.

```python
linear_electric_grid_model_set.define_optimization_problem(optimization_problem, price_data)
der_model_set.define_optimization_problem(optimization_problem, price_data)
```

- Both linear electric grid and DER model objects provide the `define_optimization_problem()` method for defining the variables, parameters, constraints and objective for their underlying dispatch optimization problem. These definitions are essentially “attached” to `optimization_problem`. The `price_data` object is passed as input to `define_optimization_problem()` because the objective definition depends on the energy price.
- Beyond the default problem variables, parameters, constraints and objective terms, custom definitions can also be manually added to `optimization_problem`. See [tutorial 3](#tutorial-3) for more on this.

```python
optimization_problem.solve()
```

- We call the `solve()` method to invoke the optimization solver. Internally this method first generates the LP/QP standard form and then passes the problem to the solver interface, i.e., direct via `gurobipy` or indirect via `cvxpy`. Please refer to the class documentation of {class}`mesmo.solutions.OptimizationProblem` for more on this.
- The problem solution is stored within `optimization_problem` in terms of variable and dual vectors.

### Retrieving and evaluating results

```python
results = mesmo.problems.Results()
results.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
results.update(der_model_set.get_optimization_results(optimization_problem))
```

- We first instantiate the results object {class}`mesmo.problems.Results` that is a container for `pd.DataFrame` and `pd.Series` objects for individual variable values. For vector variables, the solution values are typically obtained as `pd.DataFrame` with the time steps as rows and variable dimensions as columns, e.g. the nodes of the electric grid for the nodal voltage vector. For scalar variables, the solution values are obtained as `pd.Series` with time steps as rows.
- The solution values are extracted from `optimization_problem` through the `get_optimization_results()` methods of the linear electric grid and DER model objects. Retrieving through the model objects is needed because the variable dimensions are constructed from the index sets that are contained the model objects. The individual results are attached to the `results` object via its `update()` method.

```python
results.save(results_path)
```

- The `save()` method of the `results` object stores all `pd.DataFrame` and `pd.Series` objects as CSV files. Note that `results` also contains the linear electric grid and DER model objects, which are stored as binary PKL files.

```python
figure = go.Figure()
figure.add_scatter(
    x=results.branch_power_magnitude_vector_1.index,
    y=results.branch_power_magnitude_vector_1.loc[:, [('line', '1', 1)]].values.ravel()
)
figure.update_layout(
    title='Branch Power Magnitude at Line 1 (Phase 1)'
)
mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, 'branch_power_line_1_phase_1'))
```

- We use `plotly.graph_objects` for plotting and the `write_figure_plotly()` utility function to store the plotly figure to a file. The file output of `write_figure_plotly()` can be controlled via configuration parameters as described [here](configuration_reference.md#plot-configuration).
- To construct a relative output path for the figure, we use `os.path.join()`. This is the recommended approach for reproducibility rather than hard-coding paths.

## Tutorial 2

### Outline

This example demonstrates the workflow for defining a simple test case scenario and interacting with this scenario though MESMO's problem interfaces. We consider a test case with an electric grid consisting of two nodes and one line, with two DERs connected at the second node. We will utilize pre-defined line type and DER models from the `data/library` directory of the repository. The following properties are assumed:

- Node 0: 22 kV; 3-phase
- Node 1: 22 kV; 3-phase
- Line 1: Node 0 -> Node 1; 3-phase; line type: `singapore_cable_22_copper_300`; length: 0.5 km
- DER 1: Fixed load; model name: `mixed_commercial_residential`
- DER 2: Flexible load; model name: `mixed_commercial_residential`
- Start time step: `2017-01-02T00:00:00`; end time step: `2017-01-03T00:00:00`; time step interval: `00:30:00`

Based on this test case, we will demonstrate the following tasks:
   
1. Solving a nominal operation problem, i.e., a simulation problem.
2. Solving optimal operation problem with a custom peak shaving constraint at 95 % of the peak load from the nominal operation problem.
3. Making plot for the total DER power demand time series to compare nominal vs. optimal dispatch.

### Test case definition

The test case definition workflow can be outlined as follows:

1. Create a new folder in `data` directory for the test case, e.g., named `tutorial_example`.
2. Create the set of CSV files that are needed for the test case scenario definition:
   - `tutorial_example/`: Test case base directory.
   - `tutorial_example/electric_grid_ders.csv`: DER definitions.
   - `tutorial_example/electric_grid_lines.csv`: Electric line definition.
   - `tutorial_example/electric_grid_nodes.csv`: Electric grid node definitions.
   - `tutorial_example/electric_grids.csv`: Electric grid base definition.
   - `tutorial_example/scenarios.csv`: Test case scenario base definition.
3. Fill CSV files based on definition templates and data reference documentation.
   - CSV file templates can be copied from the `data/templates/` directory.
   - CSV files also can be based off existing test cases, e.g., `data/test_case_examples/singapore_6node/`.
   - The CSV file contents and units for numerical values are documented in the [data reference](data_reference.md).

The detailed contents of individual CSV files for this tutorial are omitted here, but the sample definition is included in `data/test_case_examples/tutorial_example` in the MESMO repository.

### Imports and settings

```python
import numpy as np
import os
import plotly.graph_objects as go
import mesmo
```

- We will utilize `numpy` for the numerical operations when constructing the peak load constraint.
- We will use `plotly` for plotting and `os` for path operations later on.
- The `import mesmo` call is needed to load MESMO and all its submodules.

```python
scenario_name = 'tutorial_example'
results_path = mesmo.utils.get_results_path(__file__, scenario_name)
```

- We define `scenario_name` as `'tutorial_example'` to select our newly defined test case scenario.

### Recreate the database

```python
mesmo.data_interface.recreate_database()
```

- The call to `recreate_database()` is needed whenever defining or changing test cases. The CSV files serve as the input format, but an SQLITE database is used internally for data processing. Therefore, changes in the CSV files need to be read into the SQLITE database through the `recreate_database()` function.
- This call is recommended to be included in all run scripts, such that the latest data definitions are always loaded. However, this was omitted above in [tutorial 1](#tutorial-1) for the sake of brevity. 

### Get problem objects

```python
problem_nominal = mesmo.problems.NominalOperationProblem(scenario_name)
problem_optimal = mesmo.problems.OptimalOperationProblem(scenario_name)
```

- We will work with the `mesmo.problems` submodule in this example, as compared to using the more low-level models and data submodules in [tutorial 1](#tutorial-1). The problem classes implement the appropriate setup and solve routines for each problem type. Relevant models and data are automatically obtained depending on the test case definition, e.g., the thermal grid model is only loaded if a thermal grid is defined in the test case scenario.
- The nominal operation problem represents a simulation under „nominal conditions“, i.e., all DERs are dispatched according to their nominal power time series. In this example, we utilize this problem type to represent the status quo, i.e., conventional operation of the energy systems.
- The optimal operation problem represents an optimization for optimal dispatch decisions, subject to the constraints of the DERs, electric grid and thermal grid. For this example, we amend the optimal operation problem definition with a custom constraint to ensure peak load reduction compared to the nominal operation problem.

### Solve nominal operation problem and get results

```python
problem_nominal.solve()
results_nominal = problem_nominal.get_results()
```

- Each problem object exposes `solve()` and `get_results()` methods, which are used here to obtain the solution of the nominal operation problem.
- The `solve()` method of the nominal operation problem invokes the power flow solution classes of electric and thermal grid for non-linear simulation.
- The `get_results()` methods obtains results from the individual models and returns a {class}`mesmo.problems.Results` object.

### Customize and solve optimal operation problem

```python
for timestep in problem_optimal.timesteps:
    problem_optimal.optimization_problem.define_constraint(
        (
            'variable',
            np.array([np.real(problem_optimal.electric_grid_model.der_power_vector_reference)]),
            dict(name='der_active_power_vector', timestep=timestep)
        ),
        '>=',
        (
            'constant',
            0.95 * np.min(np.sum(results_nominal.der_active_power_vector, axis=1))
        )
    )
```

- We define a custom constraint to limit the peak DER demand to 95 % of the peak load from the nominal operation problem. To this end, we compute the total peak demand via result from the nominal operation problem as `np.min(np.sum(results_nominal.der_active_power_vector, axis=1))`. Note that that `der_active_power_vector` takes a negative value for load as per convention in MESMO. Therefore, `np.min()` is needed to compute the peak.
- The total system demand is computed by multiplying the row vector `np.array([np.real(problem_optimal.electric_grid_model.der_power_vector_reference)])` with the optimization variable vector `dict(name='der_active_power_vector', timestep=timestep)`. Note that `der_active_power_vector` in the optimization problem is defined as normalized vector, i.e., each entry is normalized by the nominal / reference active power value of the corresponding DER. In fact, most optimization variables are normalized in this fashion to enable better numerical performance of the optimization solver. Therefore, to obtain the actual active power value, we multiply with the active power reference vector `np.real(problem_optimal.electric_grid_model.der_power_vector_reference)`. Lastly, `np.array([...])` constructs a row vector, such that the resulting multiplication yields a scalar value.
- Here, we directly interface the `optimization_problem` object that is a parameter of the `problem_optimal` object. For more details on the `optimization_problem.define_constraint()` method, please see [tutorial 3](#tutorial-3).

```python
problem_optimal.solve()
results_optimal = problem_optimal.get_results()
```

- The `solve()` method of the optimal operation problem invokes the `solve()` method of the optimal operation problem and passes the problem to the optimization solver.
- Once the solution has terminated successfully, the `get_results()` methods obtains results from the individual models and returns a {class}`mesmo.problems.Results` object.

### Plotting and wrapping things up

```python
figure = go.Figure()
figure.add_trace(go.Scatter(
    x=results_nominal.der_active_power_vector.index,
    y=np.abs(np.sum(results_nominal.der_active_power_vector, axis=1)),
    name='Nominal',
    line=go.scatter.Line(shape='hv')
))
figure.add_trace(go.Scatter(
    x=results_optimal.der_active_power_vector.index,
    y=np.abs(np.sum(results_optimal.der_active_power_vector, axis=1)),
    name='Optimal',
    line=go.scatter.Line(shape='hv')
))
mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, 'comparison'))
```

- We create a plot to compare the DER dispatch schedule between nominal and optimal operation problem.

```python
mesmo.utils.launch(results_path)
print(f"Results are stored in: {results_path}")
```

- The `launch()` function opens a file explorer / finder window for the given path.

## Tutorial 3

### Outline

This example demonstrates the usage of the MESMO optimization problem interface for defining and solving a simple optimization problem. While we only consider a simple stand-alone optimization problem, the interface can also be used to extend default optimization problem definitions with custom variables, constraints and objective terms as demonstrated above in [tutorial 2](#customize-and-solve-optimal-operation-problem). Note that this tutorial does not require any test case scenario definition, as we directly specify the numerical parameters of the problem in the script.

For this tutorial, we consider the following optimization problem:

```{math}
\begin{align}
    \min_{\boldsymbol{a},\boldsymbol{b}} \quad
    & \sum_{i=1}^{n=1000} b_i \\
    \text{s.t.} \quad
    & \boldsymbol{b} = \boldsymbol{a} \cdot \boldsymbol{P} \\
    & -10 \leq \boldsymbol{a} \leq +10
\end{align}
```

The matrix {math}`\boldsymbol{P} \in \mathbb{R}^{n \times n}` is an abitrary parameter matrix. The vectors  {math}`\boldsymbol{a}, \boldsymbol{b} \in \mathbb{R}^{n \times 1}` are decision variable vectors. The symbol {math}`n` defines the problem dimension.

### Imports and setting up

```python
import numpy as np
import mesmo
```

- We will utilize `numpy` for the random parameter matrix definition.
- As usual, `import mesmo` is needed to load MESMO and all its submodules.

```python
dimension = 1000
parameter_matrix = np.random.rand(dimension, dimension)
```

- We begin by defining the problem dimension and generating the `parameter_matrix` of appropriate size. We utilize `np.random.rand()` to obtain a matrix filled with random values.
- Note that for the optimization problem interface, accepted numerical values for parameter, constraint or objective definitions are 1) float values, 2) numpy arrays or 3) scipy sparse matrices.

```python
optimization_problem = mesmo.solutions.OptimizationProblem()
```

- We instantiate the optimization problem object serves as a container for the parameters, variables, constraints and objective terms.
- As documented at {class}`mesmo.solutions.OptimizationProblem`, the optimization problem objects exposes methods for problem setup & solution, which are utilized in the following.

### Defining parameters

```python
optimization_problem.define_parameter('parameter_matrix', parameter_matrix)
```

- Defining parameters is optional because numerical values can also be directly passed in the constraints and objective definitions. However, using parameters allows updating the numerical values of the problem without re-defining the complete problem.

### Defining variables

```python
optimization_problem.define_variable('a_vector', a_index=range(dimension))
optimization_problem.define_variable('b_vector', b_index=range(dimension))
```

- Variables are defined by passing a name string and index key sets. The variable dimension is determined by the dimension of the index key sets. Accepted key set values are 1) lists, 2) tuples, 3) numpy arrays, 4) pandas index objects and 5) range objects.
- If multiple index key sets are passed, the variable dimension is determined as the cartesian product of the key sets. However, note that variables always take the shape of column vectors in constraint and objective definitions. That means, multiple key sets are not interpreted as array dimensions.

### Defining constraints

```python
optimization_problem.define_constraint(
    ('variable', 1.0, dict(name='b_vector')),
    '==',
    ('variable', 'parameter_matrix', dict(name='a_vector')),
)
optimization_problem.define_constraint(
    ('constant', -10.0),
    '<=',
    ('variable', 1.0, dict(name='a_vector')),
)
optimization_problem.define_constraint(
    ('constant', +10.0),
    '>=',
    ('variable', 1.0, dict(name='a_vector')),
)
```

- Constraints are defined as list of tuples and strings, where tuples are either 1) variable terms or 2) constant terms and strings represent operators (`==`, `<=` or `>=`). If multiple variable and constant terms are on either side of the operator, these are interpreted as summation of the variables / constants.
- Constant terms are tuples in the form `('constant', numerical value)`, where the numerical value can be 1) float value, 2) numpy array, 3) scipy sparse matrix or 4) a parameter name string. The numerical value is expected to represent a column vector with appropriate size matching the constraint dimension. If a float value is given as numerical value, the value is multiplied with a column vector of ones of appropriate size.
- Variable terms are tuples in the form `('variable', numerical factor, dict(name=variable name, keys...))`, where the numerical factor can be 1) float value, 2) numpy array, 3) scipy sparse matrix or 4) a parameter name string. The numerical factor is multiplied with the variable vector and is expected to represent a matrix of appropriate size for the multiplication. If a float value is given as numerical factor, the value is multiplied with a identity matrix of appropriate size. Keys can be optionally given to select / slice a portion of the variable vector. Note that variables always take the shape of column vectors.

### Defining objective terms

```python
optimization_problem.define_objective(('variable', 1.0, dict(name='b_vector')))
```

- Objective terms are defined as list of tuples, where tuples are either 1) variable terms or 2) constant terms. Each term is expected to evaluate to a scalar value. If multiple variable and constant terms are defined, these are interpreted as summation of the variables / constants.
- Constant terms are tuples in the form `('constant', numerical value)`, where the numerical value can be 1) float value or 2) a parameter name string.
- Variable terms are tuples in the form `('variable', numerical factor, dict(name=variable name, keys...))`, where the numerical factor can be 1) float value, 2) numpy array, 3) scipy sparse matrix or 4) a parameter name string. The numerical factor is multiplied with the variable vector and is expected to represent a matrix of appropriate size for the multiplication, such that the multiplication evaluates to a scalar. If a float value is given as numerical factor, the value is multiplied with a row vector of ones of appropriate size. Keys can be optionally given to select / slice a portion of the variable vector. Note that variables always take the shape of column vectors.

### Solving and retrieving results

```python
optimization_problem.solve()
```

- Calling the `solve()` method compiles the standard form of the linear program and passes it the optimization solver, e.g. Gurobi.
- The solve method currently implements interfaces to 1) Gurobi and 2) CVXPY, where the latter is a high-level convex optimization interface, which in turn allows interfacing further third-party solvers. The intention is to implement more direct solver interfaces on as-need basis (please raise an [issue](https://github.com/mesmo-dev/mesmo/issues)), as these interfaces are assumed to allow higher performance than CVXPY for large-scale problems. However, CVXPY is kept as a fallback to allow a high degree of compatibility with various solvers.

```python
results = optimization_problem.get_results()
a_vector = results['a_vector']
b_vector = results['b_vector']
```

- Results are returned as dictionary with keys corresponding to the variable names that have been defined. The variable values are returned as `pd.DataFrame` with time steps as rows and other variable key dimensions as multi-level column index. For variables with only time steps as key dimension, the values are returned as `pd.Series` with time steps as rows.
- Note that the `results` dictionary of the optimization problem object is different from the {class}`mesmo.problems.Results` object that is returned from the `get_optimization_results()` methods of the model objects. Particularly, some variable dimensions are not appropriately reconstructed in the `results` dictionary of the optimization problem object. Therefore, the {class}`mesmo.problems.Results` object is preferred when working with the default models. 

## More examples

The `examples` directory of the repository contains the following run scripts.

### API examples

These examples demonstrate the usage of the high-level API to execute predefined problem types.

- `run_api_nominal_operation_problem.py`: Example script for setting up and solving an nominal operation problem. The nominal operation problem (alias: power flow problem, electric grid simulation problem) formulates the steady-state power flow problem for all timesteps of the given scenario subject to the nominal operation schedule of all DERs.
- `run_api_optimal_operation_problem.py`: Example script for setting up and solving an optimal operation problem. The optimal operation problem (alias: optimal dispatch problem, optimal power flow problem) formulates the optimization problem for minimizing the objective functions of DERs and grid operators subject to the model constraints of all DERs and grids.

### Advanced examples

For advanced usage of MESMO, the following examples demonstrate in a step-by-step manner how energy system models and optimization problems can be defined and solved with MESMO. These example scripts serve as a reference for setting up custom work flows.

- `run_electric_grid_optimal_operation.py`: Example script for setting up and solving an electric grid optimal operation problem.
- `run_thermal_grid_optimal_operation.py`: Example script for setting up and solving a thermal grid optimal operation problem.
- `run_multi_grid_optimal_operation.py`: Example script for setting up and solving a multi-grid optimal operation problem.
- `run_flexible_der_optimal_operation.py`: Example script for setting up and solving a flexible DER optimal operation problem.
- `run_electric_grid_power_flow_single_step.py`: Example script for setting up and solving an single step electric grid power flow problem.

### Validation scripts

Since the model implementations in MESMO are not infallible, these validation scripts are provided for model testing.

- `validation_electric_grid_power_flow.py`: Example script for testing / validating the electric grid power flow solution.
- `validation_linear_electric_grid_model.py`: Example script for testing / validating the linear electric grid model.
- `validation_electric_grid_dlmp_solution.py`: Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem.

### Other scripts

- The directory `examples/development` contains example scripts which are under development and scripts related to the development of new features for MESMO.
- The directory `examples/publications` contains scripts related to publications which based on MESMO.
