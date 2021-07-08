"""Utility functions module."""

import copy
import cvxpy as cp
import gurobipy as gp
import datetime
import functools
import glob
import itertools
import logging
import numpy as np
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import re
import time
import typing
import scipy.sparse
import subprocess
import sys

import cobmo.building_model
import fledge.config

logger = fledge.config.get_logger(__name__)

# Instantiate dictionary for execution time logging.
log_times = dict()


class ObjectBase(object):
    """FLEDGE object base class, which extends the Python object base class.

    - Requires all attributes, i.e. parameters or object variables, to be defined with type declaration at the
      beginning of the class definition. Setting a value to an attribute which has not been defined will raise an error.
      This is to ensure consistent definition structure of FLEDGE classes.
    - String representation of the object is the concatenation of the string representation of all its attributes.
      Thus, printing the object will print all its attributes.

    Example:

        Attributes should be defined in the beginning of the class definition as follows::

            class ExampleClass(ObjectBase):

                example_attribute1: str
                example_attribute2: pd.DataFrame

        In this case, ``example_attribute1`` and ``example_attribute2`` are valid attributes of the class.
    """

    def __setattr__(
            self,
            attribute_name,
            value
    ):

        # Assert that attribute name is valid.
        # - Valid attributes are those which are defined as results class attributes with type declaration.
        if not (attribute_name in typing.get_type_hints(type(self))):
            raise AttributeError(
                f"Cannot set invalid attribute '{attribute_name}'. "
                f"Please ensure that the attribute has been defined with type declaration in the class definition."
            )

        # Set attribute value.
        super().__setattr__(attribute_name, value)

    def __repr__(self) -> str:
        """Obtain string representation."""

        # Obtain attributes.
        attributes = vars(self)

        # Obtain representation string.
        repr_string = ""
        for attribute_name in attributes:
            repr_string += f"{attribute_name} = \n{attributes[attribute_name]}\n"

        return repr_string

    def copy(self):
        """Return a copy of this object. A new object will be created with a copy of the calling object’s attributes.
        Modifications to the attributes of the copy will not be reflected in the original object.
        """

        return copy.deepcopy(self)


class ResultsBase(ObjectBase):
    """Results object base class."""

    def __init__(
            self,
            **kwargs
    ):

        # Set all keyword arguments as attributes.
        for attribute_name in kwargs:
            self.__setattr__(attribute_name, kwargs[attribute_name])

    def __getitem__(self, key):
        # Enable dict-like attribute getting.
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        # Enable dict-like attribute setting.
        self.__setattr__(key, value)

    def update(
            self,
            other_results
    ):

        # Obtain attributes of other results object.
        attributes = vars(other_results)

        # Update attributes.
        # - Existing attributes are overwritten with values from the other results object.
        for attribute_name in attributes:
            if attributes[attribute_name] is not None:
                self.__setattr__(attribute_name, attributes[attribute_name])

    def save(
            self,
            results_path: str
    ):
        """Store results to files at given results path.

        - Each results variable / attribute will be stored as separate file with the attribute name as file name.
        - Pandas Series / DataFrame are stored to CSV.
        - Other objects are stored to pickle binary file (PKL).
        """

        # Obtain results attributes.
        attributes = vars(self)

        # Store each attribute to a separate file.
        for attribute_name in attributes:
            if type(attributes[attribute_name]) in (pd.Series, pd.DataFrame):
                # Pandas Series / DataFrame are stored to CSV.
                attributes[attribute_name].to_csv(os.path.join(results_path, f'{attribute_name}.csv'))
            else:
                # Other objects are stored to pickle binary file (PKL).
                with open(os.path.join(results_path, f'{attribute_name}.pkl'), 'wb') as output_file:
                    pickle.dump(attributes[attribute_name], output_file, pickle.HIGHEST_PROTOCOL)

    def load(
            self,
            results_path: str
    ):
        """Load results from given path."""

        # Obtain all CSV and PKL files at results path.
        files = glob.glob(os.path.join(results_path, '*.csv')) + glob.glob(os.path.join(results_path, '*.pkl'))

        # Load all files which correspond to valid attributes.
        for file in files:

            # Obtain file extension / attribute name.
            file_extension = os.path.splitext(file)[1]
            attribute_name = os.path.basename(os.path.splitext(file)[0])

            # Load file and set attribute value.
            if attribute_name in typing.get_type_hints(type(self)):
                if file_extension.lower() == '.csv':
                    value = pd.read_csv(file)
                else:
                    with open(file, 'rb') as input_file:
                        value = pickle.load(input_file)
                self.__setattr__(attribute_name, value)
            else:
                # Files which do not match any valid results attribute are not loaded.
                logger.debug(f"Skipping results file which does match any valid results attribute: {file}")

        return self


class OptimizationProblem(object):
    """Optimization problem object for use with CVXPY."""

    constraints: list
    objective: cp.Expression
    has_der_objective: bool = False
    has_electric_grid_objective: bool = False
    has_thermal_grid_objective: bool = False
    cvxpy_problem: cp.Problem

    def __init__(self):

        self.constraints = []
        self.objective = cp.Constant(value=0.0)

    def solve(
            self,
            keep_problem=False,
            **kwargs
    ):

        # Instantiate CVXPY problem object.
        if hasattr(self, 'cvxpy_problem') and keep_problem:
            pass
        else:
            self.cvxpy_problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

        # Solve optimization problem.
        self.cvxpy_problem.solve(
            solver=(
                fledge.config.config['optimization']['solver_name'].upper()
                if fledge.config.config['optimization']['solver_name'] is not None
                else None
            ),
            verbose=fledge.config.config['optimization']['show_solver_output'],
            **kwargs,
            **fledge.config.solver_parameters
        )

        # Assert that solver exited with an optimal solution. If not, raise an error.
        if not (self.cvxpy_problem.status == cp.OPTIMAL):
            raise cp.SolverError(f"Solver termination status: {self.cvxpy_problem.status}")


class StandardForm(object):
    """Standard form object for linear program, with objective ``min(c @ x)`` and constraints ``A @ x <= b``."""

    variables: pd.DataFrame
    constraints: pd.DataFrame
    constraints_len: int
    a_dict: dict
    b_dict: dict
    c_dict: dict
    c_constant: float
    x_vector: np.ndarray
    dual_vector: np.ndarray

    def __init__(self):

        # Instantiate index sets.
        # - Variables are instantiated with 'name' and 'timestep' keys, but more may be added in ``define_variable()``.
        # - Constraints are instantiated with 'name', 'timestep' and 'constraint_type' keys,
        #   but more may be added in ``define_constraint()``.
        self.variables = pd.DataFrame(columns=['name', 'timestep'])
        self.constraints = pd.DataFrame(columns=['name', 'timestep', 'constraint_type'])
        self.constraints_len = 0

        # Instantiate A matrix / b vector / c vector dictionaries.
        # - Final matrix / vector are only created in ``get_a_matrix()``, ``get_b_vector()``, ``get_c_vector()``.
        self.a_dict = dict()
        self.b_dict = dict()
        self.c_dict = dict()

        # Instantiate c constant base.
        self.c_constant = 0.0

    def define_variable(
            self,
            name: str,
            **keys
    ):

        # Obtain new variables based on ``keys``.
        new_variables = (
            pd.DataFrame(itertools.product([name], *[
                list(value)
                if type(value) in [pd.MultiIndex, pd.Index, pd.DatetimeIndex, list, tuple]
                else [value]
                for value in keys.values()
            ]), columns=['name', *keys.keys()])
        )
        # Add new variables to index.
        self.variables = pd.concat([self.variables, new_variables], ignore_index=True)

    def define_constraint(
            self,
            *elements: typing.Union[str, typing.Union[
                typing.Tuple[str, typing.Union[float, np.ndarray, scipy.sparse.spmatrix]],
                typing.Tuple[str, typing.Union[float, np.ndarray, scipy.sparse.spmatrix], dict]
            ]],
            **kwargs
    ):

        # Instantiate constraint element aggregation variables.
        variables = []
        constant = 0.0
        operator = None

        # Instantiate left-hand / right-hand side indicator. Starting from left-hand side.
        side = 'left'

        # Aggregate constraint elements.
        for element in elements:

            # Tuples are variables / constants.
            if issubclass(type(element), tuple):

                # Identify variables.
                if element[0] in ('variable', 'var', 'v'):

                    # Move right-hand variables to left-hand side.
                    if side == 'right':
                        factor = -1.0
                    else:
                        factor = 1.0

                    # Append element to variables.
                    variables.append((factor * element[1], element[2]))

                # Identify constants.
                elif element[0] in ('constant', 'con', 'c'):

                    # Move left-hand constants to right-hand side.
                    if side == 'left':
                        factor = -1.0
                    else:
                        factor = 1.0

                    # Add element to constant.
                    constant += factor * element[1]

                # Raise error if element type cannot be identified.
                else:
                    raise ValueError(f"Invalid constraint element type: {element[0]}")

            # Strings are operators.
            elif element in ['==', '<=', '>=']:

                # Raise error if operator is first element.
                if element == elements[0]:
                    ValueError(f"Operator is first element of a constraint.")

                # Raise error if operator is last element.
                if element == elements[-1]:
                    ValueError(f"Operator is last element of a constraint.")

                # Raise error if operator is already defined.
                if operator is not None:
                    ValueError(f"Multiple operators defined in one constraint.")

                # Set operator.
                operator = element

                # Update left-hand / right-hand side indicator. Moving to right-hand side.
                side = 'right'

            # Raise error if element type cannot be identified.
            else:
                raise ValueError(f"Invalid constraint element type: {element}")

        self.define_constraint_low_level(
            variables,
            operator,
            constant,
            **kwargs
        )

    def define_constraint_low_level(
            self,
            variables: typing.List[typing.Tuple[typing.Union[float, np.ndarray, scipy.sparse.spmatrix], dict]],
            operator: str,
            constant: typing.Union[float, np.ndarray, scipy.sparse.spmatrix],
            keys: dict = None,
            broadcast: str = None
    ):

        # Run checks for constraint index keys.
        if keys is not None:

            # Raise error if ``keys`` is not a dictionary.
            if type(keys) is not dict:
                raise TypeError(f"Constraint `keys` parameter must be a dictionary, but instead is: {type(keys)}")

            # Raise error if no 'name' key was defined.
            if 'name' not in keys.keys():
                raise ValueError(f"'name' key is required in constraint `keys` dictionary. Only found: {keys.keys()}")

            # TODO: Raise error if using reserved 'constraint_type' key.

        # For equality constraint, convert to upper / lower inequality.
        if operator in ['==']:

            # Define upper inequality.
            self.define_constraint_low_level(
                variables,
                '>=',
                constant,
                keys=dict(keys, constraint_type='==>=') if keys is not None else None,
                broadcast=broadcast
            )

            # Define lower inequality.
            self.define_constraint_low_level(
                variables,
                '<=',
                constant,
                keys=dict(keys, constraint_type='==<=') if keys is not None else None,
                broadcast=broadcast
            )

        # For inequality constraint, add into A matrix / b vector dictionaries.
        elif operator in ['<=', '>=']:

            # If greater-than-equal, invert signs.
            if operator == '>=':
                factor = -1.0
            else:
                factor = 1.0

            # Raise error if constant is not a scalar, column vector (n, 1) or flat array (n, ).
            if type(constant) is not float:
                if len(np.shape(constant)) > 1:
                    if np.shape(constant)[1] > 1:
                        raise ValueError(f"Constant must be column vector (n, 1), not row vector (1, n).")

            # Obtain constant dimension.
            if type(constant) is not float:
                dimension_constant = len(constant)
            else:
                dimension_constant = 1

            # Obtain constraint integer index based on constant dimension.
            constraint_index = tuple(range(self.constraints_len, self.constraints_len + dimension_constant))

            # Append b vector entry.
            self.b_dict[constraint_index] = factor * constant

            # Append A matrix entries.
            for variable in variables:

                # If any variable key values are empty, ignore variable & do not add any A matrix entry.
                for key_value in variable[1].values():
                    if isinstance(key_value, (list, tuple, pd.MultiIndex, pd.Index, np.ndarray)):
                        if len(key_value) == 0:
                            continue  # Skip variable & go to next iteration.

                # Obtain variable integer index & raise error if variable or key does not exist.
                variable_index = (
                    tuple(fledge.utils.get_index(self.variables, **variable[1], raise_empty_index_error=True))
                )

                # Obtain A matrix entries.
                # - Scalar values are multiplied with identity matrix of appropriate size.
                # - If broadcasting, value is repeated in block-diagonal matrix.
                if len(np.shape(variable[0])) == 0:
                    a_entry = variable[0] * scipy.sparse.eye(len(variable_index))
                else:
                    a_entry = variable[0]
                    # If broadcasting, value is repeated in block-diagonal matrix.
                    if broadcast is not None:
                        if broadcast not in variable[1].keys():
                            raise ValueError(f"Invalid broadcast dimension: {broadcast}")
                        else:
                            # TODO: Need check for order of values / index entries?
                            if type(a_entry) is np.matrix:
                                a_entry = np.array(a_entry)
                            a_entry = scipy.sparse.block_diag([a_entry] * len(variable[1][broadcast]))

                # Raise error if variable dimensions are inconsistent.
                if np.shape(a_entry) != (len(constraint_index), len(variable_index)):
                    raise ValueError(f"Dimension mismatch at variable: {variable[1]}")

                # Add A matrix entries to dictionary.
                self.a_dict[constraint_index, variable_index] = factor * a_entry

            # Append constraints index entries.
            if keys is not None:
                # Set constraint type:
                if 'constraint_type' in keys.keys():
                    if keys['constraint_type'] not in ('==>=', '==<='):
                        keys['constraint_type'] = operator
                else:
                    keys['constraint_type'] = operator
                # Obtain new constraints based on ``keys``.
                new_constraints = (
                    pd.DataFrame(itertools.product(*[
                        list(value)
                        if type(value) in [pd.MultiIndex, pd.Index, pd.DatetimeIndex, list, tuple]
                        else [value]
                        for value in keys.values()
                    ]), columns=keys.keys())
                )
                # Raise error if key set dimension does not align with constant dimension.
                if len(new_constraints) != dimension_constant:
                    raise ValueError(
                        f"Constraint key set dimension ({len(new_constraints)})"
                        f" does not align with constant dimension ({dimension_constant})."
                    )
                # Add new constraints to index.
                new_constraints.index = constraint_index
                self.constraints = self.constraints.append(new_constraints)
                self.constraints_len += len(constraint_index)
            else:
                # Only change constraints size, if no ``keys`` defined.
                self.constraints_len += len(constraint_index)

        # Raise error for invalid operator.
        else:
            ValueError(f"Invalid constraint operator: {operator}")

    def define_objective_low_level(
            self,
            variables: typing.List[typing.Tuple[typing.Union[float, np.ndarray, scipy.sparse.spmatrix], dict]],
            constant: float
    ):

        # Raise error if constant is not a scalar (1, ) or (1, 1) or float.
        if type(constant) is not float:
            if np.shape(constant) not in [(1, ), (1, 1)]:
                raise ValueError(f"Objective constant must be scalar (1, 1).")

        # Add c constant value.
        self.c_constant += constant

        # Append c vector entries.
        for variable in variables:

            # If any variable key values are empty, ignore variable & do not add any c vector entry.
            for key_value in variable[1].values():
                if isinstance(key_value, (list, tuple, pd.Index, np.ndarray)):
                    if len(key_value) == 0:
                        continue  # Skip variable & go to next iteration.

            # Obtain variable index & raise error if variable or key does not exist.
            variable_index = (
                tuple(fledge.utils.get_index(self.variables, **variable[1], raise_empty_index_error=True))
            )

            # Obtain c vector entries.
            # - Scalar values are multiplied with row vector of ones of appropriate size.
            if len(np.shape(variable[0])) == 0:
                c_entry = variable[0] * np.ones((1, len(variable_index)))
            else:
                c_entry = variable[0]

            # Raise error if vector is not a row vector (1, n) or flat array (n, ).
            if len(np.shape(constant)) > 1:
                if np.shape(constant)[0] > 1:
                    raise ValueError(
                        f"Objective factor must be row vector (1, n), not column vector (n, 1) nor matrix (m, n)."
                    )

            # Raise error if variable dimensions are inconsistent.
            if (
                    (np.shape(c_entry)[1] != len(variable_index)) or (np.shape(c_entry)[0] != 1)
                    if len(np.shape(c_entry)) > 1
                    else np.shape(c_entry)[0] != len(variable_index)
            ):
                raise ValueError(f"Objective factor dimension mismatch at variable: {variable[1]}")

            # Add c vector entries to dictionary.
            self.c_dict[variable_index] = c_entry.ravel()

    def get_a_matrix(self) -> scipy.sparse.spmatrix:

        # Log time.
        log_time('get standard-form A matrix')

        # Instantiate collections.
        values_list = []
        rows_list = []
        columns_list = []

        # Collect matrix entries.
        for constraint_index, variable_index in self.a_dict:
            rows, columns, values = scipy.sparse.find(self.a_dict[constraint_index, variable_index])
            rows = np.array(constraint_index)[rows]
            columns = np.array(variable_index)[columns]
            values_list.append(values)
            rows_list.append(rows)
            columns_list.append(columns)

        # Instantiate sparse matrix.
        a_matrix = (
            scipy.sparse.coo_matrix(
                (np.concatenate(values_list), (np.concatenate(rows_list), np.concatenate(columns_list))),
                shape=(self.constraints_len, len(self.variables))
            ).tocsr(copy=True)  # TODO: Is copy really needed here?
        )

        # Log time.
        log_time('get standard-form A matrix')

        return a_matrix

    def get_b_vector(self) -> np.ndarray:

        # Log time.
        log_time('get standard-form b vector')

        # Instantiate array.
        b_vector = np.zeros((self.constraints_len, 1))

        # Fill vector entries.
        for constraint_index in self.b_dict:
            b_vector[np.ix_(constraint_index), 0] += self.b_dict[constraint_index]

        # Log time.
        log_time('get standard-form b vector')

        return b_vector

    def get_c_vector(self) -> np.ndarray:

        # Log time.
        log_time('get standard-form c vector')

        # Instantiate array.
        c_vector = np.zeros((1, len(self.variables)))

        # Fill vector entries.
        for variable_index in self.c_dict:
            c_vector[0, np.ix_(variable_index)] += self.c_dict[variable_index]

        # Log time.
        log_time('get standard-form c vector')

        return c_vector

    def solve(self):

        if fledge.config.config['optimization']['solver_name'] == 'gurobi':
            self.solve_gurobi()
        else:
            self.solve_cvxpy()

    def solve_gurobi(self):

        # Instantiate Gurobi model.
        # - A Gurobi model holds a single optimization problem. It consists of a set of variables, a set of constraints,
        #   and the associated attributes.
        gurobipy_problem = gp.Model()
        gurobipy_problem.setParam('OutputFlag', int(fledge.config.config['optimization']['show_solver_output']))

        # Define variables.
        # - Need to express vectors as 1-D arrays to enable matrix multiplication in constraints (gurobipy limitation).
        # - Lower bound defaults to 0 and needs to be explicitly overwritten.
        x_vector = (
            gurobipy_problem.addMVar(
                shape=(len(self.variables), ),
                lb=-np.inf,
                ub=np.inf,
                vtype=gp.GRB.CONTINUOUS,
                name='x_vector'
            )
        )

        # Define constraints.
        # - 1-D arrays are interpreted as column vectors (n, 1) (based on gurobipy convention).
        constraints = self.get_a_matrix() @ x_vector <= self.get_b_vector().ravel()
        constraints = gurobipy_problem.addConstr(constraints, name='constraints')
        # TODO: Use alternative / explicit expression or not?
        # gurobipy_problem.addMConstr(
        #     A=self.get_a_matrix(),
        #     x=x_vector,
        #     sense='<=',
        #     b=b_vector,
        #     name='constraints'
        # )

        # Define objective.
        # - 1-D arrays are interpreted as column vectors (n, 1) (based on gurobipy convention).
        objective = self.get_c_vector().ravel() @ x_vector
        gurobipy_problem.setObjective(objective, gp.GRB.MINIMIZE)
        # TODO: Use alternative / explicit expression or not?
        # gurobipy_problem.setMObjective(
        #     Q=None,
        #     c=self.get_c_vector().ravel(),
        #     constant=0.0,
        #     xQ_L=None,
        #     xQ_R=None,
        #     xc=x_vector,
        #     sense=gp.GRB.MINIMIZE
        # )

        # Solve optimization problem.
        gurobipy_problem.optimize()

        # Store results.
        self.x_vector = np.transpose([x_vector.getAttr('x')])
        self.dual_vector = np.transpose([constraints.getAttr('Pi')])

    def solve_cvxpy(self):

        # Define variables.
        x_vector = cp.Variable((len(self.variables), 1))

        # Define constraints.
        constraints = [self.get_a_matrix() @ x_vector <= self.get_b_vector()]

        # Define objective.
        objective = self.get_c_vector() @ x_vector

        # Instantiate CVXPY problem.
        cvxpy_problem = cp.Problem(cp.Minimize(objective), constraints)

        # Solve optimization problem.
        cvxpy_problem.solve(
            solver=(
                fledge.config.config['optimization']['solver_name'].upper()
                if fledge.config.config['optimization']['solver_name'] is not None
                else None
            ),
            verbose=fledge.config.config['optimization']['show_solver_output'],
            **fledge.config.solver_parameters
        )

        # Assert that solver exited with an optimal solution. If not, raise an error.
        if not (cvxpy_problem.status == cp.OPTIMAL):
            raise cp.SolverError(f"Solver termination status: {cvxpy_problem.status}")

        # Store results.
        self.x_vector = x_vector.value
        self.dual_vector = constraints[0].dual_value

    def get_results(
        self,
        x_vector: typing.Union[cp.Variable, np.ndarray] = None
    ) -> dict:

        # Obtain x vector.
        if x_vector is None:
            x_vector = self.x_vector
        elif type(x_vector) is cp.Variable:
            x_vector = x_vector.value

        # Instantiate results object.
        results = {}

        # Obtain results for each variable.
        for name in self.variables.loc[:, 'name'].unique():

            # Obtain indexes.
            variable_index = fledge.utils.get_index(self.variables, name=name)
            timesteps = self.variables.loc[variable_index, 'timestep'].unique()
            columns = (
                self.variables.loc[variable_index, :].drop([
                    'name', 'timestep'
                ], axis=1).drop_duplicates().dropna(axis=1)
            )
            if len(columns.columns) > 0:
                columns = pd.MultiIndex.from_frame(columns)
            else:
                columns = pd.Index(['total'])

            # Instantiate results dataframe.
            results[name] = pd.DataFrame(index=timesteps, columns=columns)

            # Get results.
            for timestep in timesteps:
                results[name].loc[timestep, :] = (
                    x_vector[fledge.utils.get_index(self.variables, name=name, timestep=timestep), 0]
                )

        return results

    def get_duals(self) -> dict:

        # Instantiate results object.
        results = {}

        # Obtain results for each variable.
        for name in self.constraints.loc[:, 'name'].unique():

            # Obtain indexes.
            constraint_index = self.constraints.index[fledge.utils.get_index(self.constraints, name=name)]
            timesteps = self.constraints.loc[constraint_index, 'timestep'].unique()
            columns = (
                self.constraints.loc[constraint_index, :].drop([
                    'name', 'timestep', 'constraint_type'
                ], axis=1).drop_duplicates().dropna(axis=1)
            )
            if len(columns.columns) > 0:
                columns = pd.MultiIndex.from_frame(columns)
            else:
                columns = pd.Index(['total'])

            # Instantiate results dataframe.
            results[name] = pd.DataFrame(index=timesteps, columns=columns)

            # Get results.
            # TODO: Validate dual value signs.
            for timestep in timesteps:
                if self.constraints.loc[constraint_index, 'constraint_type'].str.contains('==').any():
                    results[name].loc[timestep, :] = (
                        0.0
                        - self.dual_vector[fledge.utils.get_index(
                            self.constraints, name=name, timestep=timestep, constraint_type='==>='
                        ), 0]
                        + self.dual_vector[fledge.utils.get_index(
                            self.constraints, name=name, timestep=timestep, constraint_type='==<='
                        ), 0]
                    )
                elif self.constraints.loc[constraint_index, 'constraint_type'].str.contains('>=').any():
                    results[name].loc[timestep, :] = (
                        0.0
                        - self.dual_vector[fledge.utils.get_index(
                            self.constraints, name=name, timestep=timestep, constraint_type='>='
                        ), 0]
                    )
                elif self.constraints.loc[constraint_index, 'constraint_type'].str.contains('<=').any():
                    results[name].loc[timestep, :] = (
                        0.0
                        + self.dual_vector[fledge.utils.get_index(
                            self.constraints, name=name, timestep=timestep, constraint_type='<='
                        ), 0]
                    )

        return results


def starmap(
        function: typing.Callable,
        argument_sequence: typing.Iterable[tuple],
        keyword_arguments: dict = None
) -> list:
    """Utility function to execute a function for a sequence of arguments, effectively replacing a for-loop.
    Allows running repeated function calls in-parallel, based on Python's `multiprocessing` module.

    - If configuration parameter `run_parallel` is set to True, execution is passed to `starmap`
      of multiprocessing pool, hence running the function calls in parallel.
    - Otherwise, execution is passed to `itertools.starmap`, which is the non-parallel equivalent.
    """

    # Apply keyword arguments.
    if keyword_arguments is not None:
        function_partial = functools.partial(function, **keyword_arguments)
    else:
        function_partial = function

    if fledge.config.config['multiprocessing']['run_parallel']:
        # If `run_parallel`, use starmap from multiprocessing pool for parallel execution.
        if fledge.config.parallel_pool is None:
            # Setup parallel pool on first execution.
            log_time('parallel pool setup')
            fledge.config.parallel_pool = fledge.config.get_parallel_pool()
            log_time('parallel pool setup')
        results = fledge.config.parallel_pool.starmap(function_partial, list(argument_sequence))
    else:
        # If not `run_parallel`, use `itertools.starmap` for non-parallel / sequential execution.
        results = list(itertools.starmap(function_partial, argument_sequence))

    return results


def chunk_dict(
        dict_in: dict,
        chunk_count: int = os.cpu_count()
):
    """Divide dictionary into equally sized chunks."""

    chunk_size = int(np.ceil(len(dict_in) / chunk_count))
    dict_iter = iter(dict_in)

    return [
        {j: dict_in[j] for j in itertools.islice(dict_iter, chunk_size)}
        for i in range(0, len(dict_in), chunk_size)
    ]


def chunk_list(
        list_in: typing.Union[typing.Iterable, typing.Sized],
        chunk_count: int = os.cpu_count()
):
    """Divide list into equally sized chunks."""

    chunk_size = int(np.ceil(len(list_in) / chunk_count))
    list_iter = iter(list_in)

    return [
        [j for j in itertools.islice(list_iter, chunk_size)]
        for i in range(0, len(list_in), chunk_size)
    ]


def log_time(
        label: str,
        log_level: str = 'debug',
        logger_object: logging.Logger = logger
):
    """Log start / end message and time duration for given label.

    - When called with given label for the first time, will log start message.
    - When called subsequently with the same / previously used label, will log end message and time duration since
      logging the start message.
    - The log level for start / end messages can be given as keyword argument, By default, messages are logged as
      debug messages.
    - The logger object can be given as keyword argument. By default, uses ``utils.logger`` as logger.
    - Start message: "Starting ``label``."
    - End message: "Completed ``label`` in ``duration`` seconds."

    Arguments:
        label (str): Label for the start / end message.

    Keyword Arguments:
        log_level (str): Log level to which the start / end messages are output. Choices: 'debug', 'info'.
            Default: 'debug'.
        logger_object (logging.logger.Logger): Logger object to which the start / end messages are output. Default:
            ``utils.logger``.
    """

    time_now = time.time()

    if log_level == 'debug':
        logger_handle = lambda message: logger_object.debug(message)
    elif log_level == 'info':
        logger_handle = lambda message: logger_object.info(message)
    else:
        raise ValueError(f"Invalid log level: '{log_level}'")

    if label in log_times.keys():
        logger_handle(f"Completed {label} in {(time_now - log_times.pop(label)):.6f} seconds.")
    else:
        log_times[label] = time_now
        logger_handle(f"Starting {label}.")


def get_index(
        index_set: typing.Union[pd.Index, pd.DataFrame],
        raise_empty_index_error: bool = True,
        **levels_values
):
    """Utility function for obtaining the integer index array for given index set / level / value list combination.

    :syntax:
        - ``get_index(electric_grid_model.nodes, node_type='source', phase=1)``: Get index array for entries in
          index set `electric_grid_model.nodes` with given `node_type` and `phase`.

    Arguments:
        index_set (pd.Index): Index set, e.g., `electric_grid_model.nodes`.

    Keyword Arguments:
        raise_empty_index_error (bool): If true, raise an exception if obtained index array is empty. This is
            the default behavior, because it is usually caused by an invalid level / value combination.
        level (value): All other keyword arguments are interpreted as level / value combinations, where `level`
            must correspond to a level name of the index set.
    """

    # Define handle for get_level_values() depending on index set type.
    if issubclass(type(index_set), pd.Index):
        get_level_values = lambda level: index_set.get_level_values(level)
    elif issubclass(type(index_set), pd.DataFrame):
        # get_level_values = lambda level: index_set.get(level, pd.Series(index=index_set.index, name=level))
        get_level_values = lambda level: index_set.loc[:, level]
    else:
        raise TypeError(f"Invalid index set type: {type(index_set)}")

    # Obtain mask for each level / values combination keyword arguments.
    mask = np.ones(len(index_set), dtype=bool)
    for level, values in levels_values.items():

        # Ensure that values are passed as list.
        if isinstance(values, list):
            pass
        elif isinstance(values, tuple):
            # If values are passed as tuple, wrap in list, but only if index
            # level values are tuples. Otherwise, convert to list.
            if isinstance(get_level_values(level).dropna()[0], tuple):
                values = [values]
            else:
                values = list(values)
        elif isinstance(values, np.ndarray):
            # Convert numpy arrays to list.
            values = values.tolist()
            values = [values] if not isinstance(values, list) else values
        elif isinstance(values, pd.Index):
            # Convert pandas index to list.
            values = values.to_list()
        else:
            # Convert single values into list with one item.
            values = [values]

        # Obtain mask.
        mask &= get_level_values(level).isin(values)

    # Obtain integer index array.
    index = np.flatnonzero(mask)

    # Assert that index is not empty.
    if raise_empty_index_error:
        if not (len(index) > 0):
            raise ValueError(f"Empty index returned for: {levels_values}")

    return index


def get_element_phases_array(element: pd.Series):
    """Utility function for obtaining the list of connected phases for given element data."""

    # Obtain list of connected phases.
    phases_array = (
        np.flatnonzero([
            False,  # Ground / '0' phase connection is not considered.
            element.at['is_phase_1_connected'] == 1,
            element.at['is_phase_2_connected'] == 1,
            element.at['is_phase_3_connected'] == 1
        ])
    )

    return phases_array


def get_element_phases_string(element: pd.Series):
    """Utility function for obtaining the OpenDSS phases string for given element data."""

    # Obtain string of connected phases.
    phases_string = ""
    if element.at['is_phase_1_connected'] == 1:
        phases_string += ".1"
    if element.at['is_phase_2_connected'] == 1:
        phases_string += ".2"
    if element.at['is_phase_3_connected'] == 1:
        phases_string += ".3"

    return phases_string


def get_timestamp(
        time: datetime.datetime = None
) -> str:
    """Generate formatted timestamp string, e.g., for saving results with timestamp."""

    if time is None:
        time = datetime.datetime.now()

    return time.strftime('%Y-%m-%d_%H-%M-%S')


def get_results_path(
        base_name: str,
        scenario_name: str = None
) -> str:
    """Generate results path, which is a new subfolder in the results directory. The subfolder name is
    assembled of the given base name, scenario name and current timestamp. The new subfolder is
    created on disk along with this.

    - Non-alphanumeric characters are removed from `base_name` and `scenario_name`.
    - If is a script file path or `__file__` is passed as `base_name`, the base file name without extension
      will be taken as base name.
    """

    # Preprocess results path name components, including removing non-alphanumeric characters.
    base_name = re.sub(r'\W-+', '', os.path.basename(os.path.splitext(base_name)[0])) + '_'
    scenario_name = '' if scenario_name is None else re.sub(r'\W-+', '', scenario_name) + '_'
    timestamp = fledge.utils.get_timestamp()

    # Obtain results path.
    results_path = os.path.join(fledge.config.config['paths']['results'], f'{base_name}{scenario_name}{timestamp}')

    # Instantiate results directory.
    # TODO: Catch error if dir exists.
    os.mkdir(results_path)

    return results_path


def get_alphanumeric_string(
        string: str
):
    """Create lowercase alphanumeric string from given string, replacing non-alphanumeric characters with underscore."""

    return re.sub(r'\W-+', '_', string).strip('_').lower()


def launch(path):
    """Launch the file at given path with its associated application. If path is a directory, open in file explorer."""

    if not os.path.exists(path):
        raise FileNotFoundError(f'Cannot launch file or directory that does not exist: {path}')

    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        subprocess.Popen(['xdg-open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@fledge.config.memoize('get_building_model')
def get_building_model(*args, **kwargs):
    """Wrapper function for `cobmo.building_model.BuildingModel` with caching support for better performance."""

    return cobmo.building_model.BuildingModel(*args, **kwargs)


def write_figure_plotly(
        figure: go.Figure,
        results_path: str,
        file_format=fledge.config.config['plots']['file_format']
):
    """Utility function for writing / storing plotly figure to output file. File format can be given with
    `file_format` keyword argument, otherwise the default is obtained from config parameter `plots/file_format`.

    - `results_path` should be given as file name without file extension, because the file extension is appended
      automatically based on given `file_format`.
    - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'
    """

    if file_format in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        pio.write_image(
            figure,
            f"{results_path}.{file_format}",
            width=fledge.config.config['plots']['plotly_figure_width'],
            height=fledge.config.config['plots']['plotly_figure_height']
        )
    elif file_format in ['html']:
        pio.write_html(figure, f"{results_path}.{file_format}")
    elif file_format in ['json']:
        pio.write_json(figure, f"{results_path}.{file_format}")
    else:
        raise ValueError(
            f"Invalid `file_format` for `write_figure_plotly`: {file_format}"
            f" - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'"
        )
