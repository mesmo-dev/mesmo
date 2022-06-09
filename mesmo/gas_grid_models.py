"""Gas grid models module."""

import cvxpy as cp
import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd
import scipy.constants
import scipy.sparse as sp
import scipy.sparse.linalg
import typing

import mesmo.config
import mesmo.data_interface
import mesmo.der_models
import mesmo.utils

logger = mesmo.config.get_logger(__name__)


class GasGridModel(object):
    """Gas grid model object."""

    timesteps: pd.Index
    node_names: pd.Index
    line_names: pd.Index
    der_names: pd.Index
    der_types: pd.Index
    nodes: pd.Index
    branches: pd.Index
    branch_loops: pd.Index
    ders: pd.Index
    branch_incidence_1_matrix: sp.spmatrix
    branch_incidence_2_matrix: sp.spmatrix
    branch_incidence_matrix: sp.spmatrix
    branch_incidence_matrix_no_source: sp.spmatrix
    branch_incidence_matrix_source: sp.spmatrix
    branch_incidence_matrix_no_source_no_loop: sp.spmatrix
    branch_incidence_matrix_no_source_loop: sp.spmatrix
    branch_loop_incidence_matrix: sp.spmatrix
    der_node_incidence_matrix: sp.spmatrix
    der_node_incidence_matrix_no_source: sp.spmatrix
    der_gas_consumption_vector_reference: np.ndarray
    branch_gas_flow_vector_reference: np.ndarray
    branch_gas_flow_vector: np.ndarray
    gas_branch_velocity_vector: np.ndarray
    node_pressure_vector_reference: np.ndarray
    node_pressure_source_value: float
    node_pressure_vector_reference_no_source: np.ndarray
    node_pressure_vector_reference_source: np.ndarray
    node_incidence_matrix_no_source: sp.spmatrix
    node_incidence_matrix_source: sp.spmatrix
    # TODO: Revise / reduce use of parameter attributes if possible.
    line_parameters: pd.DataFrame
    source_der_model: mesmo.der_models.DERModel
    plant_efficiency: float

    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain gas grid data.
        gas_grid_data = mesmo.data_interface.GasGridData(scenario_name)

        # Obtain index set for time steps.
        # - This is needed for optimization problem definitions within linear gas grid models.
        self.timesteps = gas_grid_data.scenario_data.timesteps

        # Obtain node / line / DER names.
        self.node_names = pd.Index(gas_grid_data.gas_grid_nodes['node_name'])
        self.line_names = pd.Index(gas_grid_data.gas_grid_lines['line_name'])
        self.der_names = pd.Index(gas_grid_data.gas_grid_ders['der_name'])
        self.der_types = pd.Index(gas_grid_data.gas_grid_ders['der_type']).unique()

        # Obtain node / branch / DER index set.
        nodes = (
            pd.concat([
                gas_grid_data.gas_grid_nodes.loc[:, 'node_name'].apply(
                    # Obtain `node_type` column.
                    lambda value:
                    'source' if value == gas_grid_data.gas_grid.at['source_node_name']
                    else 'no_source'
                ).rename('node_type'),
                gas_grid_data.gas_grid_nodes.loc[:, 'node_name']
            ], axis='columns')
        )
        self.nodes = pd.MultiIndex.from_frame(nodes)
        self.branches = pd.MultiIndex.from_product([self.line_names, ["no_loop"]], names=["branch_name", "loop_type"])
        self.branch_loops = pd.MultiIndex.from_tuples([], names=["loop_id", "branch_name"])  # Values are filled below.
        self.ders = pd.MultiIndex.from_frame(gas_grid_data.gas_grid_ders[["der_type", "der_name"]])

        # Define branch to node incidence matrix.
        self.branch_incidence_1_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=int)
        self.branch_incidence_2_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=int)

        # Add lines to branch incidence matrices and identify any loops.
        # - Uses temporary node tree variable to track construction of the network and identify any loops / cycles.
        branches_loops = self.branches.to_frame()
        node_trees = []
        for line_index, line in gas_grid_data.gas_grid_lines.iterrows():

            # Obtain indexes for positioning the line in the incidence matrices.
            node_index_1 = mesmo.utils.get_index(self.nodes, node_name=line["node_1_name"])
            node_index_2 = mesmo.utils.get_index(self.nodes, node_name=line["node_2_name"])
            branch_index = mesmo.utils.get_index(self.branches, branch_name=line["line_name"])

            # Insert connection indicators into incidence matrices.
            self.branch_incidence_1_matrix[np.ix_(branch_index, node_index_1)] += 1
            self.branch_incidence_2_matrix[np.ix_(branch_index, node_index_2)] += 1

            # Check if node 1 or node 2 are in any node trees.
            node_tree_index_1 = None
            node_tree_index_2 = None
            for node_tree_index, node_tree in enumerate(node_trees):
                if line["node_1_name"] in node_tree:
                    node_tree_index_1 = node_tree_index
                if line["node_2_name"] in node_tree:
                    node_tree_index_2 = node_tree_index
            if (node_tree_index_1 is None) and (node_tree_index_2 is None):
                # Create new tree, if neither node is on any tree.
                node_trees.append([line["node_1_name"], line["node_2_name"]])
            elif (node_tree_index_1 is not None) and (node_tree_index_2 is None):
                # Add node to tree, if other node is on any tree.
                node_trees[node_tree_index_1].append(line["node_2_name"])
            elif (node_tree_index_1 is None) and (node_tree_index_2 is not None):
                # Add node to tree, if other node is on any tree.
                node_trees[node_tree_index_2].append(line["node_1_name"])
            else:
                if node_tree_index_1 == node_tree_index_2:
                    # Mark branch as loop, if both nodes are in the same tree.
                    branches_loops.loc[self.branches[branch_index], "loop_type"] = "loop"
                else:
                    # Merge trees, if the branch connects nodes on different trees.
                    node_trees[node_tree_index_1].extend(node_trees[node_tree_index_2])
                    node_trees[node_tree_index_2] = []

        # Update branch / loop indexes.
        self.branches = pd.MultiIndex.from_frame(branches_loops)
        self.branch_loops = pd.MultiIndex.from_frame(
            pd.concat(
                [
                    pd.Series(range(sum(branches_loops.loc[:, "loop_type"] == "loop")), name="loop_id", dtype=int),
                    branches_loops.loc[branches_loops.loc[:, "loop_type"] == "loop", "branch_name"].reset_index(
                        drop=True
                    ),
                ],
                axis="columns",
            )
        )

        # Raise errors on invalid network configurations.
        node_trees = [node_tree for node_tree in node_trees if len(node_tree) > 0]
        if len(node_trees) > 1:
            raise ValueError(
                "The thermal grid contains disjoint sections of nodes:"
                + "".join(
                    [
                        f"\nSection {node_tree_index}: {node_tree}"
                        for node_tree_index, node_tree in enumerate(node_trees)
                    ]
                )
            )
        elif len(node_trees[0]) != len(self.node_names):
            raise ValueError(
                f"The thermal grid contains disconnected nodes:\n"
                f"{[node_name for node_name in self.node_names if node_name not in node_trees[0]]}"
            )

        # Obtained combined branch incidence matrix.
        self.branch_incidence_matrix = self.branch_incidence_1_matrix - self.branch_incidence_2_matrix

        # Convert DOK matrices to CSR matrices.
        self.branch_incidence_1_matrix = self.branch_incidence_1_matrix.tocsr()
        self.branch_incidence_2_matrix = self.branch_incidence_2_matrix.tocsr()
        self.branch_incidence_matrix = self.branch_incidence_matrix.tocsr()

        # Obtain shorthand definitions.
        self.branch_incidence_matrix_no_source_no_loop = self.branch_incidence_matrix[
            np.ix_(
                mesmo.utils.get_index(self.branches, loop_type="no_loop"),
                mesmo.utils.get_index(self.nodes, node_type="no_source"),
            )
        ]
        self.branch_incidence_matrix_no_source_loop = self.branch_incidence_matrix[
            np.ix_(
                mesmo.utils.get_index(self.branches, loop_type="loop", raise_empty_index_error=False),
                mesmo.utils.get_index(self.nodes, node_type="no_source"),
            )
        ]

        # Obtain branch-to-loop incidence matrix.
        self.branch_loop_incidence_matrix = sp.vstack(
            [
                -1.0 * sp.linalg.inv(self.branch_incidence_matrix_no_source_no_loop.transpose())
                # Using `sp.linalg.inv()` instead of `sp.linalg.spsolve()` to preserve dimensions in all cases.
                @ self.branch_incidence_matrix_no_source_loop.transpose(),
                sp.eye(len(self.branch_loops)),
            ]
        ).tocsr()


        # Instantiate DER-to-node incidence matrix.
        self.der_node_incidence_matrix = sp.dok_matrix((len(self.nodes), len(self.ders)), dtype=int)

        # Add DERs into DER incidence matrix.
        for der_name, der in gas_grid_data.gas_grid_ders.iterrows():

            # Obtain indexes for positioning the DER in the incidence matrix.
            node_index = mesmo.utils.get_index(self.nodes, node_name=der["node_name"])
            der_index = mesmo.utils.get_index(self.ders, der_name=der["der_name"])

            # Insert connection indicator into incidence matrices.
            self.der_node_incidence_matrix[node_index, der_index] = 1

        # Convert DOK matrices to CSR matrices.
        self.der_node_incidence_matrix = self.der_node_incidence_matrix.tocsr()


        # Obtain DER nominal gas consumption vector.
        self.der_gas_consumption_vector_reference = (
            gas_grid_data.gas_grid_ders.loc[:, 'gas_consumption_nominal'].values
        )

        # Obtain nominal branch flow vector.
        self.branch_gas_flow_vector_reference = (
            np.pi
            * (gas_grid_data.gas_grid_lines.loc[:, 'diameter'].values / 2) ** 2
            * gas_grid_data.gas_grid_lines.loc[:, 'maximum_velocity'].values
        )
        self.node_pressure_vector_reference = np.ones(len(self.nodes))
        self.node_pressure_source_value = 0.0

        # Obtain line parameters.
        self.line_parameters = gas_grid_data.gas_grid_lines.loc[:, ["length", "diameter", "absolute_roughness"]]

        # Obtain nominal node pressure vector.
        # TODO: Define proper node pressure reference vector.
        self.node_pressure_vector_reference = (
            np.ones(len(self.nodes))
        )

        # Obtain DER model source node.
        # TODO: Use state space model for simulation / optimization.
        self.source_der_model = (
            mesmo.der_models.make_der_model(
                gas_grid_data.gas_grid.at['source_der_model_name'],
                gas_grid_data.der_data,
                is_standalone=True
            )
        )

        # Define shorthands for no-source / source variables.
        # TODO: Replace local variables in power flow / linear models.
        node_incidence_matrix = sp.identity(len(self.nodes)).tocsr()
        self.node_incidence_matrix_no_source = node_incidence_matrix[
            np.ix_(range(len(self.nodes)), mesmo.utils.get_index(self.nodes, node_type="no_source"))
        ]
        self.node_incidence_matrix_source = node_incidence_matrix[
            np.ix_(range(len(self.nodes)), mesmo.utils.get_index(self.nodes, node_type="source"))
        ]
        self.der_node_incidence_matrix_no_source = self.der_node_incidence_matrix[
            np.ix_(mesmo.utils.get_index(self.nodes, node_type="no_source"), range(len(self.ders)))
        ]
        self.branch_incidence_matrix_no_source = self.branch_incidence_matrix[
            np.ix_(range(len(self.branches)), mesmo.utils.get_index(self.nodes, node_type="no_source"))
        ]
        self.branch_incidence_matrix_source = self.branch_incidence_matrix[
            np.ix_(range(len(self.branches)), mesmo.utils.get_index(self.nodes, node_type="source"))
        ]
        self.node_pressure_vector_reference_no_source = self.node_pressure_vector_reference[
            mesmo.utils.get_index(self.nodes, node_type="no_source")
        ]
        self.node_pressure_vector_reference_source = self.node_pressure_vector_reference[
            mesmo.utils.get_index(self.nodes, node_type="source")
        ]

    def get_branch_loss_coefficient_vector(self, branch_flow_vector: np.ndarray):

            # Obtain branch velocity vector.
            gas_branch_velocity_vector = (
                    4.0 * branch_flow_vector / (np.pi * self.line_parameters.loc[:, "diameter"].values ** 2)
            )

            # Obtain branch Reynolds coefficient vector.
            branch_reynold_vector = (
                    np.abs(gas_branch_velocity_vector)
                    * self.line_parameters.loc[:, "diameter"].values
                    / mesmo.config.gas_viscosity
            )

            # Obtain branch friction factor vector.
            @np.vectorize
            def branch_friction_factor_vector(reynold, absolute_roughness, diameter):

                # No flow.
                if reynold == 0:
                    friction_factor = 0

                # Laminar Flow, based on Hagen-Poiseuille velocity profile, analytical correlation.
                elif 0 < reynold < 4000:
                    friction_factor = 64 / reynold

                # Turbulent flow, Swamee-Jain formula, approximating correlation of Colebrook-White equation.
                elif 4000 <= reynold:
                    if not (reynold <= 100000000 and 0.000001 <= ((absolute_roughness / 1000) / diameter) <= 0.01):
                        logger.warning(
                            "Exceeding validity range of Swamee-Jain formula for calculation of friction factor."
                        )
                    friction_factor = 0.25 / (
                        np.log(
                            (absolute_roughness / 1000) / (3.7 * diameter) + 5.74 / (reynold ** 0.9)
                        )
                    ) ** 2
                else:
                    raise ValueError(f"Invalid Reynolds coefficient: {reynold}")

                # Convert from 1/m to 1/km.
                friction_factor *= 1.0e3

                return friction_factor

            # Obtain branch head loss coefficient vector.
            # - Darcy-Weisbach Equation pressure-loss form.
            branch_loss_coefficient_vector = (
                branch_friction_factor_vector(
                    branch_reynold_vector,
                    self.line_parameters.loc[:, "absolute_roughness"].values,
                    self.line_parameters.loc[:, "diameter"].values,
                )
                * self.line_parameters.loc[:, "length"].values
                * gas_branch_velocity_vector ** 2
                * mesmo.config.gas_density
                / (
                        2.0
                        * self.line_parameters.loc[:, "diameter"].values
                )
            )

            return branch_loss_coefficient_vector



class GasGridDEROperationResults(mesmo.utils.ResultsBase):

    der_gas_consumption_vector: pd.DataFrame
    der_gas_consumption_vector_per_unit: pd.DataFrame


class GasGridOperationResults(GasGridDEROperationResults):

    gas_grid_model: GasGridModel
    node_pressure_vector: pd.DataFrame
    node_pressure_vector_per_unit: pd.DataFrame
    branch_gas_flow_vector: pd.DataFrame
    branch_gas_flow_vector_per_unit: pd.DataFrame


class GasFlowSolution(object):
    """Gas grid flow solution object."""

    der_gas_consumption_vector: np.ndarray
    source_flow: float
    branch_gas_flow_vector: np.ndarray
    gas_branch_velocity_vector: np.ndarray
    gas_branch_reynold_vector: np.ndarray
    gas_branch_friction_factor_vector: np.ndarray
    branch_loss_coefficient_vector: np.ndarray
    source_pressure: float
    node_pressure_vector: np.ndarray

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain gas grid model.
        gas_grid_model = GasGridModel(scenario_name)

        self.__init__(
            gas_grid_model
        )

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel
    ):

        # Obtain DER gas consumption vector.
        der_gas_consumption_vector = gas_grid_model.der_gas_consumption_vector_reference

        self.__init__(
            gas_grid_model,
            der_gas_consumption_vector
        )

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel,
            der_gas_consumption_vector: np.ndarray
    ):

        # Obtain DER gas consumption vector.
        self.der_gas_consumption_vector = der_gas_consumption_vector.ravel()

        # Define shorthand for DER volume flow vector.
        der_flow_vector = (
            self.der_gas_consumption_vector
            / mesmo.config.gas_density
        )
        # Obtain DER / source volume flow vector.

        self.source_flow = (
            -1.0 * np.sum(self.der_gas_consumption_vector)
        )

        # Obtain branch volume flow vector.
        self.branch_gas_flow_vector = (
            scipy.sparse.linalg.spsolve(
                gas_grid_model.branch_incidence_matrix_no_source.transpose(),
                gas_grid_model.der_node_incidence_matrix_no_source
                @ np.transpose([der_flow_vector]),
            )
        ).ravel()

        # Obtain node / source pressure vector.
        node_pressure_vector_no_source = scipy.sparse.linalg.spsolve(
            gas_grid_model.branch_incidence_matrix_no_source.tocsc(),
            (
                gas_grid_model.get_branch_loss_coefficient_vector(self.branch_gas_flow_vector)
                * self.branch_gas_flow_vector
                * np.abs(self.branch_gas_flow_vector)
            ),
        )
        self.node_pressure_vector = (
            gas_grid_model.node_incidence_matrix_no_source
            @ node_pressure_vector_no_source
            + gas_grid_model.node_incidence_matrix_source
            @ gas_grid_model.node_pressure_vector_reference_source
            * gas_grid_model.node_pressure_source_value
        )





class GasFlowSolutionSet(object):

    power_flow_solutions: typing.Dict[pd.Timestamp, GasFlowSolution]
    gas_grid_model: GasGridModel
    der_gas_consumption_vector: pd.DataFrame
    timesteps: pd.Index

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel,
            der_operation_results: GasGridDEROperationResults,
            **kwargs
    ):

        der_gas_consumption_vector = der_operation_results.der_gas_consumption_vector

        self.__init__(
            gas_grid_model,
            der_gas_consumption_vector,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel,
            der_gas_consumption_vector: pd.DataFrame,
            power_flow_solution_method=GasFlowSolution
    ):

        # Store attributes.
        self.gas_grid_model = gas_grid_model
        self.der_gas_consumption_vector = der_gas_consumption_vector
        self.timesteps = self.gas_grid_model.timesteps

        # Obtain power flow solutions.
        power_flow_solutions = (
            mesmo.utils.starmap(
                power_flow_solution_method,
                zip(
                    itertools.repeat(self.gas_grid_model),
                    der_gas_consumption_vector.values
                )
            )
        )
        self.power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))

    def get_results(self) -> GasGridOperationResults:

        raise NotImplementedError



class LinearGasGridModel(object):
    """Linear gas grid model object."""

    gas_grid_model: GasGridModel
    gas_power_flow_solution: GasFlowSolution
    sensitivity_branch_flow_by_node_consumption: sp.spmatrix
    sensitivity_branch_flow_by_der_consumption: sp.spmatrix
    sensitivity_node_pressure_by_node_consumption: sp.spmatrix
    sensitivity_node_pressure_by_der_consumption: sp.spmatrix
    sensitivity_pump_power_by_node_consumption: np.array
    sensitivity_pump_power_by_der_consumption: np.array

    @multimethod
    def __init__(
            self,
            scenario_name: str,
    ):

        # Obtain gas grid model.
        gas_grid_model = (
            GasGridModel(scenario_name)
        )

        # Obtain DER consumption vector.
        der_gas_consumption_vector = (
            gas_grid_model.der_gas_consumption_vector_reference
        )

        # Obtain gas power flow solution.
        gas_flow_solution = (
            GasFlowSolution(
                gas_grid_model,
                der_gas_consumption_vector
            )
        )

        self.__init__(
            gas_grid_model,
            gas_flow_solution
        )

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel,
            gas_flow_solution: GasFlowSolution,
    ):

        # Store gas grid model.
        self.gas_grid_model = gas_grid_model

        # Store gas power flow solution.
        self.gas_flow_solution = gas_flow_solution

        # Obtain inverse / transpose incidence matrices.
        node_index_no_source = (
            mesmo.utils.get_index(self.gas_grid_model.nodes, node_type='no_source')  # Define shorthand.
        )
        branch_node_incidence_matrix_inverse = (
            sp.dok_matrix(
                (len(self.gas_grid_model.branches), len(self.gas_grid_model.nodes)),
                dtype=float
            )
        )
        branch_node_incidence_matrix_inverse[np.ix_(
            range(len(self.gas_grid_model.branches)),
            node_index_no_source
        )] = (
            scipy.sparse.linalg.inv(
                self.gas_grid_model.branch_node_incidence_matrix[node_index_no_source, :].tocsc()
            )
        )
        branch_node_incidence_matrix_inverse = branch_node_incidence_matrix_inverse.tocsr()
        branch_node_incidence_matrix_transpose_inverse = (
            sp.dok_matrix(
                (len(self.gas_grid_model.nodes), len(self.gas_grid_model.branches)),
                dtype=float
            )
        )
        branch_node_incidence_matrix_transpose_inverse[np.ix_(
            node_index_no_source,
            range(len(self.gas_grid_model.branches))
        )] = (
            scipy.sparse.linalg.inv(
                self.gas_grid_model.branch_node_incidence_matrix[node_index_no_source, :].transpose()
            )
        )
        branch_node_incidence_matrix_transpose_inverse = branch_node_incidence_matrix_transpose_inverse.tocsr()
        der_node_incidence_matrix_transpose = np.transpose(self.gas_grid_model.der_node_incidence_matrix)

        # Obtain sensitivity matrices.
        self.sensitivity_node_consumption_by_der_consumption = (
            self.gas_grid_model.der_node_incidence_matrix
        )
        self.sensitivity_branch_flow_by_node_consumption = (
            self.gas_grid_model.branch_node_incidence_matrix_inverse
        )
        self.sensitivity_branch_flow_by_der_consumption = (
            self.sensitivity_branch_flow_by_node_consumption
            @ self.sensitivity_node_consumption_by_der_consumption
        )
        self.sensitivity_node_pressure_by_der_consumption = (
            self.sensitivity_node_pressure_by_node_consumption
            @ self.sensitivity_node_consumption_by_der_consumption
        )
        self.sensitivity_node_pressure_by_node_consumption = (
            branch_node_incidence_matrix_transpose_inverse
            @ sp.diags(
                gas_flow_solution.branch_friction_factor_vector
                * gas_grid_model.line_length_vector
                * gas_grid_model.gas_branch_velocity_vector ** 2
                * gas_grid_model.gas_density
                / (
                    2.0
                    * self.gas_grid_model.line_diameter_vector
                )
            )
            @ self.sensitivity_branch_flow_by_node_consumption
        )


LinearGasGridModelGlobal = LinearGasGridModel


class LinearGasGridModelSet(object):

    linear_gas_grid_models: typing.Dict[pd.Timestamp, LinearGasGridModel]
    gas_grid_model: GasGridModel
    timesteps: pd.Index

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel,
            gas_flow_solution_set: GasFlowSolutionSet,
            linear_gas_grid_model_method: typing.Type[LinearGasGridModel] = LinearGasGridModelGlobal
    ):

        self.check_linear_gas_grid_model_method(linear_gas_grid_model_method)

        # Obtain linear gas grid models.
        linear_gas_grid_models = (
            mesmo.utils.starmap(
                linear_gas_grid_model_method,
                zip(
                    itertools.repeat(gas_grid_model),
                    gas_flow_solution_set.power_flow_solutions.values()
                )
            )
        )
        linear_gas_grid_models = (
            dict(zip(gas_grid_model.timesteps, linear_gas_grid_models))
        )

        self.__init__(
            gas_grid_model,
            linear_gas_grid_models
        )

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel,
            gas_flow_solution: GasFlowSolution,
            linear_gas_grid_model_method: typing.Type[LinearGasGridModel] = LinearGasGridModelGlobal
    ):

        self.check_linear_gas_grid_model_method(linear_gas_grid_model_method)

        # Obtain linear gas grid models.
        linear_gas_grid_model = LinearGasGridModelGlobal(gas_grid_model, gas_flow_solution)
        linear_gas_grid_models = (
            dict(zip(gas_grid_model.timesteps, itertools.repeat(linear_gas_grid_model)))
        )

        self.__init__(
            gas_grid_model,
            linear_gas_grid_models
        )

    @multimethod
    def __init__(
            self,
            gas_grid_model: GasGridModel,
            linear_gas_grid_models: typing.Dict[pd.Timestamp, LinearGasGridModel]
    ):

        # Store attributes.
        self.gas_grid_model = gas_grid_model
        self.timesteps = self.gas_grid_model.timesteps
        self.linear_gas_grid_models = linear_gas_grid_models

    @staticmethod
    def check_linear_gas_grid_model_method(linear_gas_grid_model_method):

        if not issubclass(linear_gas_grid_model_method, LinearGasGridModel):
            raise ValueError(f"Invalid linear gas grid model method: {linear_gas_grid_model_method}")

    def define_optimization_problem(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData,
            **kwargs
    ):

        # Defined optimization problem definitions through respective sub-methods.
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_parameters(
            optimization_problem,
            price_data,
            **kwargs
        )
        self.define_optimization_constraints(optimization_problem)
        self.define_optimization_objective(optimization_problem)

    def define_optimization_variables(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem
    ):

        # Define DER consumption vector variables.
        # - Only if these have not yet been defined within `DERModelSet`.
        if 'der_gas_consumption_vector' not in optimization_problem.variables.loc[:, 'name'].values:
            optimization_problem.define_variable(
                'der_gas_consumption_vector', timestep=self.timesteps, der=self.gas_grid_model.ders
            )

        # Define node pressure, branch flow and pump power variables.
        optimization_problem.define_variable(
            'node_pressure_vector', timestep=self.timesteps, node=self.gas_grid_model.nodes
        )
        optimization_problem.define_variable(
            'branch_flow_vector', timestep=self.timesteps, branch=self.gas_grid_model.branches
        )

    def define_optimization_parameters(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData,
            node_pressure_vector_minimum: np.ndarray = None,
            branch_flow_vector_maximum: np.ndarray = None,
    ):

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')

        # Define pressure variable term.
        optimization_problem.define_parameter(
            'pressure_variable',
            sp.block_diag([
                sp.diags(linear_gas_grid_model.gas_grid_model.node_pressure_vector_reference ** -1)
                @ linear_gas_grid_model.sensitivity_node_pressure_by_der_consumption
                @ sp.diags(linear_gas_grid_model.gas_grid_model.der_gas_consumption_vector_reference)
                for linear_gas_grid_model in self.linear_gas_grid_models.values()
            ])
        )

        # Define pressure constant term.
        optimization_problem.define_parameter(
            'pressure_constant',
            np.concatenate([
                sp.diags(linear_gas_grid_model.gas_grid_model.node_pressure_vector_reference ** -1)
                @ (
                    np.transpose([linear_gas_grid_model.gas_flow_solution.node_pressure_vector])
                    - linear_gas_grid_model.sensitivity_node_pressure_by_der_consumption
                    @ np.transpose([linear_gas_grid_model.gas_flow_solution.der_gas_consumption_vector])
                ) for linear_gas_grid_model in self.linear_gas_grid_models.values()
            ])
        )

        # Define branch flow variable term.
        optimization_problem.define_parameter(
            'branch_flow_variable',
            sp.block_diag([
                sp.diags(linear_gas_grid_model.gas_grid_model.branch_flow_vector_reference ** -1)
                @ linear_gas_grid_model.sensitivity_branch_flow_by_der_consumption
                @ sp.diags(linear_gas_grid_model.gas_grid_model.der_gas_consumption_vector_reference)
                for linear_gas_grid_model in self.linear_gas_grid_models.values()
            ])
        )

        # Define branch flow constant term.
        optimization_problem.define_parameter(
            'branch_flow_constant',
            np.concatenate([
                sp.diags(linear_gas_grid_model.gas_grid_model.branch_flow_vector_reference ** -1)
                @ (
                    np.transpose([linear_gas_grid_model.gas_flow_solution.branch_flow_vector])
                    - linear_gas_grid_model.sensitivity_branch_flow_by_der_consumption
                    @ np.transpose([linear_gas_grid_model.gas_flow_solution.der_gas_consumption_vector])
                ) for linear_gas_grid_model in self.linear_gas_grid_models.values()
            ])
        )

        # Define pressure limits.
        optimization_problem.define_parameter(
            'node_pressure_minimum',
            np.concatenate([
                node_pressure_vector_minimum.ravel()
                / linear_gas_grid_model.gas_grid_model.node_pressure_vector_reference
                for linear_gas_grid_model in self.linear_gas_grid_models.values()
            ])
            if node_pressure_vector_minimum is not None
            else -np.inf * np.ones((len(self.gas_grid_model.nodes) * len(self.timesteps), ))
        )

        # Define branch flow limits.
        optimization_problem.define_parameter(
            'branch_flow_minimum',
            np.concatenate([
                - branch_flow_vector_maximum.ravel()
                / linear_gas_grid_model.gas_grid_model.branch_flow_vector_reference
                for linear_gas_grid_model in self.linear_gas_grid_models.values()
            ])
            if branch_flow_vector_maximum is not None
            else -np.inf * np.ones((len(self.gas_grid_model.branches) * len(self.timesteps), ))
        )
        optimization_problem.define_parameter(
            'branch_flow_maximum',
            np.concatenate([
                branch_flow_vector_maximum.ravel()
                / linear_gas_grid_model.gas_grid_model.branch_flow_vector_reference
                for linear_gas_grid_model in self.linear_gas_grid_models.values()
            ])
            if branch_flow_vector_maximum is not None
            else +np.inf * np.ones((len(self.gas_grid_model.branches) * len(self.timesteps), ))
        )

        # Define objective parameters.
        optimization_problem.define_parameter(
            'gas_grid_gas_cost',
            np.array([price_data.price_timeseries.loc[:, ('gas', 'source', 'source')].values])
            * -1.0 * timestep_interval_hours  # In Wh.
            @ sp.block_diag(
                [np.array([self.gas_grid_model.der_gas_consumption_vector_reference])] * len(self.timesteps)
            )
        )
        optimization_problem.define_parameter(
            'gas_grid_gas_cost_sensitivity',
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * np.concatenate(
                [np.array([self.gas_grid_model.der_gas_consumption_vector_reference ** 2])] * len(self.timesteps),
                axis=1
            )
        )

    def define_optimization_constraints(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
    ):

        # Define pressure equation.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='node_pressure_vector', timestep=self.timesteps,
                node=self.gas_grid_model.nodes
            )),
            '==',
            ('variable', 'pressure_variable', dict(
                name='der_gas_consumption_vector', timestep=self.timesteps,
                der=self.gas_grid_model.ders
            )),
            ('constant', 'pressure_constant', dict(timestep=self.timesteps)),
        )

        # Define branch flow equation.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='branch_flow_vector', timestep=self.timesteps,
                branch=self.gas_grid_model.branches
            )),
            '==',
            ('variable', 'branch_flow_variable', dict(
                name='der_gas_consumption_vector', timestep=self.timesteps,
                der=self.gas_grid_model.ders
            )),
            ('constant', 'branch_flow_constant', dict(timestep=self.timesteps)),
        )

        # Define pressure limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='node_pressure_vector', timestep=self.timesteps,
                node=self.gas_grid_model.nodes
            )),
            '>=',
            ('constant', 'node_pressure_minimum', dict(timestep=self.timesteps)),
            keys=dict(
                name='node_pressure_vector_minimum_constraint', timestep=self.timesteps,
                node=self.gas_grid_model.nodes
            ),
        )

        # Define branch flow limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='branch_flow_vector', timestep=self.timesteps,
                branch=self.gas_grid_model.branches
            )),
            '>=',
            ('constant', 'branch_flow_minimum', dict(timestep=self.timesteps)),
            keys=dict(
                name='branch_flow_vector_minimum_constraint', timestep=self.timesteps,
                branch=self.gas_grid_model.branches
            ),
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='branch_flow_vector', timestep=self.timesteps,
                branch=self.gas_grid_model.branches
            )),
            '<=',
            ('constant', 'branch_flow_maximum', dict(timestep=self.timesteps)),
            keys=dict(
                name='branch_flow_vector_maximum_constraint', timestep=self.timesteps,
                branch=self.gas_grid_model.branches
            ),
        )

    def define_optimization_objective(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem
    ):

        # Set objective flag.
        optimization_problem.flags['has_gas_grid_objective'] = True

        # Define objective for gas loads.
        # - Defined as cost of gas supply at gas grid source node.
        # - Only defined here, if not yet defined as cost of gas supply at the DER node
        #   in `mesmo.der_models.DERModel.define_optimization_objective`.
        if not optimization_problem.flags.get('has_der_objective'):

            # Gas cost / revenue.
            # - Cost for load / demand, revenue for generation / supply.
            optimization_problem.define_objective(
                (
                    'variable',
                    'gas_grid_gas_cost',
                    dict(name='der_gas_consumption_vector', timestep=self.timesteps, der=self.gas_grid_model.ders)
                ), (
                    'variable',
                    'gas_grid_gas_cost_sensitivity',
                    dict(name='der_gas_consumption_vector', timestep=self.timesteps, der=self.gas_grid_model.ders),
                    dict(name='der_gas_consumption_vector', timestep=self.timesteps, der=self.gas_grid_model.ders)
                )
            )

    def evaluate_optimization_objective(
            self,
            results: GasGridOperationResults,
            price_data: mesmo.data_interface.PriceData
    ) -> float:

        # Instantiate optimization problem.
        optimization_problem = mesmo.utils.OptimizationProblem()
        self.define_optimization_parameters(optimization_problem, price_data)
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_objective(optimization_problem)

        # Instantiate variable vector.
        x_vector = np.zeros((len(optimization_problem.variables), 1))

        # Set variable vector values.
        objective_variable_names = [
            'der_gas_consumption_vector_per_unit'
        ]
        for variable_name in objective_variable_names:
            index = mesmo.utils.get_index(optimization_problem.variables, name=variable_name.replace('_per_unit', ''))
            x_vector[index, 0] = results[variable_name].values.ravel()

        # Obtain objective value.
        objective = optimization_problem.evaluate_objective(x_vector)

        return objective

    # def get_optimization_dlmps(
    #         self,
    #         optimization_problem: mesmo.utils.OptimizationProblem,
    #         price_data: mesmo.data_interface.PriceData
    # ) -> ThermalGridDLMPResults:
    #
    #     # Obtain individual duals.
    #     node_pressure_vector_minimum_dual = (
    #         optimization_problem.duals['node_head_vector_minimum_constraint'].loc[
    #             self.thermal_grid_model.timesteps, self.thermal_grid_model.nodes
    #         ]
    #         / np.array([(self.thermal_grid_model.node_head_vector_reference)])
    #     )
    #     branch_flow_vector_minimum_dual = (
    #         optimization_problem.duals['branch_flow_vector_minimum_constraint'].loc[
    #             self.thermal_grid_model.timesteps, self.thermal_grid_model.branches
    #         ]
    #         / np.array([self.thermal_grid_model.branch_flow_vector_reference])
    #     )
    #     branch_flow_vector_maximum_dual = (
    #         -1.0 * optimization_problem.duals['branch_flow_vector_maximum_constraint'].loc[
    #             self.thermal_grid_model.timesteps, self.thermal_grid_model.branches
    #         ]
    #         / np.array([self.thermal_grid_model.branch_flow_vector_reference])
    #     )
    #
    #     # Instantiate DLMP variables.
    #     thermal_grid_energy_dlmp_node_thermal_power = (
    #         pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #     thermal_grid_head_dlmp_node_thermal_power = (
    #
    #
    #         pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #     thermal_grid_congestion_dlmp_node_thermal_power = (
    #         pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #     thermal_grid_pump_dlmp_node_thermal_power = (
    #         pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #
    #     thermal_grid_energy_dlmp_der_thermal_power = (
    #         pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #     thermal_grid_head_dlmp_der_thermal_power = (
    #         pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #     thermal_grid_congestion_dlmp_der_thermal_power = (
    #         pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #     thermal_grid_pump_dlmp_der_thermal_power = (
    #         pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
    #     )
    #
    #     # Obtain DLMPs.
    #     for timestep in self.thermal_grid_model.timesteps:
    #         thermal_grid_energy_dlmp_node_thermal_power.loc[timestep, :] = (
    #             price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
    #             / self.thermal_grid_model.plant_efficiency
    #         )
    #         thermal_grid_head_dlmp_node_thermal_power.loc[timestep, :] = (
    #             (
    #                 self.linear_thermal_grid_models[timestep].sensitivity_node_head_by_node_power.transpose()
    #                 @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
    #             ).ravel()
    #         )
    #         thermal_grid_congestion_dlmp_node_thermal_power.loc[timestep, :] = (
    #             (
    #                 self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_node_power.transpose()
    #                 @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
    #             ).ravel()
    #             + (
    #                 self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_node_power.transpose()
    #                 @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
    #             ).ravel()
    #         )
    #         thermal_grid_pump_dlmp_node_thermal_power.loc[timestep, :] = (
    #             -1.0 * self.linear_thermal_grid_models[timestep].sensitivity_pump_power_by_node_power.ravel()
    #             * price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
    #         )
    #
    #         thermal_grid_energy_dlmp_der_thermal_power.loc[timestep, :] = (
    #             price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
    #             / self.thermal_grid_model.plant_efficiency
    #         )
    #         thermal_grid_head_dlmp_der_thermal_power.loc[timestep, :] = (
    #             (
    #                 self.linear_thermal_grid_models[timestep].sensitivity_node_head_by_der_power.transpose()
    #                 @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
    #             ).ravel()
    #         )
    #         thermal_grid_congestion_dlmp_der_thermal_power.loc[timestep, :] = (
    #             (
    #                 self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_der_power.transpose()
    #                 @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
    #             ).ravel()
    #             + (
    #                 self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_der_power.transpose()
    #                 @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
    #             ).ravel()
    #         )
    #         thermal_grid_pump_dlmp_der_thermal_power.loc[timestep, :] = (
    #             -1.0 * self.linear_thermal_grid_models[timestep].sensitivity_pump_power_by_der_power.ravel()
    #             * price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
    #         )
    #
    #     thermal_grid_total_dlmp_node_thermal_power = (
    #         thermal_grid_energy_dlmp_node_thermal_power
    #         + thermal_grid_head_dlmp_node_thermal_power
    #         + thermal_grid_congestion_dlmp_node_thermal_power
    #         + thermal_grid_pump_dlmp_node_thermal_power
    #     )
    #     thermal_grid_total_dlmp_der_thermal_power = (
    #         thermal_grid_energy_dlmp_der_thermal_power
    #         + thermal_grid_head_dlmp_der_thermal_power
    #         + thermal_grid_congestion_dlmp_der_thermal_power
    #         + thermal_grid_pump_dlmp_der_thermal_power
    #     )
    #
    #     # Obtain total DLMPs in format similar to `mesmo.data_interface.PriceData.price_timeseries`.
    #     thermal_grid_total_dlmp_price_timeseries = (
    #         pd.concat(
    #             [
    #                 price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].rename(
    #                     ('source', 'source')
    #                 ),
    #                 thermal_grid_total_dlmp_der_thermal_power
    #             ],
    #             axis='columns',
    #             keys=['thermal_power', 'thermal_power'],
    #             names=['commodity_type']
    #         )
    #     )
    #     # Redefine columns to avoid slicing issues.
    #     thermal_grid_total_dlmp_price_timeseries.columns = (
    #         price_data.price_timeseries.columns[
    #             price_data.price_timeseries.columns.isin(thermal_grid_total_dlmp_price_timeseries.columns)
    #         ]
    #     )
    #
    #     return ThermalGridDLMPResults(
    #         thermal_grid_energy_dlmp_node_thermal_power=thermal_grid_energy_dlmp_node_thermal_power,
    #         thermal_grid_head_dlmp_node_thermal_power=thermal_grid_head_dlmp_node_thermal_power,
    #         thermal_grid_congestion_dlmp_node_thermal_power=thermal_grid_congestion_dlmp_node_thermal_power,
    #         thermal_grid_pump_dlmp_node_thermal_power=thermal_grid_pump_dlmp_node_thermal_power,
    #         thermal_grid_total_dlmp_node_thermal_power=thermal_grid_total_dlmp_node_thermal_power,
    #         thermal_grid_energy_dlmp_der_thermal_power=thermal_grid_energy_dlmp_der_thermal_power,
    #         thermal_grid_head_dlmp_der_thermal_power=thermal_grid_head_dlmp_der_thermal_power,
    #         thermal_grid_congestion_dlmp_der_thermal_power=thermal_grid_congestion_dlmp_der_thermal_power,
    #         thermal_grid_pump_dlmp_der_thermal_power=thermal_grid_pump_dlmp_der_thermal_power,
    #         thermal_grid_total_dlmp_der_thermal_power=thermal_grid_total_dlmp_der_thermal_power,
    #         thermal_grid_total_dlmp_price_timeseries=thermal_grid_total_dlmp_price_timeseries
    #     )

    def get_optimization_results(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem
    ) -> GasGridOperationResults:

        # Obtain results.
        der_gas_consumption_vector_per_unit = (
            optimization_problem.results['der_gas_consumption_vector'].loc[
                self.gas_grid_model.timesteps, self.gas_grid_model.ders
            ]
        )
        der_gas_consumption_vector = (
            der_gas_consumption_vector_per_unit
            * (self.gas_grid_model.der_gas_consumption_vector_reference)
        )
        node_pressure_vector_per_unit = (
            optimization_problem.results['node_pressure_vector'].loc[
                self.gas_grid_model.timesteps, self.gas_grid_model.nodes
            ]
        )
        node_pressure_vector = (
            node_pressure_vector_per_unit
            * (self.gas_grid_model.node_pressure_vector_reference)
        )
        branch_flow_vector_per_unit = (
            optimization_problem.results['branch_flow_vector'].loc[
                self.gas_grid_model.timesteps, self.gas_grid_model.branches
            ]
        )
        branch_flow_vector = (
            branch_flow_vector_per_unit
            * self.gas_grid_model.branch_flow_vector_reference
        )

        return GasGridOperationResults(
            gas_grid_model=self.gas_grid_model,
            der_gas_consumption_vector=der_gas_consumption_vector,
            der_gas_consumption_vector_per_unit=der_gas_consumption_vector_per_unit,
            node_pressure_vector=node_pressure_vector,
            node_pressure_vector_per_unit=node_pressure_vector_per_unit,
            branch_flow_vector=branch_flow_vector,
            branch_flow_vector_per_unit=branch_flow_vector_per_unit
        )
