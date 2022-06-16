"""Thermal grid models module."""

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
import mesmo.solutions
import mesmo.utils

logger = mesmo.config.get_logger(__name__)


class ThermalGridModel(mesmo.utils.ObjectBase):
    """Thermal grid model object."""

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
    der_thermal_power_vector_reference: np.ndarray
    branch_flow_vector_reference: np.ndarray
    node_head_vector_reference: np.ndarray
    node_head_source_value: float
    node_head_vector_reference_no_source: np.ndarray
    node_head_vector_reference_source: np.ndarray
    node_incidence_matrix_no_source: sp.spmatrix
    node_incidence_matrix_source: sp.spmatrix
    # TODO: Revise / reduce use of parameter attributes if possible.
    line_parameters: pd.DataFrame
    energy_transfer_station_head_loss: float
    enthalpy_difference_distribution_water: float
    distribution_pump_efficiency: float
    source_der_model: mesmo.der_models.DERModel
    plant_efficiency: float

    def __init__(self, scenario_name: str):

        # Obtain thermal grid data.
        thermal_grid_data = mesmo.data_interface.ThermalGridData(scenario_name)

        # Obtain index set for time steps.
        # - This is needed for optimization problem definitions within linear thermal grid models.
        self.timesteps = thermal_grid_data.scenario_data.timesteps

        # Obtain node / line / DER names.
        self.node_names = pd.Index(thermal_grid_data.thermal_grid_nodes["node_name"])
        self.line_names = pd.Index(thermal_grid_data.thermal_grid_lines["line_name"])
        self.der_names = pd.Index(thermal_grid_data.thermal_grid_ders["der_name"])
        self.der_types = pd.Index(thermal_grid_data.thermal_grid_ders["der_type"]).unique()

        # Obtain node / branch / DER index set.
        nodes = pd.concat(
            [
                thermal_grid_data.thermal_grid_nodes.loc[:, "node_name"]
                .apply(
                    # Obtain `node_type` column.
                    lambda value: "source"
                    if value == thermal_grid_data.thermal_grid.at["source_node_name"]
                    else "no_source"
                )
                .rename("node_type"),
                thermal_grid_data.thermal_grid_nodes.loc[:, "node_name"],
            ],
            axis="columns",
        )
        self.nodes = pd.MultiIndex.from_frame(nodes)
        self.branches = pd.MultiIndex.from_product([self.line_names, ["no_loop"]], names=["branch_name", "loop_type"])
        self.branch_loops = pd.MultiIndex.from_tuples([], names=["loop_id", "branch_name"])  # Values are filled below.
        self.ders = pd.MultiIndex.from_frame(thermal_grid_data.thermal_grid_ders[["der_type", "der_name"]])

        # Instantiate branch-to-node incidence matrices.
        self.branch_incidence_1_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=int)
        self.branch_incidence_2_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=int)

        # Add lines to branch incidence matrices and identify any loops.
        # - Uses temporary node tree variable to track construction of the network and identify any loops / cycles.
        branches_loops = self.branches.to_frame()
        node_trees = []
        for line_index, line in thermal_grid_data.thermal_grid_lines.iterrows():

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
        for der_name, der in thermal_grid_data.thermal_grid_ders.iterrows():

            # Obtain indexes for positioning the DER in the incidence matrix.
            node_index = mesmo.utils.get_index(self.nodes, node_name=der["node_name"])
            der_index = mesmo.utils.get_index(self.ders, der_name=der["der_name"])

            # Insert connection indicator into incidence matrices.
            self.der_node_incidence_matrix[node_index, der_index] = 1

        # Convert DOK matrices to CSR matrices.
        self.der_node_incidence_matrix = self.der_node_incidence_matrix.tocsr()

        # Obtain DER nominal thermal power vector.
        self.der_thermal_power_vector_reference = thermal_grid_data.thermal_grid_ders.loc[
            :, "thermal_power_nominal"
        ].values

        # Obtain nominal branch flow vector.
        self.branch_flow_vector_reference = (
            np.pi
            * (thermal_grid_data.thermal_grid_lines.loc[:, "diameter"].values / 2) ** 2
            * thermal_grid_data.thermal_grid_lines.loc[:, "maximum_velocity"].values
        )

        # Obtain nominal branch flow vector.
        # TODO: Define proper node head reference vector.
        self.node_head_vector_reference = np.ones(len(self.nodes))
        self.node_head_source_value = 0.0

        # Obtain line parameters.
        self.line_parameters = thermal_grid_data.thermal_grid_lines.loc[:, ["length", "diameter", "absolute_roughness"]]

        # Obtain other system parameters.
        self.energy_transfer_station_head_loss = float(
            thermal_grid_data.thermal_grid["energy_transfer_station_head_loss"]
        )
        self.enthalpy_difference_distribution_water = float(
            thermal_grid_data.thermal_grid["enthalpy_difference_distribution_water"]
        )
        self.distribution_pump_efficiency = float(thermal_grid_data.thermal_grid["distribution_pump_efficiency"])

        # Obtain DER model source node.
        # TODO: Use state space model for simulation / optimization.
        self.source_der_model = mesmo.der_models.make_der_model(
            thermal_grid_data.thermal_grid.at["source_der_model_name"], thermal_grid_data.der_data, is_standalone=True
        )
        # TODO: Remove temporary workaround: Obtain efficiency factors.
        if thermal_grid_data.thermal_grid.at["source_der_type"] == "cooling_plant":
            self.plant_efficiency = self.source_der_model.cooling_plant_efficiency
        elif thermal_grid_data.thermal_grid.at["source_der_type"] == "heating_plant":
            self.plant_efficiency = self.source_der_model.thermal_efficiency
        else:
            raise ValueError(f"Incompatible der model type: {thermal_grid_data.thermal_grid.at['source_der_type']}")

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
        self.node_head_vector_reference_no_source = self.node_head_vector_reference[
            mesmo.utils.get_index(self.nodes, node_type="no_source")
        ]
        self.node_head_vector_reference_source = self.node_head_vector_reference[
            mesmo.utils.get_index(self.nodes, node_type="source")
        ]

    def get_branch_loss_coefficient_vector(self, branch_flow_vector: np.ndarray):

        # Obtain branch velocity vector.
        branch_velocity_vector = (
            4.0 * branch_flow_vector / (np.pi * self.line_parameters.loc[:, "diameter"].values ** 2)
        )

        # Obtain branch Reynolds coefficient vector.
        branch_reynold_vector = (
            np.abs(branch_velocity_vector)
            * self.line_parameters.loc[:, "diameter"].values
            / mesmo.config.water_kinematic_viscosity
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
                friction_factor = (
                    1.325 / (np.log((absolute_roughness / 1000) / (3.7 * diameter) + 5.74 / (reynold**0.9))) ** 2
                )

            else:
                raise ValueError(f"Invalid Reynolds coefficient: {reynold}")

            # Convert from 1/m to 1/km.
            friction_factor *= 1.0e3

            return friction_factor

        # Obtain branch head loss coefficient vector.
        branch_loss_coefficient_vector = (
            branch_friction_factor_vector(
                branch_reynold_vector,
                self.line_parameters.loc[:, "absolute_roughness"].values,
                self.line_parameters.loc[:, "diameter"].values,
            )
            * 8.0
            * self.line_parameters.loc[:, "length"].values
            / (
                mesmo.config.gravitational_acceleration
                * self.line_parameters.loc[:, "diameter"].values ** 5
                * np.pi**2
            )
        )

        return branch_loss_coefficient_vector


class ThermalGridDEROperationResults(mesmo.utils.ResultsBase):

    der_thermal_power_vector: pd.DataFrame
    der_thermal_power_vector_per_unit: pd.DataFrame


class ThermalGridOperationResults(ThermalGridDEROperationResults):

    thermal_grid_model: ThermalGridModel
    node_head_vector: pd.DataFrame
    node_head_vector_per_unit: pd.DataFrame
    branch_flow_vector: pd.DataFrame
    branch_flow_vector_per_unit: pd.DataFrame
    pump_power: pd.DataFrame


class ThermalGridDLMPResults(mesmo.utils.ResultsBase):

    thermal_grid_energy_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_head_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_congestion_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_pump_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_energy_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_head_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_congestion_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_pump_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_price_timeseries: pd.DataFrame


class ThermalPowerFlowSolutionBase(mesmo.utils.ObjectBase):
    """Thermal grid power flow solution object."""

    der_thermal_power_vector: np.ndarray
    node_head_vector: np.ndarray
    branch_flow_vector: np.ndarray
    pump_power: float

    @multimethod
    def __init__(self, scenario_name: str):

        # Obtain thermal grid model.
        thermal_grid_model = ThermalGridModel(scenario_name)

        self.__init__(thermal_grid_model)

    @multimethod
    def __init__(self, thermal_grid_model: ThermalGridModel):

        # Obtain DER thermal power vector.
        der_thermal_power_vector = thermal_grid_model.der_thermal_power_vector_reference

        self.__init__(thermal_grid_model, der_thermal_power_vector)

    @multimethod
    def __init__(self, thermal_grid_model: ThermalGridModel, der_thermal_power_vector: np.ndarray):
        raise NotImplementedError


class ThermalPowerFlowSolutionExplicit(ThermalPowerFlowSolutionBase):

    # Enable calls to `__init__` method definitions in parent class.
    @multimethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @multimethod
    def __init__(self, thermal_grid_model: ThermalGridModel, der_thermal_power_vector: np.ndarray):

        # Obtain DER thermal power vector.
        self.der_thermal_power_vector = der_thermal_power_vector.ravel()
        # Define shorthand for DER volume flow vector.
        der_flow_vector = (
            self.der_thermal_power_vector
            / mesmo.config.water_density
            / thermal_grid_model.enthalpy_difference_distribution_water
        )

        # Obtain branch volume flow vector.
        self.branch_flow_vector = (
            scipy.sparse.linalg.spsolve(
                thermal_grid_model.branch_incidence_matrix_no_source.transpose(),
                thermal_grid_model.der_node_incidence_matrix_no_source @ np.transpose([der_flow_vector]),
            )
        ).ravel()

        # Obtain node head vector.
        node_head_vector_no_source = scipy.sparse.linalg.spsolve(
            thermal_grid_model.branch_incidence_matrix_no_source.tocsc(),
            (
                thermal_grid_model.get_branch_loss_coefficient_vector(self.branch_flow_vector)
                * self.branch_flow_vector
                * np.abs(self.branch_flow_vector)
            ),
        )
        self.node_head_vector = (
            thermal_grid_model.node_incidence_matrix_no_source @ node_head_vector_no_source
            + thermal_grid_model.node_incidence_matrix_source
            @ thermal_grid_model.node_head_vector_reference_source
            * thermal_grid_model.node_head_source_value
        )

        # Obtain pump power loss.
        self.pump_power = (
            (2.0 * np.max(np.abs(self.node_head_vector)) + thermal_grid_model.energy_transfer_station_head_loss)
            * -1.0
            * np.sum(der_flow_vector)  # Source volume flow.
            * mesmo.config.water_density
            * mesmo.config.gravitational_acceleration
            / thermal_grid_model.distribution_pump_efficiency
        )


class ThermalPowerFlowSolutionNewtonRaphson(ThermalPowerFlowSolutionBase):

    # Enable calls to `__init__` method definitions in parent class.
    @multimethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @multimethod
    def __init__(
        self,
        thermal_grid_model: ThermalGridModel,
        der_thermal_power_vector: np.ndarray,
        head_iteration_limit=100,
        head_tolerance=1e-2,
    ):

        # Obtain DER thermal power vector and DER volume flow vector.
        self.der_thermal_power_vector = der_thermal_power_vector.ravel()
        der_flow_vector = (
            self.der_thermal_power_vector
            / mesmo.config.water_density
            / thermal_grid_model.enthalpy_difference_distribution_water
        )

        # Obtain nodal volume flow vector.
        node_flow_vector_no_source = (
            thermal_grid_model.der_node_incidence_matrix_no_source @ np.transpose([der_flow_vector])
        ).ravel()

        # Obtain initial nodal head and branch volume flow vectors for first iteration.
        # TODO: Enable passing previous solution for initialization.
        node_head_vector_initial_no_source = thermal_grid_model.node_head_vector_reference_no_source.copy()
        branch_flow_vector_initial = thermal_grid_model.branch_flow_vector_reference.copy()

        # Instantiate Newton-Raphson iteration variables.
        head_iteration = 0
        head_change = np.inf

        # Run Newton-Raphson iterations.
        while (head_iteration < head_iteration_limit) & (head_change > head_tolerance):

            # Detect zero branch volume flows.
            branch_flow_vector_valid_index = branch_flow_vector_initial != 0.0

            # Replace zero branch volume flows with very small value, based on minium absolute branch volume flow.
            # - This is to avoid numerical issues due to singularity of the jacobian matrix.
            branch_flow_abs_min = np.min(np.abs(branch_flow_vector_initial[branch_flow_vector_valid_index]))
            if branch_flow_abs_min == 0.0:
                branch_flow_abs_min = 1e-9
            else:
                branch_flow_abs_min *= 1e-9
            branch_flow_vector_initial[~branch_flow_vector_valid_index] = branch_flow_abs_min

            # Calculate branch loss coefficient and jacobian matrix.
            branch_loss_coefficient_vector = thermal_grid_model.get_branch_loss_coefficient_vector(
                branch_flow_vector_initial
            )
            jacobian_branch_head_loss = (
                2 * sp.diags(np.abs(branch_flow_vector_initial)) @ sp.diags(branch_loss_coefficient_vector)
            )
            jacobian_branch_head_loss_inverse = (
                0.5
                * sp.diags(np.abs(branch_flow_vector_initial) ** -1)
                @ sp.diags(branch_loss_coefficient_vector**-1)
            )

            # Calculate nodal head vector.
            node_head_vector_estimate_no_source = scipy.sparse.linalg.spsolve(
                (
                    np.transpose(thermal_grid_model.branch_incidence_matrix_no_source)
                    @ jacobian_branch_head_loss_inverse
                    @ thermal_grid_model.branch_incidence_matrix_no_source
                ),
                (
                    (
                        -1.0
                        * np.transpose(thermal_grid_model.branch_incidence_matrix_no_source)
                        @ jacobian_branch_head_loss_inverse
                    )
                    @ (
                        (0.5 * jacobian_branch_head_loss @ branch_flow_vector_initial)
                        - (
                            -1.0
                            * thermal_grid_model.branch_incidence_matrix_source
                            @ thermal_grid_model.node_head_vector_reference_source
                            * thermal_grid_model.node_head_source_value
                        )
                    )
                    + node_flow_vector_no_source
                ),
            )
            node_head_vector_estimate = (
                thermal_grid_model.node_incidence_matrix_no_source @ node_head_vector_estimate_no_source
                + thermal_grid_model.node_incidence_matrix_source
                @ thermal_grid_model.node_head_vector_reference_source
                * thermal_grid_model.node_head_source_value
            )

            # Calculate branch volume flow vector.
            branch_flow_vector_estimate = branch_flow_vector_initial - jacobian_branch_head_loss_inverse @ (
                (0.5 * jacobian_branch_head_loss @ branch_flow_vector_initial)
                + (-1.0 * thermal_grid_model.branch_incidence_matrix @ node_head_vector_estimate)
            )

            # Update head change iteration variable.
            head_change = np.max(np.abs(node_head_vector_estimate_no_source - node_head_vector_initial_no_source))

            # Update initial values for next iteration.
            node_head_vector_initial_no_source = node_head_vector_estimate_no_source.copy()
            branch_flow_vector_initial = branch_flow_vector_estimate.copy()

            # Update iteration counter.
            head_iteration += 1

        # For fixed-point algorithm, reaching the iteration limit is considered undesired and triggers a warning
        if head_iteration >= head_iteration_limit:
            logger.warning(
                "Newton-Raphson solution algorithm reached " f"maximum limit of {head_iteration_limit} iterations."
            )

        # Obtain node head vector.
        self.node_head_vector = node_head_vector_estimate

        # Obtain branch volume flow vector.
        self.branch_flow_vector = branch_flow_vector_estimate

        # Obtain pump power loss.
        self.pump_power = (
            (2.0 * np.max(np.abs(self.node_head_vector)) + thermal_grid_model.energy_transfer_station_head_loss)
            * -1.0
            * np.sum(der_flow_vector)  # Source volume flow.
            * mesmo.config.water_density
            * mesmo.config.gravitational_acceleration
            / thermal_grid_model.distribution_pump_efficiency
        )


class ThermalPowerFlowSolution(ThermalPowerFlowSolutionBase):
    """Thermal grid power flow solution object."""

    # Enable calls to `__init__` method definitions in parent class.
    @multimethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @multimethod
    def __init__(self, thermal_grid_model: ThermalGridModel, der_thermal_power_vector: np.ndarray):

        # Select power flow solution method, depending on whether network is radial or meshed.
        if len(thermal_grid_model.branch_loops) == 0:
            # Use explicit thermal power flow solution method.
            ThermalPowerFlowSolutionExplicit.__init__(self, thermal_grid_model, der_thermal_power_vector)
        else:
            # Use Newton-Raphson method.
            ThermalPowerFlowSolutionNewtonRaphson.__init__(self, thermal_grid_model, der_thermal_power_vector)


class ThermalPowerFlowSolutionSet(mesmo.utils.ObjectBase):

    power_flow_solutions: typing.Dict[pd.Timestamp, ThermalPowerFlowSolution]
    thermal_grid_model: ThermalGridModel
    der_thermal_power_vector: pd.DataFrame
    timesteps: pd.Index

    @multimethod
    def __init__(
        self, thermal_grid_model: ThermalGridModel, der_operation_results: ThermalGridDEROperationResults, **kwargs
    ):

        der_thermal_power_vector = der_operation_results.der_thermal_power_vector

        self.__init__(thermal_grid_model, der_thermal_power_vector, **kwargs)

    @multimethod
    def __init__(
        self,
        thermal_grid_model: ThermalGridModel,
        der_thermal_power_vector: pd.DataFrame,
        power_flow_solution_method=ThermalPowerFlowSolution,
    ):

        # Store attributes.
        self.thermal_grid_model = thermal_grid_model
        self.der_thermal_power_vector = der_thermal_power_vector
        self.timesteps = self.thermal_grid_model.timesteps

        # Obtain power flow solutions.
        power_flow_solutions = mesmo.utils.starmap(
            power_flow_solution_method, zip(itertools.repeat(self.thermal_grid_model), der_thermal_power_vector.values)
        )
        self.power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))

    def get_results(self) -> ThermalGridOperationResults:

        raise NotImplementedError


class LinearThermalGridModelBase(mesmo.utils.ObjectBase):
    """Linear thermal grid model object."""

    thermal_grid_model: ThermalGridModel
    thermal_power_flow_solution: ThermalPowerFlowSolution
    sensitivity_branch_flow_by_node_power: sp.spmatrix
    sensitivity_branch_flow_by_der_power: sp.spmatrix
    sensitivity_node_head_by_node_power: sp.spmatrix
    sensitivity_node_head_by_der_power: sp.spmatrix
    sensitivity_pump_power_by_node_power: np.array
    sensitivity_pump_power_by_der_power: np.array

    @multimethod
    def __init__(
        self,
        scenario_name: str,
    ):

        # Obtain thermal grid model.
        thermal_grid_model = ThermalGridModel(scenario_name)

        # Obtain DER power vector.
        der_thermal_power_vector = thermal_grid_model.der_thermal_power_vector_reference

        # Obtain thermal power flow solution.
        thermal_power_flow_solution = ThermalPowerFlowSolution(thermal_grid_model, der_thermal_power_vector)

        self.__init__(thermal_grid_model, thermal_power_flow_solution)


class LinearThermalGridModelGlobal(LinearThermalGridModelBase):

    # Enable calls to `__init__` method definitions in parent class.
    @multimethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @multimethod
    def __init__(
        self,
        thermal_grid_model: ThermalGridModel,
        thermal_power_flow_solution: ThermalPowerFlowSolution,
    ):

        # Store thermal grid model.
        self.thermal_grid_model = thermal_grid_model

        # Store thermal power flow solution.
        self.thermal_power_flow_solution = thermal_power_flow_solution

        # Obtain linearization reference point.
        der_power_vector_reference = self.thermal_power_flow_solution.der_thermal_power_vector
        branch_flow_vector_reference = self.thermal_power_flow_solution.branch_flow_vector.copy()

        # Replace zero branch volume flows with very small value, based on minium absolute branch volume flow.
        # - This is to avoid numerical issues due to singularity of the jacobian matrix.
        branch_flow_vector_valid_index = branch_flow_vector_reference != 0.0
        branch_flow_abs_min = np.min(np.abs(branch_flow_vector_reference[branch_flow_vector_valid_index]))
        if branch_flow_abs_min == 0.0:
            branch_flow_abs_min = 1e-9
        else:
            branch_flow_abs_min *= 1e-9
        branch_flow_vector_reference[~branch_flow_vector_valid_index] = branch_flow_abs_min

        # Calculate branch loss coefficient and jacobian matrix.
        branch_loss_coefficient_vector = thermal_grid_model.get_branch_loss_coefficient_vector(
            branch_flow_vector_reference
        )
        jacobian_branch_head_loss_inverse = (
            0.5 * sp.diags(np.abs(branch_flow_vector_reference) ** -1) @ sp.diags(branch_loss_coefficient_vector**-1)
        )

        # Obtain sensitivity matrices.
        self.sensitivity_node_head_by_node_power = sp.dok_matrix(
            (len(thermal_grid_model.nodes), len(thermal_grid_model.nodes)), dtype=float
        )
        self.sensitivity_node_head_by_node_power[
            np.ix_(
                mesmo.utils.get_index(thermal_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(thermal_grid_model.nodes, node_type="no_source"),
            )
        ] = (
            (2.0**-1.5)
            * scipy.sparse.linalg.inv(
                np.transpose(thermal_grid_model.branch_incidence_matrix_no_source)
                @ jacobian_branch_head_loss_inverse
                @ thermal_grid_model.branch_incidence_matrix_no_source
            )
            * self.thermal_grid_model.enthalpy_difference_distribution_water
        )
        self.sensitivity_node_head_by_node_power = self.sensitivity_node_head_by_node_power.tocsr()
        self.sensitivity_node_head_by_der_power = (
            self.sensitivity_node_head_by_node_power @ self.thermal_grid_model.der_node_incidence_matrix
        )
        self.sensitivity_branch_flow_by_node_power = (
            2.0
            * jacobian_branch_head_loss_inverse
            @ thermal_grid_model.branch_incidence_matrix
            @ self.sensitivity_node_head_by_node_power
        )

        self.sensitivity_branch_flow_by_der_power = (
            self.sensitivity_branch_flow_by_node_power @ self.thermal_grid_model.der_node_incidence_matrix
        )
        # TODO: Revise pump power sensitivity equation.
        self.sensitivity_pump_power_by_node_power = (
            (
                -1.0
                * der_power_vector_reference
                / mesmo.config.water_density
                / thermal_grid_model.enthalpy_difference_distribution_water
            )  # DER volume flow vector.
            @ (-2.0 * np.transpose(self.thermal_grid_model.der_node_incidence_matrix))
            @ self.sensitivity_node_head_by_node_power
            * mesmo.config.water_density
            * mesmo.config.gravitational_acceleration
            / self.thermal_grid_model.distribution_pump_efficiency
        ) + (
            -1.0
            * self.thermal_grid_model.energy_transfer_station_head_loss
            * mesmo.config.gravitational_acceleration
            / self.thermal_grid_model.enthalpy_difference_distribution_water
            / self.thermal_grid_model.distribution_pump_efficiency
        )
        self.sensitivity_pump_power_by_der_power = np.array(
            [self.sensitivity_pump_power_by_node_power @ self.thermal_grid_model.der_node_incidence_matrix]
        )


class LinearThermalGridModelLocal(LinearThermalGridModelGlobal):

    # Enable calls to `__init__` method definitions in parent class.
    @multimethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @multimethod
    def __init__(
        self,
        thermal_grid_model: ThermalGridModel,
        thermal_power_flow_solution: ThermalPowerFlowSolution,
    ):

        # Initialize linear model from global approximation method.
        super().__init__(thermal_grid_model, thermal_power_flow_solution)

        # Modify sensitivities for local approximation method.
        self.sensitivity_node_head_by_node_power *= 2.0
        self.sensitivity_node_head_by_der_power *= 2.0


class LinearThermalGridModelSet(mesmo.utils.ObjectBase):

    linear_thermal_grid_models: typing.Dict[pd.Timestamp, LinearThermalGridModelBase]
    thermal_grid_model: ThermalGridModel
    timesteps: pd.Index

    @multimethod
    def __init__(self, scenario_name: str):

        # Obtain electric grid model & reference power flow solution.
        thermal_grid_model = ThermalGridModel(scenario_name)
        thermal_power_flow_solution = ThermalPowerFlowSolution(thermal_grid_model)

        self.__init__(thermal_grid_model, thermal_power_flow_solution)

    @multimethod
    def __init__(
        self,
        thermal_grid_model: ThermalGridModel,
        thermal_power_flow_solution_set: ThermalPowerFlowSolutionSet,
        linear_thermal_grid_model_method: typing.Type[LinearThermalGridModelBase] = LinearThermalGridModelGlobal,
    ):

        self.check_linear_thermal_grid_model_method(linear_thermal_grid_model_method)

        # Obtain linear thermal grid models.
        linear_thermal_grid_models = mesmo.utils.starmap(
            linear_thermal_grid_model_method,
            zip(itertools.repeat(thermal_grid_model), thermal_power_flow_solution_set.power_flow_solutions.values()),
        )
        linear_thermal_grid_models = dict(zip(thermal_grid_model.timesteps, linear_thermal_grid_models))

        self.__init__(thermal_grid_model, linear_thermal_grid_models)

    @multimethod
    def __init__(
        self,
        thermal_grid_model: ThermalGridModel,
        thermal_power_flow_solution: ThermalPowerFlowSolution,
        linear_thermal_grid_model_method: typing.Type[LinearThermalGridModelBase] = LinearThermalGridModelGlobal,
    ):

        self.check_linear_thermal_grid_model_method(linear_thermal_grid_model_method)

        # Obtain linear thermal grid models.
        linear_thermal_grid_model = LinearThermalGridModelGlobal(thermal_grid_model, thermal_power_flow_solution)
        linear_thermal_grid_models = dict(
            zip(thermal_grid_model.timesteps, itertools.repeat(linear_thermal_grid_model))
        )

        self.__init__(thermal_grid_model, linear_thermal_grid_models)

    @multimethod
    def __init__(
        self,
        thermal_grid_model: ThermalGridModel,
        linear_thermal_grid_models: typing.Dict[pd.Timestamp, LinearThermalGridModelBase],
    ):

        # Store attributes.
        self.thermal_grid_model = thermal_grid_model
        self.timesteps = self.thermal_grid_model.timesteps
        self.linear_thermal_grid_models = linear_thermal_grid_models

    @staticmethod
    def check_linear_thermal_grid_model_method(linear_thermal_grid_model_method):

        if not issubclass(linear_thermal_grid_model_method, LinearThermalGridModelBase):
            raise ValueError(f"Invalid linear thermal grid model method: {linear_thermal_grid_model_method}")

    def define_optimization_problem(
        self,
        optimization_problem: mesmo.solutions.OptimizationProblem,
        price_data: mesmo.data_interface.PriceData,
        scenarios: typing.Union[list, pd.Index] = None,
        **kwargs,
    ):

        # Defined optimization problem definitions through respective sub-methods.
        self.define_optimization_variables(optimization_problem, scenarios=scenarios)
        self.define_optimization_parameters(optimization_problem, price_data, scenarios=scenarios, **kwargs)
        self.define_optimization_constraints(optimization_problem, scenarios=scenarios)
        self.define_optimization_objective(optimization_problem, scenarios=scenarios)

    def define_optimization_variables(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Define DER power vector variables.
        optimization_problem.define_variable(
            "der_thermal_power_vector", scenario=scenarios, timestep=self.timesteps, der=self.thermal_grid_model.ders
        )

        # Define node head, branch flow and pump power variables.
        optimization_problem.define_variable(
            "node_head_vector", scenario=scenarios, timestep=self.timesteps, node=self.thermal_grid_model.nodes
        )
        optimization_problem.define_variable(
            "branch_flow_vector", scenario=scenarios, timestep=self.timesteps, branch=self.thermal_grid_model.branches
        )
        optimization_problem.define_variable("pump_power", scenario=scenarios, timestep=self.timesteps)

    def define_optimization_parameters(
        self,
        optimization_problem: mesmo.solutions.OptimizationProblem,
        price_data: mesmo.data_interface.PriceData,
        node_head_vector_minimum: np.ndarray = None,
        branch_flow_vector_maximum: np.ndarray = None,
        scenarios: typing.Union[list, pd.Index] = None,
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta("1h")

        # Define head variable term.
        optimization_problem.define_parameter(
            "head_variable",
            sp.block_diag(
                [
                    sp.diags(linear_thermal_grid_model.thermal_grid_model.node_head_vector_reference**-1)
                    @ linear_thermal_grid_model.sensitivity_node_head_by_der_power
                    @ sp.diags(linear_thermal_grid_model.thermal_grid_model.der_thermal_power_vector_reference)
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            ),
        )

        # Define head constant term.
        optimization_problem.define_parameter(
            "head_constant",
            np.concatenate(
                [
                    sp.diags(linear_thermal_grid_model.thermal_grid_model.node_head_vector_reference**-1)
                    @ (
                        np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.node_head_vector])
                        - linear_thermal_grid_model.sensitivity_node_head_by_der_power
                        @ np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.der_thermal_power_vector])
                    )
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            ),
        )

        # Define branch flow variable term.
        optimization_problem.define_parameter(
            "branch_flow_variable",
            sp.block_diag(
                [
                    sp.diags(linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference**-1)
                    @ linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                    @ sp.diags(linear_thermal_grid_model.thermal_grid_model.der_thermal_power_vector_reference)
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            ),
        )

        # Define branch flow constant term.
        optimization_problem.define_parameter(
            "branch_flow_constant",
            np.concatenate(
                [
                    sp.diags(linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference**-1)
                    @ (
                        np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.branch_flow_vector])
                        - linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                        @ np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.der_thermal_power_vector])
                    )
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            ),
        )

        # Define pump power variable term.
        optimization_problem.define_parameter(
            "pump_power_variable",
            sp.block_diag(
                [
                    linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                    @ sp.diags(linear_thermal_grid_model.thermal_grid_model.der_thermal_power_vector_reference)
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            ),
        )

        # Define pump power constant term.
        optimization_problem.define_parameter(
            "pump_power_constant",
            np.concatenate(
                [
                    # TODO: Fix pump power sensitivity.
                    [0.0]
                    # linear_thermal_grid_model.thermal_power_flow_solution.pump_power
                    # - linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                    # @ np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.der_thermal_power_vector])
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            ),
        )

        # Define head limits.
        optimization_problem.define_parameter(
            "node_head_minimum",
            np.concatenate(
                [
                    node_head_vector_minimum.ravel()
                    / linear_thermal_grid_model.thermal_grid_model.node_head_vector_reference
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            )
            if node_head_vector_minimum is not None
            else -np.inf * np.ones((len(self.thermal_grid_model.nodes) * len(self.timesteps),)),
        )

        # Define branch flow limits.
        optimization_problem.define_parameter(
            "branch_flow_minimum",
            np.concatenate(
                [
                    -branch_flow_vector_maximum.ravel()
                    / linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            )
            if branch_flow_vector_maximum is not None
            else -np.inf * np.ones((len(self.thermal_grid_model.branches) * len(self.timesteps),)),
        )
        optimization_problem.define_parameter(
            "branch_flow_maximum",
            np.concatenate(
                [
                    branch_flow_vector_maximum.ravel()
                    / linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference
                    for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
                ]
            )
            if branch_flow_vector_maximum is not None
            else +np.inf * np.ones((len(self.thermal_grid_model.branches) * len(self.timesteps),)),
        )

        # Define objective parameters.
        optimization_problem.define_parameter(
            "thermal_grid_thermal_power_cost",
            np.array([price_data.price_timeseries.loc[:, ("thermal_power", "source", "source")].values])
            * -1.0
            * timestep_interval_hours  # In Wh.
            / self.thermal_grid_model.plant_efficiency
            @ sp.block_diag(
                [np.array([self.thermal_grid_model.der_thermal_power_vector_reference])] * len(self.timesteps)
            ),
        )
        optimization_problem.define_parameter(
            "thermal_grid_thermal_power_cost_sensitivity",
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * np.concatenate([self.thermal_grid_model.der_thermal_power_vector_reference**2] * len(self.timesteps)),
        )
        optimization_problem.define_parameter(
            "thermal_grid_pump_power_cost",
            price_data.price_timeseries.loc[:, ("thermal_power", "source", "source")].values
            * timestep_interval_hours,  # In Wh.
        )
        optimization_problem.define_parameter(
            "thermal_grid_pump_power_cost_sensitivity",
            price_data.price_sensitivity_coefficient * timestep_interval_hours,  # In Wh.
        )

    def define_optimization_constraints(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Define head equation.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="node_head_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    node=self.thermal_grid_model.nodes,
                ),
            ),
            "==",
            (
                "variable",
                "head_variable",
                dict(
                    name="der_thermal_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.thermal_grid_model.ders,
                ),
            ),
            ("constant", "head_constant", dict(scenario=scenarios, timestep=self.timesteps)),
            broadcast="scenario",
        )

        # Define branch flow equation.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_flow_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.thermal_grid_model.branches,
                ),
            ),
            "==",
            (
                "variable",
                "branch_flow_variable",
                dict(
                    name="der_thermal_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.thermal_grid_model.ders,
                ),
            ),
            ("constant", "branch_flow_constant", dict(scenario=scenarios, timestep=self.timesteps)),
        )

        # Define pump power equation.
        optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="pump_power", scenario=scenarios, timestep=self.timesteps)),
            "==",
            (
                "variable",
                "pump_power_variable",
                dict(
                    name="der_thermal_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.thermal_grid_model.ders,
                ),
            ),
            ("constant", "pump_power_constant", dict(scenario=scenarios, timestep=self.timesteps)),
            broadcast="scenario",
        )

        # Define head limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="node_head_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    node=self.thermal_grid_model.nodes,
                ),
            ),
            ">=",
            ("constant", "node_head_minimum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="node_head_vector_minimum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                node=self.thermal_grid_model.nodes,
            ),
            broadcast="scenario",
        )

        # Define branch flow limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_flow_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.thermal_grid_model.branches,
                ),
            ),
            ">=",
            ("constant", "branch_flow_minimum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="branch_flow_vector_minimum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                branch=self.thermal_grid_model.branches,
            ),
            broadcast="scenario",
        )
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_flow_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.thermal_grid_model.branches,
                ),
            ),
            "<=",
            ("constant", "branch_flow_maximum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="branch_flow_vector_maximum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                branch=self.thermal_grid_model.branches,
            ),
            broadcast="scenario",
        )

    def define_optimization_objective(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Set objective flag.
        optimization_problem.flags["has_thermal_grid_objective"] = True

        # Define objective for thermal loads.
        # - Defined as cost of thermal supply at thermal grid source node.
        # - Only defined here, if not yet defined as cost of thermal power supply at the DER node
        #   in `mesmo.der_models.DERModel.define_optimization_objective`.
        if not optimization_problem.flags.get("has_der_objective"):

            # Thermal power cost / revenue.
            # - Cost for load / demand, revenue for generation / supply.
            optimization_problem.define_objective(
                (
                    "variable",
                    "thermal_grid_thermal_power_cost",
                    dict(
                        name="der_thermal_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.thermal_grid_model.ders,
                    ),
                ),
                (
                    "variable",
                    "thermal_grid_thermal_power_cost_sensitivity",
                    dict(
                        name="der_thermal_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.thermal_grid_model.ders,
                    ),
                    dict(
                        name="der_thermal_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.thermal_grid_model.ders,
                    ),
                ),
                broadcast="scenario",
            )

        # Define pump power cost.
        optimization_problem.define_objective(
            (
                "variable",
                "thermal_grid_pump_power_cost",
                dict(name="pump_power", scenario=scenarios, timestep=self.timesteps),
            ),
            (
                "variable",
                "thermal_grid_pump_power_cost_sensitivity",
                dict(name="pump_power", scenario=scenarios, timestep=self.timesteps),
                dict(name="pump_power", scenario=scenarios, timestep=self.timesteps),
            ),
            broadcast="scenario",
        )

    def evaluate_optimization_objective(
        self, results: ThermalGridOperationResults, price_data: mesmo.data_interface.PriceData
    ) -> float:

        # Instantiate optimization problem.
        optimization_problem = mesmo.solutions.OptimizationProblem()
        self.define_optimization_parameters(optimization_problem, price_data)
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_objective(optimization_problem)

        # Instantiate variable vector.
        x_vector = np.zeros((len(optimization_problem.variables), 1))

        # Set variable vector values.
        objective_variable_names = ["der_thermal_power_vector_per_unit", "pump_power"]
        for variable_name in objective_variable_names:
            index = mesmo.utils.get_index(optimization_problem.variables, name=variable_name.replace("_per_unit", ""))
            x_vector[index, 0] = results[variable_name].values.ravel()

        # Obtain objective value.
        objective = optimization_problem.evaluate_objective(x_vector)

        return objective

    def get_optimization_dlmps(
        self,
        optimization_problem: mesmo.solutions.OptimizationProblem,
        price_data: mesmo.data_interface.PriceData,
        scenarios: typing.Union[list, pd.Index] = None,
    ) -> ThermalGridDLMPResults:

        # Obtain results index sets, depending on if / if not scenarios given.
        if scenarios in [None, [None]]:
            scenarios = [None]
            ders = self.thermal_grid_model.ders
            nodes = self.thermal_grid_model.nodes
            branches = self.thermal_grid_model.branches
        else:
            ders = pd.MultiIndex.from_product(
                (scenarios, self.thermal_grid_model.ders.to_flat_index()), names=["scenario", "der"]
            )
            nodes = pd.MultiIndex.from_product(
                (scenarios, self.thermal_grid_model.nodes.to_flat_index()), names=["scenario", "node"]
            )
            branches = pd.MultiIndex.from_product(
                (scenarios, self.thermal_grid_model.branches.to_flat_index()), names=["scenario", "branch"]
            )

        # Obtain individual duals.
        node_head_vector_minimum_dual = optimization_problem.duals["node_head_vector_minimum_constraint"].loc[
            self.thermal_grid_model.timesteps, nodes
        ] / np.concatenate([self.thermal_grid_model.node_head_vector_reference] * len(scenarios))
        branch_flow_vector_minimum_dual = optimization_problem.duals["branch_flow_vector_minimum_constraint"].loc[
            self.thermal_grid_model.timesteps, branches
        ] / np.concatenate([self.thermal_grid_model.branch_flow_vector_reference] * len(scenarios))
        branch_flow_vector_maximum_dual = (
            -1.0
            * optimization_problem.duals["branch_flow_vector_maximum_constraint"].loc[
                self.thermal_grid_model.timesteps, branches
            ]
            / np.concatenate([self.thermal_grid_model.branch_flow_vector_reference] * len(scenarios))
        )

        # Instantiate DLMP variables.
        thermal_grid_energy_dlmp_node_thermal_power = pd.DataFrame(
            columns=nodes, index=self.thermal_grid_model.timesteps, dtype=float
        )
        thermal_grid_head_dlmp_node_thermal_power = pd.DataFrame(
            columns=nodes, index=self.thermal_grid_model.timesteps, dtype=float
        )
        thermal_grid_congestion_dlmp_node_thermal_power = pd.DataFrame(
            columns=nodes, index=self.thermal_grid_model.timesteps, dtype=float
        )
        thermal_grid_pump_dlmp_node_thermal_power = pd.DataFrame(
            columns=nodes, index=self.thermal_grid_model.timesteps, dtype=float
        )

        thermal_grid_energy_dlmp_der_thermal_power = pd.DataFrame(
            columns=ders, index=self.thermal_grid_model.timesteps, dtype=float
        )
        thermal_grid_head_dlmp_der_thermal_power = pd.DataFrame(
            columns=ders, index=self.thermal_grid_model.timesteps, dtype=float
        )
        thermal_grid_congestion_dlmp_der_thermal_power = pd.DataFrame(
            columns=ders, index=self.thermal_grid_model.timesteps, dtype=float
        )
        thermal_grid_pump_dlmp_der_thermal_power = pd.DataFrame(
            columns=ders, index=self.thermal_grid_model.timesteps, dtype=float
        )

        # Obtain DLMPs.
        for timestep in self.thermal_grid_model.timesteps:
            thermal_grid_energy_dlmp_node_thermal_power.loc[timestep, :] = (
                price_data.price_timeseries.at[timestep, ("thermal_power", "source", "source")]
                / self.thermal_grid_model.plant_efficiency
            )
            thermal_grid_head_dlmp_node_thermal_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_thermal_grid_models[timestep].sensitivity_node_head_by_node_power] * len(scenarios)
                ).transpose()
                @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
            ).ravel()
            thermal_grid_congestion_dlmp_node_thermal_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_node_power] * len(scenarios)
                ).transpose()
                @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
            ).ravel() + (
                sp.block_diag(
                    [self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_node_power] * len(scenarios)
                ).transpose()
                @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
            ).ravel()
            thermal_grid_pump_dlmp_node_thermal_power.loc[timestep, :] = (
                -1.0
                * np.concatenate(
                    [self.linear_thermal_grid_models[timestep].sensitivity_pump_power_by_node_power.ravel()]
                    * len(scenarios)
                ).transpose()
                * price_data.price_timeseries.at[timestep, ("thermal_power", "source", "source")]
            )

            thermal_grid_energy_dlmp_der_thermal_power.loc[timestep, :] = (
                price_data.price_timeseries.at[timestep, ("thermal_power", "source", "source")]
                / self.thermal_grid_model.plant_efficiency
            )
            thermal_grid_head_dlmp_der_thermal_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_thermal_grid_models[timestep].sensitivity_node_head_by_der_power] * len(scenarios)
                ).transpose()
                @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
            ).ravel()
            thermal_grid_congestion_dlmp_der_thermal_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_der_power] * len(scenarios)
                ).transpose()
                @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
            ).ravel() + (
                sp.block_diag(
                    [self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_der_power] * len(scenarios)
                ).transpose()
                @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
            ).ravel()
            thermal_grid_pump_dlmp_der_thermal_power.loc[timestep, :] = (
                -1.0
                * np.concatenate(
                    [self.linear_thermal_grid_models[timestep].sensitivity_pump_power_by_der_power.ravel()]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("thermal_power", "source", "source")]
            )

        thermal_grid_total_dlmp_node_thermal_power = (
            thermal_grid_energy_dlmp_node_thermal_power
            + thermal_grid_head_dlmp_node_thermal_power
            + thermal_grid_congestion_dlmp_node_thermal_power
            + thermal_grid_pump_dlmp_node_thermal_power
        )
        thermal_grid_total_dlmp_der_thermal_power = (
            thermal_grid_energy_dlmp_der_thermal_power
            + thermal_grid_head_dlmp_der_thermal_power
            + thermal_grid_congestion_dlmp_der_thermal_power
            + thermal_grid_pump_dlmp_der_thermal_power
        )

        # Obtain total DLMPs in format similar to `mesmo.data_interface.PriceData.price_timeseries`.
        thermal_grid_total_dlmp_price_timeseries = pd.concat(
            [
                price_data.price_timeseries.loc[:, ("thermal_power", "source", "source")].rename(("source", "source")),
                thermal_grid_total_dlmp_der_thermal_power,
            ],
            axis="columns",
            keys=["thermal_power", "thermal_power"],
            names=["commodity_type"],
        )
        # Redefine columns to avoid slicing issues.
        thermal_grid_total_dlmp_price_timeseries.columns = price_data.price_timeseries.columns[
            price_data.price_timeseries.columns.isin(thermal_grid_total_dlmp_price_timeseries.columns)
        ]

        return ThermalGridDLMPResults(
            thermal_grid_energy_dlmp_node_thermal_power=thermal_grid_energy_dlmp_node_thermal_power,
            thermal_grid_head_dlmp_node_thermal_power=thermal_grid_head_dlmp_node_thermal_power,
            thermal_grid_congestion_dlmp_node_thermal_power=thermal_grid_congestion_dlmp_node_thermal_power,
            thermal_grid_pump_dlmp_node_thermal_power=thermal_grid_pump_dlmp_node_thermal_power,
            thermal_grid_total_dlmp_node_thermal_power=thermal_grid_total_dlmp_node_thermal_power,
            thermal_grid_energy_dlmp_der_thermal_power=thermal_grid_energy_dlmp_der_thermal_power,
            thermal_grid_head_dlmp_der_thermal_power=thermal_grid_head_dlmp_der_thermal_power,
            thermal_grid_congestion_dlmp_der_thermal_power=thermal_grid_congestion_dlmp_der_thermal_power,
            thermal_grid_pump_dlmp_der_thermal_power=thermal_grid_pump_dlmp_der_thermal_power,
            thermal_grid_total_dlmp_der_thermal_power=thermal_grid_total_dlmp_der_thermal_power,
            thermal_grid_total_dlmp_price_timeseries=thermal_grid_total_dlmp_price_timeseries,
        )

    def get_optimization_results(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ) -> ThermalGridOperationResults:

        # Obtain results index sets, depending on if / if not scenarios given.
        if scenarios in [None, [None]]:
            scenarios = [None]
            ders = self.thermal_grid_model.ders
            nodes = self.thermal_grid_model.nodes
            branches = self.thermal_grid_model.branches
            pump_power = ["pump_power"]
        else:
            ders = (scenarios, self.thermal_grid_model.ders)
            nodes = (scenarios, self.thermal_grid_model.nodes)
            branches = (scenarios, self.thermal_grid_model.branches)
            pump_power = scenarios

        # Obtain results.
        der_thermal_power_vector_per_unit = optimization_problem.results["der_thermal_power_vector"].loc[
            self.thermal_grid_model.timesteps, ders
        ]
        der_thermal_power_vector = der_thermal_power_vector_per_unit * np.concatenate(
            [self.thermal_grid_model.der_thermal_power_vector_reference] * len(scenarios)
        )
        node_head_vector_per_unit = optimization_problem.results["node_head_vector"].loc[
            self.thermal_grid_model.timesteps, nodes
        ]
        node_head_vector = node_head_vector_per_unit * np.concatenate(
            [self.thermal_grid_model.node_head_vector_reference] * len(scenarios)
        )
        branch_flow_vector_per_unit = optimization_problem.results["branch_flow_vector"].loc[
            self.thermal_grid_model.timesteps, branches
        ]
        branch_flow_vector = branch_flow_vector_per_unit * np.concatenate(
            [self.thermal_grid_model.branch_flow_vector_reference] * len(scenarios)
        )
        pump_power = optimization_problem.results["pump_power"].loc[self.thermal_grid_model.timesteps, pump_power]

        return ThermalGridOperationResults(
            thermal_grid_model=self.thermal_grid_model,
            der_thermal_power_vector=der_thermal_power_vector,
            der_thermal_power_vector_per_unit=der_thermal_power_vector_per_unit,
            node_head_vector=node_head_vector,
            node_head_vector_per_unit=node_head_vector_per_unit,
            branch_flow_vector=branch_flow_vector,
            branch_flow_vector_per_unit=branch_flow_vector_per_unit,
            pump_power=pump_power,
        )
