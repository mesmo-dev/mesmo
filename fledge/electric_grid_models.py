"""Electric grid models."""

import numpy as np
import pandas as pd
import scipy.sparse

import fledge.config
import fledge.database_interface

logger = fledge.config.get_logger(__name__)


class ElectricGridIndex(object):
    """Electric grid index object."""

    node_dimension: int
    branch_dimension: int
    load_dimension: int
    phases: pd.Index
    node_names: pd.Index
    node_types: pd.Index
    nodes_phases: pd.Index
    line_names: pd.Index
    transformer_names: pd.Index
    branch_names: pd.Index
    branch_types: pd.Index
    branches_phases: pd.Index
    load_names: pd.Index
    node_by_node_name: dict
    node_by_phase: dict
    node_by_node_type: dict
    node_by_load_name: dict
    branch_by_line_name: dict
    branch_by_transformer_name: dict
    branch_by_phase: dict
    load_by_load_name: dict

    def __init__(
            self,
            scenario_name: str = None,
            electric_grid_data: fledge.database_interface.ElectricGridData = None
    ):
        """Instantiate electric grid index object for given `electric_grid_data`"""

        # Load electric grid data, if none.
        if electric_grid_data is None:
            electric_grid_data = fledge.database_interface.ElectricGridData(scenario_name)

        # Obtain transformers for one / first winding.
        electric_grid_transformers_one_winding = (
            electric_grid_data.electric_grid_transformers.loc[
            electric_grid_data.electric_grid_transformers['winding'] == 1,
            :
            ]
        )

        # Define node dimension, i.e., number of phases of all nodes, which
        # will be the dimension of the nodal admittance matrix.
        # - The admittance matrix has one entry for each phase of each node in
        #   both dimensions.
        # - There cannot be "empty" dimensions for missing phases of nodes,
        #   because the matrix would become singular.
        # - Therefore the admittance matrix must have the exact number of existing
        #   phases of all nodes.
        self.node_dimension = (
            electric_grid_data.electric_grid_nodes.loc[
                :,
                [
                    'is_phase_1_connected',
                    'is_phase_2_connected',
                    'is_phase_3_connected'
                ]
            ].sum().sum()
        )

        # Define branch dimension, i.e., number of phases of all branches, which
        # will be the first dimension of the branch admittance matrices.
        # - Branches consider all power delivery elements, i.e., lines as well as
        #   transformers.
        # - The second dimension of the branch admittance matrices is the number of
        #   phases of all nodes.
        # - TODO: Add switches.
        line_dimension = (
            electric_grid_data.electric_grid_lines.loc[
            :,
            [
                'is_phase_1_connected',
                'is_phase_2_connected',
                'is_phase_3_connected'
            ]
            ].sum().sum()
        )
        transformer_dimension = (
            electric_grid_transformers_one_winding.loc[
                :,
                [
                    'is_phase_1_connected',
                    'is_phase_2_connected',
                    'is_phase_3_connected'
                ]
            ].sum().sum()
        )
        self.branch_dimension = line_dimension + transformer_dimension

        # Define load dimension, i.e., number of all loads, which
        # will be the second dimension of the load incidence matrix.
        self.load_dimension = (
            electric_grid_data.electric_grid_loads.shape[0]
        )

        # Create `nodes` data frame, i.e., collection of all phases of all nodes
        # for generating indexing functions for the admittance matrix.
        nodes = (
            pd.DataFrame(
                None,
                index=range(self.node_dimension),
                columns=[
                    'node_name',
                    'phase',
                    'node_type'
                ]
            )
        )
        # Fill `node_name`.
        nodes['node_name'] = (
            pd.concat([
                electric_grid_data.electric_grid_nodes.loc[
                    electric_grid_data.electric_grid_nodes['is_phase_1_connected'] == 1,
                    'node_name'
                ],
                electric_grid_data.electric_grid_nodes.loc[
                    electric_grid_data.electric_grid_nodes['is_phase_2_connected'] == 1,
                    'node_name'
                ],
                electric_grid_data.electric_grid_nodes.loc[
                    electric_grid_data.electric_grid_nodes['is_phase_3_connected'] == 1,
                    'node_name'
                ]
            ], ignore_index=True)
        )
        # Fill `phase`.
        nodes['phase'] = (
            np.concatenate([
                np.repeat('1', sum(electric_grid_data.electric_grid_nodes['is_phase_1_connected'] == 1)),
                np.repeat('2', sum(electric_grid_data.electric_grid_nodes['is_phase_2_connected'] == 1)),
                np.repeat('3', sum(electric_grid_data.electric_grid_nodes['is_phase_3_connected'] == 1))
            ])
        )
        # Fill `node_type`.
        nodes['node_type'] = 'no_source'
        # Set `node_type` for source node.
        nodes.loc[
            nodes['node_name'] == (electric_grid_data.electric_grids['source_node_name'][0]),
            'node_type'
        ] = 'source'
        # TODO: Sort nodes

        # Create `branches` data frame, i.e., collection of phases of all branches
        # for generating indexing functions for the branch admittance matrices.
        # - Transformers must have same number of phases per winding and exactly
        #   two windings.
        branches = (
            pd.DataFrame(
                None,
                index=range(self.branch_dimension),
                columns=[
                    'branch_name',
                    'phase',
                    'branch_type'
                ]
            )
        )
        # Fill `branch_name`.
        branches['branch_name'] = (
            pd.concat([
                electric_grid_data.electric_grid_lines.loc[
                    electric_grid_data.electric_grid_lines['is_phase_1_connected'] == 1,
                    'line_name'
                ],
                electric_grid_data.electric_grid_lines.loc[
                    electric_grid_data.electric_grid_lines['is_phase_2_connected'] == 1,
                    'line_name'
                ],
                electric_grid_data.electric_grid_lines.loc[
                    electric_grid_data.electric_grid_lines['is_phase_3_connected'] == 1,
                    'line_name'
                ],
                electric_grid_transformers_one_winding.loc[
                    electric_grid_transformers_one_winding['is_phase_1_connected'] == 1,
                    'transformer_name'
                ],
                electric_grid_transformers_one_winding.loc[
                    electric_grid_transformers_one_winding['is_phase_2_connected'] == 1,
                    'transformer_name'
                ],
                electric_grid_transformers_one_winding.loc[
                    electric_grid_transformers_one_winding['is_phase_3_connected'] == 1,
                    'transformer_name'
                ]
            ], ignore_index=True)
        )
        # Fill `phase`.
        branches['phase'] = (
            np.concatenate([
                np.repeat('1', sum(electric_grid_data.electric_grid_lines['is_phase_1_connected'] == 1)),
                np.repeat('2', sum(electric_grid_data.electric_grid_lines['is_phase_2_connected'] == 1)),
                np.repeat('3', sum(electric_grid_data.electric_grid_lines['is_phase_3_connected'] == 1)),
                np.repeat('1', sum(electric_grid_transformers_one_winding['is_phase_1_connected'] == 1)),
                np.repeat('2', sum(electric_grid_transformers_one_winding['is_phase_2_connected'] == 1)),
                np.repeat('3', sum(electric_grid_transformers_one_winding['is_phase_3_connected'] == 1))
            ])
        )
        # Fill `branch_type`.
        branches['branch_type'] = (
            np.concatenate([
                np.repeat('line', line_dimension),
                np.repeat('transformer', transformer_dimension)
            ])
        )
        # TODO: Sort branches

        # Define index vectors for various element types
        # for easier index definitions, e.g., in the optimization problem.
        self.phases = pd.Index(['1', '2', '3'])
        self.node_names = pd.Index(electric_grid_data.electric_grid_nodes['node_name'])
        self.node_types = pd.Index(['source', 'no_source'])
        self.nodes_phases = pd.Index(zip(nodes['node_name'], nodes['phase']))
        self.line_names = pd.Index(electric_grid_data.electric_grid_lines['line_name'])
        self.transformer_names = pd.Index(electric_grid_transformers_one_winding['transformer_name'])
        self.branch_names = pd.Index(branches['branch_name'])
        self.branch_types = pd.Index(['line', 'transformer'])
        self.branches_phases = pd.Index(zip(branches['branch_name'], branches['phase'], branches['branch_type']))
        self.load_names = pd.Index(electric_grid_data.electric_grid_loads['load_name'])

        # Generate indexing dictionaries for the nodal admittance matrix,
        # i.e., for all phases of all nodes.
        # - This is a workaround to avoid low-performance string search operations
        #   for each indexing access.
        # - Instead, the appropriate boolean index vectors are pre-generated here
        #   and stored into data frames.

        # Index by node name.
        self.node_by_node_name = dict.fromkeys(self.node_names)
        # Index by phase.
        self.node_by_phase = dict.fromkeys(self.phases)
        # Index by node type.
        self.node_by_node_type = dict.fromkeys(self.node_types)
        # Index by load name.
        self.node_by_load_name = dict.fromkeys(self.load_names)

        # Generate indexing dictionaries for the branch admittance matrices,
        # i.e., for all phases of all branches.
        # - This is a workaround to avoid low-performance string search operations
        #   for each indexing access.
        # - Instead, the appropriate boolean index vectors are pre-generated here
        #   and stored into data frames.

        # Index by line name.
        self.branch_by_line_name = dict.fromkeys(self.line_names)
        # Index by transformer name.
        self.branch_by_transformer_name = dict.fromkeys(self.transformer_names)
        # Index by phase.
        self.branch_by_phase = dict.fromkeys(self.phases)

        # Generate indexing dictionary for the load incidence matrix.

        # Index by load name.
        self.load_by_load_name = dict.fromkeys(self.load_names)
