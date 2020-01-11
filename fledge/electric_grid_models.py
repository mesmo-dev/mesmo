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
            electric_grid_data: fledge.database_interface.ElectricGridData = None,
            scenario_name: str = None
    ):
        """Instantiate electric grid index object for given `electric_grid_data` or `scenario_name`."""

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
        for node_name in self.node_names:
            self.node_by_node_name[node_name] = np.nonzero(nodes['node_name'] == node_name)[0].tolist()
        # Index by phase.
        self.node_by_phase = dict.fromkeys(self.phases)
        for phase in self.phases:
            self.node_by_phase[phase] = np.nonzero(nodes['phase'] == phase)[0].tolist()
        # Index by node type.
        self.node_by_node_type = dict.fromkeys(self.node_types)
        for node_type in self.node_types:
            self.node_by_node_type[node_type] = np.nonzero(nodes['node_type'] == node_type)[0].tolist()
        # Index by load name.
        self.node_by_load_name = dict.fromkeys(self.load_names)
        # TODO: Find a more efficient way to represent load / node / phase comparison.
        for load_name in self.load_names:
            load_phases = (
                np.where(
                    [
                        electric_grid_data.electric_grid_loads.at[load_name, 'is_phase_1_connected'] == 1,
                        electric_grid_data.electric_grid_loads.at[load_name, 'is_phase_2_connected'] == 1,
                        electric_grid_data.electric_grid_loads.at[load_name, 'is_phase_3_connected'] == 1
                    ],
                    ['1', '2', '3'],
                    [None, None, None]
                )
            )
            self.node_by_load_name[load_name] = (
                np.nonzero(
                    (nodes['node_name'] == electric_grid_data.electric_grid_loads.at[load_name, 'node_name'])
                    & (nodes['phase'].isin(load_phases))
                )[0].tolist()
            )

        # Generate indexing dictionaries for the branch admittance matrices,
        # i.e., for all phases of all branches.
        # - This is a workaround to avoid low-performance string search operations
        #   for each indexing access.
        # - Instead, the appropriate boolean index vectors are pre-generated here
        #   and stored into data frames.

        # Index by line name.
        self.branch_by_line_name = dict.fromkeys(self.line_names)
        for line_name in self.line_names:
            self.branch_by_line_name[line_name] = (
                np.nonzero(
                    (branches['branch_name'] == line_name)
                    & (branches['branch_type'] == 'line')
                )[0].tolist()
            )
        # Index by transformer name.
        self.branch_by_transformer_name = dict.fromkeys(self.transformer_names)
        for transformer_name in self.transformer_names:
            self.branch_by_transformer_name[transformer_name] = (
                np.nonzero(
                    (branches['branch_name'] == transformer_name)
                    & (branches['branch_type'] == 'transformer')
                )[0].tolist()
            )
        # Index by phase.
        self.branch_by_phase = dict.fromkeys(self.phases)
        for phase in self.phases:
            self.branch_by_phase[phase] = np.nonzero(branches['phase'] == phase)[0].tolist()

        # Generate indexing dictionary for the load incidence matrix.

        # Index by load name.
        self.load_by_load_name = (
            dict(zip(self.load_names, np.arange(len(self.load_names))[np.newaxis].transpose().tolist()))
        )


class ElectricGridModel(object):
    """Electric grid model object."""

    electric_grid_data: fledge.database_interface.ElectricGridData
    index: ElectricGridIndex
    node_admittance_matrix: scipy.sparse.dok_matrix
    node_transformation_matrix: scipy.sparse.dok_matrix
    branch_admittance_1_matrix: scipy.sparse.dok_matrix
    branch_admittance_2_matrix: scipy.sparse.dok_matrix
    branch_incidence_1_matrix: scipy.sparse.dok_matrix
    branch_incidence_2_matrix: scipy.sparse.dok_matrix
    load_incidence_wye_matrix: scipy.sparse.dok_matrix
    load_incidence_delta_matrix: scipy.sparse.dok_matrix
    node_voltage_vector_no_load: np.ndarray
    load_power_vector_nominal: np.ndarray

    def __init__(
            self,
            electric_grid_data: fledge.database_interface.ElectricGridData = None,
            scenario_name: str = None
    ):
        """Instantiate electric grid model object for given `electric_grid_data` or `scenario_name`.

        - The nodal no-load voltage vector can be constructed by
          1) `voltage_no_load_method="by_definition"`, i.e., the nodal voltage
          definition in the database is taken, or by
          2) `voltage_no_load_method="by_calculation"`, i.e., the no-load voltage is
          calculated from the source node voltage and the nodal admittance matrix.
        """

        # Load electric grid data, if none.
        if electric_grid_data is None:
            electric_grid_data = fledge.database_interface.ElectricGridData(scenario_name)

        # Obtain electric grid index.
        self.index = ElectricGridIndex(electric_grid_data)

        # Define sparse matrices for nodal admittance, nodal transformation,
        # branch admittance, branch incidence and load matrix matrix entries.
        self.node_admittance_matrix = (
             scipy.sparse.dok_matrix((self.index.node_dimension, self.index.node_dimension), dtype=np.complex)
        )
        self.node_transformation_matrix = (
             scipy.sparse.dok_matrix((self.index.node_dimension, self.index.node_dimension), dtype=np.int)
        )
        self.branch_admittance_1_matrix = (
             scipy.sparse.dok_matrix((self.index.branch_dimension, self.index.node_dimension), dtype=np.complex)
        )
        self.branch_admittance_2_matrix = (
             scipy.sparse.dok_matrix((self.index.branch_dimension, self.index.node_dimension), dtype=np.complex)
        )
        self.branch_incidence_1_matrix = (
             scipy.sparse.dok_matrix((self.index.branch_dimension, self.index.node_dimension), dtype=np.int)
        )
        self.branch_incidence_2_matrix = (
             scipy.sparse.dok_matrix((self.index.branch_dimension, self.index.node_dimension), dtype=np.int)
        )
        self.load_incidence_wye_matrix = (
             scipy.sparse.dok_matrix((self.index.node_dimension, self.index.load_dimension), dtype=np.float)
        )
        self.load_incidence_delta_matrix = (
             scipy.sparse.dok_matrix((self.index.node_dimension, self.index.load_dimension), dtype=np.float)
        )

        # Define utility function to insert sub matrix into a sparse matrix at given row/column indexes.
        # - Ensures that values are added element-by-element to support sparse matrices.
        def insert_sub_matrix(matrix, sub_matrix, row_indexes, col_indexes):
            for row, sub_matrix_row in zip(row_indexes, sub_matrix.tolist()):
                for col, value in zip(col_indexes, sub_matrix_row):
                    matrix[row, col] += value  # In-place operator `+=` and empty return value for better performance.
            return

        # Add lines to admittance, transformation and incidence matrices.
        for line_index, line in electric_grid_data.electric_grid_lines.iterrows():
            # Obtain line resistance and reactance matrix entries for the line.
            rxc_matrix_entries_index = (
                electric_grid_data.electric_grid_line_types_matrices.loc[:, 'line_type'] == line['line_type']
            )
            r_matrix_entries = (
                electric_grid_data.electric_grid_line_types_matrices.loc[rxc_matrix_entries_index, 'r'].values
            )
            x_matrix_entries = (
                electric_grid_data.electric_grid_line_types_matrices.loc[rxc_matrix_entries_index, 'x'].values
            )
            c_matrix_entries = (
                electric_grid_data.electric_grid_line_types_matrices.loc[rxc_matrix_entries_index, 'c'].values
            )

            # Obtain the full line resistance and reactance matrices.
            # Data only contains upper half entries.
            rxc_matrix_full_index = (
                np.array([
                    [1, 2, 4],
                    [2, 3, 5],
                    [4, 5, 6]
                ]) - 1
            )
            # TODO: Remove usage of n_phases.
            rxc_matrix_full_index = (
                rxc_matrix_full_index[:line['n_phases'], :line['n_phases']]
            )
            r_matrix = r_matrix_entries[rxc_matrix_full_index]
            x_matrix = x_matrix_entries[rxc_matrix_full_index]
            c_matrix = c_matrix_entries[rxc_matrix_full_index]

            # Construct line series admittance matrix.
            series_admittance_matrix = (
                np.linalg.inv(
                    (r_matrix + 1j * x_matrix)
                    * line['length']
                )
            )

            # Construct line shunt admittance.
            # Note: nF to Ω with X = 1 / (2π * f * C)
            # TODO: Check line shunt admittance.
            base_frequency = 60.0  # TODO: Define base frequency in the database
            shunt_admittance_matrix = (
                c_matrix
                * 2 * np.pi * base_frequency * 1e-9
                * 0.5j
                * line['length']
            )

            # Construct line element admittance matrices according to:
            # https://doi.org/10.1109/TPWRS.2017.2728618
            admittance_matrix_11 = (
                series_admittance_matrix
                + shunt_admittance_matrix
            )
            admittance_matrix_12 = (
                - series_admittance_matrix
            )
            admittance_matrix_21 = (
                - series_admittance_matrix
            )
            admittance_matrix_22 = (
                series_admittance_matrix
                + shunt_admittance_matrix
            )

            # Obtain indexes for positioning the line element matrices
            # in the full admittance matrices.
            node_index_1 = (
                self.index.node_by_node_name[line['node_1_name']]
            )
            node_index_2 = (
                self.index.node_by_node_name[line['node_2_name']]
            )
            branch_index = (
                self.index.branch_by_line_name[line['line_name']]
            )

            # Add line element matrices to the nodal admittance matrix.
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_11,
                node_index_1,
                node_index_1
            )
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_12,
                node_index_1,
                node_index_2
            )
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_21,
                node_index_2,
                node_index_1
            )
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_22,
                node_index_2,
                node_index_2
            )

            # Add line element matrices to the branch admittance matrices.
            insert_sub_matrix(
                self.branch_admittance_1_matrix,
                admittance_matrix_11,
                branch_index,
                node_index_1
            )
            insert_sub_matrix(
                self.branch_admittance_1_matrix,
                admittance_matrix_12,
                branch_index,
                node_index_2
            )
            insert_sub_matrix(
                self.branch_admittance_2_matrix,
                admittance_matrix_21,
                branch_index,
                node_index_1
            )
            insert_sub_matrix(
                self.branch_admittance_2_matrix,
                admittance_matrix_22,
                branch_index,
                node_index_2
            )

            # Add line element matrices to the branch incidence matrices.
            insert_sub_matrix(
                self.branch_incidence_1_matrix,
                np.identity(len(branch_index), dtype=np.int),
                branch_index,
                node_index_1
            )
            insert_sub_matrix(
                self.branch_incidence_2_matrix,
                np.identity(len(branch_index), dtype=np.int),
                branch_index,
                node_index_2
            )

        # Add transformers to admittance, transformation and incidence matrices.
        # - Note: This setup only works for transformers with exactly two windings
        #   and identical number of phases at each winding / side.
    
        # Define transformer factor matrices according to:
        # https://doi.org/10.1109/TPWRS.2017.2728618
        transformer_factors_1 = (
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        )
        transformer_factors_2 = (
            1 / 3
            * np.array([
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 2]
            ])
        )
        transformer_factors_3 = (
            1 / np.sqrt(3)
            * np.array([
                [-1, 1, 0],
                [0, -1, 1],
                [1, 0, -1]
            ])
        )

        # Add transformers to admittance matrix.
        for transformer_index, transformer in (
            electric_grid_data.electric_grid_transformers.loc[
                electric_grid_data.electric_grid_transformers['winding'] == 1,
                :
            ].iterrows()
        ):
            # Obtain transformer windings.
            windings = (
                electric_grid_data.electric_grid_transformers.loc[
                    (
                        electric_grid_data.electric_grid_transformers['transformer_name']
                        == transformer['transformer_name']
                    ),
                    :
                ].reset_index()
            )

            # Obtain primary and secondary voltage.
            voltage_1 = (
                electric_grid_data.electric_grid_nodes.loc[
                    (
                        electric_grid_data.electric_grid_nodes['node_name']
                        == windings.at[0, 'node_name']
                    ),
                    'voltage'
                ]
            ).values[0]
            voltage_2 = (
                electric_grid_data.electric_grid_nodes.loc[
                    (
                        electric_grid_data.electric_grid_nodes['node_name']
                        == windings.at[1, 'node_name']
                    ),
                    'voltage'
                ]
            ).values[0]
    
            # Obtain transformer type.
            type = (
                windings.at[0, 'connection']
                + "-"
                + windings.at[1, 'connection']
            )
    
            # Obtain transformer resistance and reactance.
            resistance_percentage = transformer['resistance_percentage']
            reactance_percentage = (
                electric_grid_data.electric_grid_transformer_reactances.loc[
                    (
                        electric_grid_data.electric_grid_transformer_reactances['transformer_name']
                        == transformer['transformer_name']
                    ),
                    'reactance_percentage'
                ]
            ).values[0]
    
            # Calculate transformer admittance.
            admittance = (
                (
                    (
                        2 * resistance_percentage / 100
                        + 1j * reactance_percentage / 100
                    )
                    * (
                        voltage_2 ** 2
                        / windings.at[0, 'power']
                    )
                ) ** -1
            )
    
            # Calculate turn ratio.
            turn_ratio = (
                (
                    1.0  # TODO: Replace `1.0` with actual tap position.
                    * voltage_1
                )
                / (
                    1.0  # TODO: Replace `1.0` with actual tap position.
                    * voltage_2
                )
            )
    
            # Construct transformer element admittance matrices according to:
            # https://doi.org/10.1109/TPWRS.2017.2728618
            # - TODO: Add warning if wye-transformer is not grounded
            if type == "wye-wye":
                admittance_matrix_11 = (
                    admittance
                    * transformer_factors_1
                    / turn_ratio ** 2
                )
                admittance_matrix_12 = (
                    - 1 * admittance
                    * transformer_factors_1
                    / turn_ratio
                )
                admittance_matrix_21 = (
                    - 1 * admittance
                    * transformer_factors_1
                    / turn_ratio
                )
                admittance_matrix_22 = (
                    admittance
                    * transformer_factors_1
                )
            elif type == "delta-wye":
                admittance_matrix_11 = (
                    admittance
                    * transformer_factors_2
                    / turn_ratio ** 2
                )
                admittance_matrix_12 = (
                    - 1 * admittance
                    * - 1 * np.transpose(transformer_factors_3)
                    / turn_ratio
                )
                admittance_matrix_21 = (
                    - 1 * admittance
                    * - 1 * transformer_factors_3
                    / turn_ratio
                )
                admittance_matrix_22 = (
                    admittance
                    * transformer_factors_1
                )
            elif type == "wye-delta":
                admittance_matrix_11 = (
                    admittance
                    * transformer_factors_1
                    / turn_ratio ** 2
                )
                admittance_matrix_12 = (
                    - 1 * admittance
                    * - 1 * transformer_factors_3
                    / turn_ratio
                )
                admittance_matrix_21 = (
                    - 1 * admittance
                    * - 1 * np.transpose(transformer_factors_3)
                    / turn_ratio
                )
                admittance_matrix_22 = (
                    admittance
                    * transformer_factors_2
                )
            elif type == "delta-delta":
                admittance_matrix_11 = (
                    admittance
                    * transformer_factors_2
                    / turn_ratio ** 2
                )
                admittance_matrix_12 = (
                    - 1 * admittance
                    * transformer_factors_2
                    / turn_ratio
                )
                admittance_matrix_21 = (
                    - 1 * admittance
                    * transformer_factors_2
                    / turn_ratio
                )
                admittance_matrix_22 = (
                    admittance
                    * transformer_factors_2
                )
            else:
                logger.error(f"Unknown transformer type: {type}")
    
            # Obtain indexes for positioning the transformer element
            # matrices in the full matrices.
            node_index_1 = (
                self.index.node_by_node_name[windings.at[0, 'node_name']]
            )
            node_index_2 = (
                self.index.node_by_node_name[windings.at[1, 'node_name']]
            )
            branch_index = (
                self.index.branch_by_transformer_name[transformer['transformer_name']]
            )
    
            # Add transformer element matrices to the nodal admittance matrix.
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_11,
                node_index_1,
                node_index_1
            )
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_12,
                node_index_1,
                node_index_2
            )
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_21,
                node_index_2,
                node_index_1
            )
            insert_sub_matrix(
                self.node_admittance_matrix,
                admittance_matrix_22,
                node_index_2,
                node_index_2
            )
    
            # Add transformer element matrices to the branch admittance matrices.
            insert_sub_matrix(
                self.branch_admittance_1_matrix,
                admittance_matrix_11,
                branch_index,
                node_index_1
            )
            insert_sub_matrix(
                self.branch_admittance_1_matrix,
                admittance_matrix_12,
                branch_index,
                node_index_2
            )
            insert_sub_matrix(
                self.branch_admittance_2_matrix,
                admittance_matrix_21,
                branch_index,
                node_index_1
            )
            insert_sub_matrix(
                self.branch_admittance_2_matrix,
                admittance_matrix_22,
                branch_index,
                node_index_2
            )
    
            # Add transformer element matrices to the branch incidence matrices.
            insert_sub_matrix(
                self.branch_incidence_1_matrix,
                np.identity(len(branch_index), dtype=np.int),
                branch_index,
                node_index_1
            )
            insert_sub_matrix(
                self.branch_incidence_2_matrix,
                np.identity(len(branch_index), dtype=np.int),
                branch_index,
                node_index_2
            )

        # Define transformation matrix according to:
        # https://doi.org/10.1109/TPWRS.2018.2823277
        transformation_entries = (
            np.array([
                [1, -1, 0],
                [0, 1, -1],
                [-1, 0, 1]
            ])
        )
        for node_index, node in electric_grid_data.electric_grid_nodes.iterrows():
            # Obtain node transformation matrix index.
            transformation_index = (
                np.nonzero(
                    [
                        node['is_phase_1_connected'],
                        node['is_phase_2_connected'],
                        node['is_phase_3_connected']
                    ]
                    == 1
                )[0].tolist()
            )

            # Construct node transformation matrix.
            transformation_matrix = (
                transformation_entries[transformation_index, transformation_index]
            )

            # Obtain index for positioning node transformation matrix in full
            # transformation matrix.
            node_index = (
                self.index.node_by_node_name[node['node_name']]
            )

            # Add node transformation matrix to full transformation matrix.
            insert_sub_matrix(
                self.node_transformation_matrix,
                transformation_matrix,
                node_index,
                node_index
            )

        # Add loads to load incidence matrix.
        for load_index, load in electric_grid_data.electric_grid_loads.iterrows():
            # Obtain load connection type.
            connection = load['connection']

            # Obtain indexes for positioning load in incidence matrix.
            node_index = self.index.node_by_load_name[load['load_name']]
            load_index = self.index.load_by_load_name[load['load_name']]

            if connection == "wye":
                # Define incidence matrix entries.
                # - Wye loads are represented as balanced loads across all
                #   their connected phases.
                incidence_matrix = (
                    - np.ones((len(node_index), 1), dtype=np.float)
                    / len(node_index)
                )
                insert_sub_matrix(
                    self.load_incidence_wye_matrix,
                    incidence_matrix,
                    node_index,
                    load_index
                )
            elif connection == "delta":
                # Obtain phases of the delta load.
                phases = (
                    np.nonzero([
                        load['is_phase_1_connected'] == 1,
                        load['is_phase_2_connected'] == 1,
                        load['is_phase_3_connected'] == 1
                    ])[0].tolist()
                )

                # Select connection node based on phase arrangement of delta load.
                # - Delta loads must be single-phase.
                if phases in ([1, 2], [2, 3]):
                    node_index = [node_index[1]]
                elif phases == [1, 3]:
                    node_index = [node_index[2]]
                else:
                    logger.error(f"Unknown delta load phase arrangement: {phases}")

                # Define incidence matrix entry.
                # - Delta loads are assumed to be single-phase.
                incidence_matrix = np.array([- 1])
                insert_sub_matrix(
                    self.load_incidence_delta_matrix,
                    incidence_matrix,
                    node_index,
                    load_index
                )
            else:
                logger.error(f"Unknown load connection type: {connection}")
