"""Electric grid models module."""

from multimethod import multimethod
import numpy as np
import opendssdirect
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface
import fledge.utils

logger = fledge.config.get_logger(__name__)


class ElectricGridModel(object):
    """Electric grid model object."""

    der_power_vector_nominal: np.ndarray


class ElectricGridModelDefault(ElectricGridModel):
    """Electric grid model object consisting of the index sets for node names / branch names / der names / phases /
    node types / branch types, the nodal admittance / transformation matrices, branch admittance /
    incidence matrices, DER incidence matrices and no load voltage vector as well as nominal power vector.

    :syntax:
        - ``ElectricGridModelDefault(electric_grid_data)``: Instantiate electric grid model for given
          `electric_grid_data`.
        - ``ElectricGridModelDefault(scenario_name)``: Instantiate electric grid model for given `scenario_name`.
          The required `electric_grid_data` is obtained from the database.

    Arguments:
        scenario_name (str): FLEDGE scenario name.
        electric_grid_data (fledge.database_interface.ElectricGridData): Electric grid data object.

    Keyword Arguments:
        voltage_no_load_method (str): Choices: `by_definition`, `by_calculation`. Default: `by_definition`.
            Defines the construction method for the no load voltage vector.
            If `by_definition`, the nodal voltage definition in the database is taken.
            If `by_calculation`, the no load voltage is calculated from the source node voltage
            and the nodal admittance matrix.

    Attributes:
        phases (pd.Index): Index set of the phases.
        node_names (pd.Index): Index set of the node names.
        node_types (pd.Index): Index set of the node types.
        line_names (pd.Index): Index set of the line names.
        transformer_names (pd.Index): Index set of the transformer names.
        branch_names (pd.Index): Index set of the branch names, i.e., all line names and transformer names.
        branch_types (pd.Index): Index set of the branch types.
        der_names (pd.Index): Index set of the DER names.
        der_types (pd.Index): Index set of the DER types.
        branches (pd.Index): Multi-level / tuple index set of the branch types, branch names and phases
            corresponding to the dimension of the branch admittance matrices.
        nodes (pd.Index): Multi-level / tuple index set of the node types, node names and phases
            corresponding to the dimension of the node admittance matrices.
        ders (pd.Index): Index set of the DER names, corresponding to the dimension of the DER power vector.
        node_admittance_matrix (scipy.sparse.spmatrix): Nodal admittance matrix.
        node_transformation_matrix (scipy.sparse.spmatrix): Nodal transformation matrix.
        branch_admittance_1_matrix (scipy.sparse.spmatrix): Branch admittance matrix in the 'from' direction.
        branch_admittance_2_matrix (scipy.sparse.spmatrix): Branch admittance matrix in the 'to' direction.
        branch_incidence_1_matrix (scipy.sparse.spmatrix): Branch incidence matrix in the 'from' direction.
        branch_incidence_2_matrix (scipy.sparse.spmatrix): Branch incidence matrix in the 'to' direction.
        der_incidence_wye_matrix (scipy.sparse.spmatrix): Load incidence matrix for 'wye' DERs.
        der_incidence_delta_matrix (scipy.sparse.spmatrix): Load incidence matrix for 'delta' DERs.
        node_voltage_vector_no_load (np.ndarray): Nodal voltage at no load conditions.
        der_power_vector_nominal (np.ndarray): Load power vector at nominal power conditions.
    """

    phases: pd.Index
    node_names: pd.Index
    node_types: pd.Index
    line_names: pd.Index
    transformer_names: pd.Index
    branch_names: pd.Index
    branch_types: pd.Index
    der_names: pd.Index
    der_types: pd.Index
    nodes: pd.Index
    branches: pd.Index
    ders: pd.Index
    node_admittance_matrix: scipy.sparse.spmatrix
    node_transformation_matrix: scipy.sparse.spmatrix
    branch_admittance_1_matrix: scipy.sparse.spmatrix
    branch_admittance_2_matrix: scipy.sparse.spmatrix
    branch_incidence_1_matrix: scipy.sparse.spmatrix
    branch_incidence_2_matrix: scipy.sparse.spmatrix
    der_incidence_wye_matrix: scipy.sparse.spmatrix
    der_incidence_delta_matrix: scipy.sparse.spmatrix
    node_voltage_vector_no_load: np.ndarray

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            **kwargs
    ):

        # Obtain electric grid data.
        electric_grid_data = fledge.database_interface.ElectricGridData(scenario_name)

        # Instantiate electric grid model object.
        self.__init__(electric_grid_data, **kwargs)

    @multimethod
    def __init__(
            self,
            electric_grid_data: fledge.database_interface.ElectricGridData,
            voltage_no_load_method='by_definition'
    ):

        # Obtain transformer data for first winding for generating index sets.
        electric_grid_transformers_first_winding = (
            electric_grid_data.electric_grid_transformers.loc[
                electric_grid_data.electric_grid_transformers['winding'] == 1,
                :
            ]
        )

        # Obtain index sets for phases / node names / node types / line names / transformer names /
        # branch types / DER names.
        self.phases = pd.Index([1, 2, 3])
        self.node_names = pd.Index(electric_grid_data.electric_grid_nodes['node_name'])
        self.node_types = pd.Index(['source', 'no_source'])
        self.line_names = pd.Index(electric_grid_data.electric_grid_lines['line_name'])
        self.transformer_names = pd.Index(electric_grid_transformers_first_winding['transformer_name'])
        self.branch_types = pd.Index(['line', 'transformer'])
        self.der_names = pd.Index(electric_grid_data.electric_grid_ders['der_name'])
        self.der_types = pd.Index(electric_grid_data.electric_grid_ders['der_type'].unique())

        # Obtain nodes index set, i.e., collection of all phases of all nodes
        # for generating indexing functions for the admittance matrix.
        # - The admittance matrix has one entry for each phase of each node in
        #   both dimensions.
        # - There cannot be "empty" dimensions for missing phases of nodes,
        #   because the matrix would become singular.
        # - Therefore the admittance matrix must have the exact number of existing
        #   phases of all nodes.
        # - Nodes are sorted to match the order returned from OpenDSS
        #   to enable comparing results.
        node_dimension = (
            int(electric_grid_data.electric_grid_nodes.loc[
                :,
                [
                    'is_phase_1_connected',
                    'is_phase_2_connected',
                    'is_phase_3_connected'
                ]
            ].sum().sum())
        )
        self.nodes = (
            pd.DataFrame(
                None,
                index=range(node_dimension),
                columns=[
                    'node_type',
                    'node_name',
                    'phase'
                ]
            )
        )
        # Fill `node_name`.
        self.nodes['node_name'] = (
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
        self.nodes['phase'] = (
            np.concatenate([
                np.repeat(1, sum(electric_grid_data.electric_grid_nodes['is_phase_1_connected'] == 1)),
                np.repeat(2, sum(electric_grid_data.electric_grid_nodes['is_phase_2_connected'] == 1)),
                np.repeat(3, sum(electric_grid_data.electric_grid_nodes['is_phase_3_connected'] == 1))
            ])
        )
        # Fill `node_type`.
        self.nodes['node_type'] = 'no_source'
        # Set `node_type` for source node.
        self.nodes.loc[
            self.nodes['node_name'] == (electric_grid_data.electric_grid['source_node_name']),
            'node_type'
        ] = 'source'
        # Sort nodes to match order in `fledge.power_flow_solvers.get_voltage_opendss`.
        self.nodes.sort_values(['node_name', 'phase'], inplace=True)
        self.nodes = pd.MultiIndex.from_frame(self.nodes)

        # Obtain branches index set, i.e., collection of phases of all branches
        # for generating indexing functions for the branch admittance matrices.
        # - Branches consider all power delivery elements, i.e., lines as well as
        #   transformers.
        # - The second dimension of the branch admittance matrices is the number of
        #   phases of all nodes.
        # - Transformers must have same number of phases per winding and exactly
        #   two windings.
        # - TODO: Add switches.
        line_dimension = (
            int(electric_grid_data.electric_grid_lines.loc[
                :,
                [
                    'is_phase_1_connected',
                    'is_phase_2_connected',
                    'is_phase_3_connected'
                ]
            ].sum().sum())
        )
        transformer_dimension = (
            int(electric_grid_transformers_first_winding.loc[
                :,
                [
                    'is_phase_1_connected',
                    'is_phase_2_connected',
                    'is_phase_3_connected'
                ]
            ].sum().sum())
        )
        self.branches = (
            pd.DataFrame(
                None,
                index=range(line_dimension + transformer_dimension),
                columns=[
                    'branch_type',
                    'branch_name',
                    'phase'
                ]
            )
        )
        # Fill `branch_name`.
        self.branches['branch_name'] = (
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
                electric_grid_transformers_first_winding.loc[
                    electric_grid_transformers_first_winding['is_phase_1_connected'] == 1,
                    'transformer_name'
                ],
                electric_grid_transformers_first_winding.loc[
                    electric_grid_transformers_first_winding['is_phase_2_connected'] == 1,
                    'transformer_name'
                ],
                electric_grid_transformers_first_winding.loc[
                    electric_grid_transformers_first_winding['is_phase_3_connected'] == 1,
                    'transformer_name'
                ]
            ], ignore_index=True)
        )
        # Fill `phase`.
        self.branches['phase'] = (
            np.concatenate([
                np.repeat(1, sum(electric_grid_data.electric_grid_lines['is_phase_1_connected'] == 1)),
                np.repeat(2, sum(electric_grid_data.electric_grid_lines['is_phase_2_connected'] == 1)),
                np.repeat(3, sum(electric_grid_data.electric_grid_lines['is_phase_3_connected'] == 1)),
                np.repeat(1, sum(electric_grid_transformers_first_winding['is_phase_1_connected'] == 1)),
                np.repeat(2, sum(electric_grid_transformers_first_winding['is_phase_2_connected'] == 1)),
                np.repeat(3, sum(electric_grid_transformers_first_winding['is_phase_3_connected'] == 1))
            ])
        )
        # Fill `branch_type`.
        self.branches['branch_type'] = (
            np.concatenate([
                np.repeat('line', line_dimension),
                np.repeat('transformer', transformer_dimension)
            ])
        )
        self.branches.sort_values(['branch_type', 'branch_name', 'phase'], inplace=True)
        self.branches = pd.MultiIndex.from_frame(self.branches)

        # Obtain index set for DERs.
        self.ders = pd.MultiIndex.from_frame(electric_grid_data.electric_grid_ders[['der_type', 'der_name']])

        # Define sparse matrices for nodal admittance, nodal transformation,
        # branch admittance, branch incidence and der incidence matrix entries.
        self.node_admittance_matrix = (
             scipy.sparse.dok_matrix((len(self.nodes), len(self.nodes)), dtype=np.complex)
        )
        self.node_transformation_matrix = (
             scipy.sparse.dok_matrix((len(self.nodes), len(self.nodes)), dtype=np.int)
        )
        self.branch_admittance_1_matrix = (
             scipy.sparse.dok_matrix((len(self.branches), len(self.nodes)), dtype=np.complex)
        )
        self.branch_admittance_2_matrix = (
             scipy.sparse.dok_matrix((len(self.branches), len(self.nodes)), dtype=np.complex)
        )
        self.branch_incidence_1_matrix = (
             scipy.sparse.dok_matrix((len(self.branches), len(self.nodes)), dtype=np.int)
        )
        self.branch_incidence_2_matrix = (
             scipy.sparse.dok_matrix((len(self.branches), len(self.nodes)), dtype=np.int)
        )
        self.der_incidence_wye_matrix = (
             scipy.sparse.dok_matrix((len(self.nodes), len(self.ders)), dtype=np.float)
        )
        self.der_incidence_delta_matrix = (
             scipy.sparse.dok_matrix((len(self.nodes), len(self.ders)), dtype=np.float)
        )

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
                fledge.utils.get_index(
                    self.nodes,
                    node_name=line['node_1_name'],
                    phase=fledge.utils.get_element_phases_array(line)
                )
            )
            node_index_2 = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=line['node_2_name'],
                    phase=fledge.utils.get_element_phases_array(line)
                )
            )
            branch_index = (
                fledge.utils.get_index(
                    self.branches,
                    branch_type='line',
                    branch_name=line['line_name']
                )
            )

            # Add line element matrices to the nodal admittance matrix.
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_1)] += admittance_matrix_11
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_2)] += admittance_matrix_12
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_1)] += admittance_matrix_21
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_2)] += admittance_matrix_22

            # Add line element matrices to the branch admittance matrices.
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_11
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_12
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_21
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_22

            # Add line element matrices to the branch incidence matrices.
            self.branch_incidence_1_matrix[np.ix_(branch_index, node_index_1)] += (
                np.identity(len(branch_index), dtype=np.int)
            )
            self.branch_incidence_2_matrix[np.ix_(branch_index, node_index_2)] += (
                 np.identity(len(branch_index), dtype=np.int)
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
                raise ValueError

            # Obtain indexes for positioning the transformer element
            # matrices in the full matrices.
            node_index_1 = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=windings.at[0, 'node_name'],
                    phase=fledge.utils.get_element_phases_array(transformer)
                )
            )
            node_index_2 = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=windings.at[1, 'node_name'],
                    phase=fledge.utils.get_element_phases_array(transformer)
                )
            )
            branch_index = (
                fledge.utils.get_index(
                    self.branches,
                    branch_type='transformer',
                    branch_name=transformer['transformer_name']
                )
            )

            # Add transformer element matrices to the nodal admittance matrix.
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_1)] += admittance_matrix_11
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_2)] += admittance_matrix_12
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_1)] += admittance_matrix_21
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_2)] += admittance_matrix_22

            # Add transformer element matrices to the branch admittance matrices.
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_11
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_12
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_21
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_22

            # Add transformer element matrices to the branch incidence matrices.
            self.branch_incidence_1_matrix[np.ix_(branch_index, node_index_1)] += (
                np.identity(len(branch_index), dtype=np.int)
            )
            self.branch_incidence_2_matrix[np.ix_(branch_index, node_index_2)] += (
                np.identity(len(branch_index), dtype=np.int)
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
        for node_name, node in electric_grid_data.electric_grid_nodes.iterrows():
            # Obtain node phases index.
            phases_index = fledge.utils.get_element_phases_array(node) - 1

            # Construct node transformation matrix.
            transformation_matrix = transformation_entries[np.ix_(phases_index, phases_index)]

            # Obtain index for positioning node transformation matrix in full
            # transformation matrix.
            node_index = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=node['node_name']
                )
            )

            # Add node transformation matrix to full transformation matrix.
            self.node_transformation_matrix[np.ix_(node_index, node_index)] = transformation_matrix

        # Add ders to der incidence matrix.
        for der_name, der in electric_grid_data.electric_grid_ders.iterrows():
            # Obtain der connection type.
            connection = der['connection']

            # Obtain indexes for positioning the DER in the incidence matrix.
            node_index = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=der['node_name'],
                    phase=fledge.utils.get_element_phases_array(der)
                )
            )
            der_index = (
                fledge.utils.get_index(
                    self.ders,
                    der_name=der['der_name']
                )
            )

            if connection == "wye":
                # Define incidence matrix entries.
                # - Wye ders are represented as balanced ders across all
                #   their connected phases.
                incidence_matrix = (
                    np.ones((len(node_index), 1), dtype=np.float)
                    / len(node_index)
                )
                self.der_incidence_wye_matrix[np.ix_(node_index, der_index)] = incidence_matrix

            elif connection == "delta":
                # Obtain phases of the delta der.
                phases_list = fledge.utils.get_element_phases_array(der).tolist()

                # Select connection node based on phase arrangement of delta der.
                # - Delta ders must be single-phase.
                if phases_list in ([1, 2], [2, 3]):
                    node_index = [node_index[1]]
                elif phases_list == [1, 3]:
                    node_index = [node_index[2]]
                else:
                    logger.error(f"Unknown delta phase arrangement: {phases_list}")
                    raise ValueError

                # Define incidence matrix entry.
                # - Delta ders are assumed to be single-phase.
                incidence_matrix = np.array([1])
                self.der_incidence_wye_matrix[np.ix_(node_index, der_index)] = incidence_matrix

            else:
                logger.error(f"Unknown der connection type: {connection}")
                raise ValueError

        # Convert sparse matrices for nodal admittance, nodal transformation,
        # branch admittance, branch incidence and der incidence matrices.
        # - Converting from DOK to CSR format for more efficient calculations
        #   according to <https://docs.scipy.org/doc/scipy/reference/sparse.html>.
        self.node_admittance_matrix = self.node_admittance_matrix.tocsr()
        self.node_transformation_matrix = self.node_transformation_matrix.tocsr()
        self.branch_admittance_1_matrix = self.branch_admittance_1_matrix.tocsr()
        self.branch_admittance_2_matrix = self.branch_admittance_2_matrix.tocsr()
        self.branch_incidence_1_matrix = self.branch_incidence_1_matrix.tocsr()
        self.branch_incidence_2_matrix = self.branch_incidence_2_matrix.tocsr()
        self.der_incidence_wye_matrix = self.der_incidence_wye_matrix.tocsr()
        self.der_incidence_delta_matrix = self.der_incidence_delta_matrix.tocsr()

        # Construct no load voltage vector for the grid.
        # - The nodal no load voltage vector can be constructed by
        #   1) `voltage_no_load_method="by_definition"`, i.e., the nodal voltage
        #   definition in the database is taken, or by
        #   2) `voltage_no_load_method="by_calculation"`, i.e., the no load voltage is
        #   calculated from the source node voltage and the nodal admittance matrix.
        # - TODO: Check if no load voltage divide by sqrt(3) is correct.
        self.node_voltage_vector_no_load = (
            np.zeros((len(self.nodes), 1), dtype=np.complex)
        )
        # Define phase orientations.
        voltage_phase_factors = (
            np.array([
                np.exp(0 * 1j),  # Phase 1.
                np.exp(- 2 * np.pi / 3 * 1j),  # Phase 2.
                np.exp(2 * np.pi / 3 * 1j)  # Phase 3.
            ])
        )
        if voltage_no_load_method == 'by_definition':
            for node_name, node in electric_grid_data.electric_grid_nodes.iterrows():
                # Obtain node phases index.
                phases_index = fledge.utils.get_element_phases_array(node) - 1

                # Obtain node voltage level.
                voltage = node['voltage']

                # Obtain node index for positioning the node voltage in the voltage vector.
                node_index = (
                    fledge.utils.get_index(
                        self.nodes,
                        node_name=node['node_name']
                    )
                )

                # Insert voltage into voltage vector.
                self.node_voltage_vector_no_load[node_index] = (
                    np.transpose([
                        voltage
                        * voltage_phase_factors[phases_index]
                        / np.sqrt(3)
                    ])
                )
        elif voltage_no_load_method == 'by_calculation':
            # Obtain source node.
            node = (
                electric_grid_data.electric_grid_nodes.loc[
                    electric_grid_data.electric_grid['source_node_name'],
                    :
                ]
            )

            # Obtain source node phases index.
            phases_index = fledge.utils.get_element_phases_array(node) - 1

            # Obtain source node voltage level.
            voltage = node['voltage']

            # Obtain source / no source node indexes for positioning the node voltage in the voltage vector.
            node_source_index = (
                fledge.utils.get_index(
                    self.nodes,
                    node_type='source'
                )
            )
            node_no_source_index = (
                fledge.utils.get_index(
                    self.nodes,
                    node_type='no_source'
                )
            )

            # Insert source node voltage into voltage vector.
            self.node_voltage_vector_no_load[node_source_index] = (
                np.transpose([
                    voltage
                    * voltage_phase_factors[phases_index]
                    / np.sqrt(3)
                ])
            )

            # Calculate all remaining no load node voltages.
            # TODO: Debug no load voltage calculation.
            self.node_voltage_vector_no_load[node_no_source_index] = (
                np.transpose([
                    scipy.sparse.linalg.spsolve(
                        - self.node_admittance_matrix[np.ix_(node_no_source_index, node_no_source_index)],
                        (
                            self.node_admittance_matrix[np.ix_(node_no_source_index, node_source_index)]
                            @ self.node_voltage_vector_no_load[node_source_index]
                        )
                    )
                ])
            )

        # Construct nominal DER power vector.
        self.der_power_vector_nominal = (
            # TODO: Remove load multiplier.
            electric_grid_data.electric_grid['load_multiplier']
            * (
                electric_grid_data.electric_grid_ders.loc[:, 'active_power']
                + 1j * electric_grid_data.electric_grid_ders.loc[:, 'reactive_power']
            ).values
        )


class ElectricGridModelOpenDSS(ElectricGridModel):
    """OpenDSS electric grid model object.

    - Instantiate OpenDSS circuit by running generating OpenDSS commands corresponding to given `electric_grid_data`,
      utilizing the `OpenDSSDirect.py` package.
    - The OpenDSS circuit can be accessed with the API of
      `OpenDSSDirect.py`: http://dss-extensions.org/OpenDSSDirect.py/opendssdirect.html
    - Due to dependency on `OpenDSSDirect.py`, creating multiple objects of this type may result in erroneous behavior.

    :syntax:
        - ``ElectricGridModelOpenDSS(electric_grid_data)``: Initialize OpenDSS circuit model for given
          `electric_grid_data`.
        - ``ElectricGridModelOpenDSS(scenario_name)`` Initialize OpenDSS circuit model for given `scenario_name`.
          The required `electric_grid_data` is obtained from the database.

    Parameters:
        scenario_name (str): FLEDGE scenario name.
        electric_grid_data (fledge.database_interface.ElectricGridData): Electric grid data object.
    """

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain electric grid data.
        electric_grid_data = (
            fledge.database_interface.ElectricGridData(scenario_name)
        )

        self.__init__(electric_grid_data)

    @multimethod
    def __init__(
            self,
            electric_grid_data: fledge.database_interface.ElectricGridData
    ):

        # Construct nominal DER power vector.
        self.der_power_vector_nominal = (
            # TODO: Remove load multiplier.
            electric_grid_data.electric_grid['load_multiplier']
            * (
                electric_grid_data.electric_grid_ders.loc[:, 'active_power']
                + 1j * electric_grid_data.electric_grid_ders.loc[:, 'reactive_power']
            ).values
        )

        # Clear OpenDSS.
        opendss_command_string = "clear"
        logger.debug(f"opendss_command_string = {opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Obtain extra definitions string.
        if pd.isnull(electric_grid_data.electric_grid['extra_definitions_string']):
            extra_definitions_string = ""
        else:
            extra_definitions_string = (
                electric_grid_data.electric_grid['extra_definitions_string']
            )

        # Add circuit info to OpenDSS command string.
        opendss_command_string = (
            f"new circuit.{electric_grid_data.electric_grid['electric_grid_name']}"
            + f" phases={electric_grid_data.electric_grid['n_phases']}"
            + f" bus1={electric_grid_data.electric_grid['source_node_name']}"
            + f" basekv={electric_grid_data.electric_grid['source_voltage'] / 1e3}"
            + f" {extra_definitions_string}"
        )

        # Create circuit in OpenDSS.
        logger.debug(f"opendss_command_string = {opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Define line codes.
        for line_type_index, line_type in electric_grid_data.electric_grid_line_types.iterrows():
            # Obtain line resistance and reactance matrix entries for the line.
            matrices = (
                electric_grid_data.electric_grid_line_types_matrices.loc[
                    electric_grid_data.electric_grid_line_types_matrices['line_type'] == line_type['line_type'],
                    ['r', 'x', 'c']
                ]
            )

            # Add line type name and number of phases to OpenDSS command string.
            opendss_command_string = (
                f"new linecode.{line_type['line_type']}"
                + f" nphases={line_type['n_phases']}"
            )

            # Add resistance and reactance matrix entries to OpenDSS command string,
            # with formatting depending on number of phases.
            if line_type['n_phases'] == 1:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f}]".format(*matrices['r'])
                    + " xmatrix = "
                    + "[{:.8f}]".format(*matrices['x'])
                    + " cmatrix = "
                    + "[{:.8f}]".format(*matrices['c'])
                )
            elif line_type['n_phases'] == 2:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices['r'])
                    + " xmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices['x'])
                    + " cmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices['c'])
                )
            elif line_type['n_phases'] == 3:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices['r'])
                    + f" xmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices['x'])
                    + f" cmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices['c'])
                )

            # Create line code in OpenDSS.
            logger.debug(f"opendss_command_string = {opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define lines.
        for line_index, line in electric_grid_data.electric_grid_lines.iterrows():
            # Obtain number of phases for the line.
            n_phases = len(fledge.utils.get_element_phases_array(line))

            # Add line name, phases, node connections, line type and length
            # to OpenDSS command string.
            opendss_command_string = (
                f"new line.{line['line_name']}"
                + f" phases={line['n_phases']}"
                + f" bus1={line['node_1_name']}{fledge.utils.get_element_phases_string(line)}"
                + f" bus2={line['node_2_name']}{fledge.utils.get_element_phases_string(line)}"
                + f" linecode={line['line_type']}"
                + f" length={line['length']}"
            )

            # Create line in OpenDSS.
            logger.debug(f"opendss_command_string = {opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define transformers.
        # - Note: This setup only works for transformers with
        #   identical number of phases at each winding / side.
        for transformer_index, transformer in (
                electric_grid_data.electric_grid_transformers.loc[
                    electric_grid_data.electric_grid_transformers['winding'] == 1,
                    :
                ].iterrows()
        ):
            # Obtain number of phases for the transformer.
            # This assumes identical number of phases at all windings.
            n_phases = len(fledge.utils.get_element_phases_array(transformer))

            # Obtain windings for the transformer.
            windings = (
                electric_grid_data.electric_grid_transformers.loc[
                    (
                        electric_grid_data.electric_grid_transformers['transformer_name']
                        == transformer['transformer_name']
                    ),
                    :
                ]
            )

            # Obtain reactances for the transformer.
            reactances = (
                electric_grid_data.electric_grid_transformer_reactances.loc[
                    (
                        electric_grid_data.electric_grid_transformer_reactances['transformer_name']
                        == transformer['transformer_name']
                    ),
                    :
                ]
            )

            # Obtain taps for the transformer.
            taps = (
                electric_grid_data.electric_grid_transformer_taps.loc[
                    (
                        electric_grid_data.electric_grid_transformer_taps['transformer_name']
                        == transformer['transformer_name']
                    ),
                    :
                ]
            )

            # Add transformer name, number of phases / windings and reactances
            # to OpenDSS command string.
            opendss_command_string = (
                f"new transformer.{transformer['transformer_name']}"
                + f" phases={n_phases}"
                + f" windings={len(windings['winding'])}"
                + f" xscarray={[x for x in reactances['reactance_percentage']]}"
            )
            for winding_index, winding in windings.iterrows():
                # Obtain nominal voltage level for each winding.
                voltage = electric_grid_data.electric_grid_nodes.at[winding['node_name'], 'voltage']

                # Obtain node phases connection string for each winding.
                if winding['connection'] == "wye":
                    if winding['is_phase_0_connected'] == 0:
                        # Enforce wye-open connection according to:
                        # OpenDSS Manual April 2018, page 136, "rneut".
                        node_phases_string = (
                            fledge.utils.get_element_phases_string(winding)
                            + ".4"
                        )
                    elif winding['is_phase_0_connected'] == 1:
                        # Enforce wye-grounded connection.
                        node_phases_string = (
                            fledge.utils.get_element_phases_string(winding)
                            + ".0"
                        )
                        # Remove leading ".0".
                        node_phases_string = node_phases_string[2:]
                elif winding['connection'] == "delta":
                    if winding['is_phase_0_connected'] == 0:
                        node_phases_string = (
                            fledge.utils.get_element_phases_string(winding)
                        )
                    elif winding['is_phase_0_connected'] == 1:
                        node_phases_string = (
                            fledge.utils.get_element_phases_string(winding)
                        )
                        # Remove leading ".0"
                        node_phases_string = node_phases_string[2:]
                        logger.warn(
                            "No ground connection possible for delta-connected"
                            + f" transformer {transformer['transformer_name']}."
                        )
                else:
                    logger.error(f"Unknown transformer connection type: {winding['connection']}")
                    raise ValueError

                # Add node connection, nominal voltage / power and resistance
                # to OpenDSS command string for each winding.
                opendss_command_string += (
                    f" wdg={winding['winding']}"
                    + f" bus={winding['node_name']}" + node_phases_string
                    + f" conn={winding['connection']}"
                    + f" kv={voltage / 1000}"
                    + f" kva={winding['power'] / 1000}"
                    + f" %r={winding['resistance_percentage']}"
                )

                # Add maximum / minimum level
                # to OpenDSS command string for each winding.
                for winding_index in np.flatnonzero(taps['winding'] == winding['winding']):
                    opendss_command_string += (
                        " maxtap="
                        + f"{taps.at[winding_index, 'tap_maximum_voltage_per_unit']}"
                        + f" mintap="
                        + f"{taps.at[winding_index, 'tap_minimum_voltage_per_unit']}"
                    )

            # Create transformer in OpenDSS.
            logger.debug(f"opendss_command_string = {opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define ders.
        # TODO: At the moment, all ders are modelled as loads in OpenDSS.
        for der_index, der in electric_grid_data.electric_grid_ders.iterrows():
            # Obtain number of phases for the der.
            n_phases = len(fledge.utils.get_element_phases_array(der))

            # Obtain nominal voltage level for the der.
            voltage = electric_grid_data.electric_grid_nodes.at[der['node_name'], 'voltage']
            # Convert to line-to-neutral voltage for single-phase ders, according to:
            # https://sourceforge.net/p/electricdss/discussion/861976/thread/9c9e0efb/
            if n_phases == 1:
                voltage /= np.sqrt(3)

            # Add node connection, model type, voltage, nominal power
            # to OpenDSS command string.
            opendss_command_string = (
                f"new load.{der['der_name']}"
                # TODO: Check if any difference without ".0" for wye-connected ders.
                + f" bus1={der['node_name']}{fledge.utils.get_element_phases_string(der)}"
                + f" phases={n_phases}"
                + f" conn={der['connection']}"
                # All loads are modelled as constant P/Q according to:
                # OpenDSS Manual April 2018, page 150, "Model"
                + f" model=1"
                + f" kv={voltage / 1000}"
                + f" kw={- der['active_power'] / 1000}"
                + f" kvar={- der['reactive_power'] / 1000}"
                # Set low V_min to avoid switching to impedance model according to:
                # OpenDSS Manual April 2018, page 150, "Vminpu"
                + f" vminpu=0.6"
                # Set high V_max to avoid switching to impedance model according to:
                # OpenDSS Manual April 2018, page 150, "Vmaxpu"
                + f" vmaxpu=1.4"
            )

            # Create der in OpenDSS.
            logger.debug(f"opendss_command_string = {opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # TODO: Add switches.

        # Set control mode and voltage bases.
        opendss_command_string = (
            "set voltagebases="
            + f"{electric_grid_data.electric_grid['voltage_bases_string']}"
            + "\nset controlmode="
            + f"{electric_grid_data.electric_grid['control_mode_string']}"
            + "\nset loadmult="
            + f"{electric_grid_data.electric_grid['load_multiplier']}"
            + "\ncalcvoltagebases"
        )
        logger.debug(f"opendss_command_string = {opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Set solution mode to "single snapshot power flow" according to:
        # OpenDSSComDoc, November 2016, page 1
        opendss_command_string = "set mode=0"
        logger.debug(f"opendss_command_string = {opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)
