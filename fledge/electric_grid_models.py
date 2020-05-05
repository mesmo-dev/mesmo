"""Electric grid models module."""

from multimethod import multimethod
import natsort
import numpy as np
import opendssdirect
import pandas as pd
import pyomo.environ as pyo
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface
import fledge.utils

logger = fledge.config.get_logger(__name__)


class ElectricGridModel(object):
    """Electric grid model object.

    Note:
        This abstract class only defines the expected variables of linear electric grid model objects,
        but does not implement any functionality.

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
    der_power_vector_nominal: np.ndarray

    def __init__(
            self,
            electric_grid_data: fledge.database_interface.ElectricGridData
    ):

        # Obtain index sets for phases / node names / node types / line names / transformer names /
        # branch types / DER names.
        self.phases = (
            pd.Index(
                np.unique(np.concatenate(
                    electric_grid_data.electric_grid_nodes.apply(
                        fledge.utils.get_element_phases_array,
                        axis=1
                    ).values
                ))
            )
        )
        self.node_names = pd.Index(electric_grid_data.electric_grid_nodes['node_name'])
        self.node_types = pd.Index(['source', 'no_source'])
        self.line_names = pd.Index(electric_grid_data.electric_grid_lines['line_name'])
        self.transformer_names = pd.Index(electric_grid_data.electric_grid_transformers['transformer_name'])
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
            int(electric_grid_data.electric_grid_transformers.loc[
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
                electric_grid_data.electric_grid_transformers.loc[
                    electric_grid_data.electric_grid_transformers['is_phase_1_connected'] == 1,
                    'transformer_name'
                ],
                electric_grid_data.electric_grid_transformers.loc[
                    electric_grid_data.electric_grid_transformers['is_phase_2_connected'] == 1,
                    'transformer_name'
                ],
                electric_grid_data.electric_grid_transformers.loc[
                    electric_grid_data.electric_grid_transformers['is_phase_3_connected'] == 1,
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
                np.repeat(1, sum(electric_grid_data.electric_grid_transformers['is_phase_1_connected'] == 1)),
                np.repeat(2, sum(electric_grid_data.electric_grid_transformers['is_phase_2_connected'] == 1)),
                np.repeat(3, sum(electric_grid_data.electric_grid_transformers['is_phase_3_connected'] == 1))
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
        self.__init__(
            electric_grid_data,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            electric_grid_data: fledge.database_interface.ElectricGridData,
            voltage_no_load_method='by_definition'
    ):

        # Obtain electric grid indexes, via `ElectricGridModel.__init__()`.
        super().__init__(electric_grid_data)

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
            # Obtain phases vector.
            phases_vector = fledge.utils.get_element_phases_array(line)

            # Obtain line resistance / reactance / capacitance matrix entries for the line.
            matrices_index = (
                electric_grid_data.electric_grid_line_types_matrices.loc[:, 'line_type'] == line['line_type']
            )
            resistance_matrix = (
                electric_grid_data.electric_grid_line_types_matrices.loc[matrices_index, 'resistance'].values
            )
            reactance_matrix = (
                electric_grid_data.electric_grid_line_types_matrices.loc[matrices_index, 'reactance'].values
            )
            capacitance_matrix = (
                electric_grid_data.electric_grid_line_types_matrices.loc[matrices_index, 'capacitance'].values
            )

            # Obtain the full line resistance and reactance matrices.
            # Data only contains upper half entries.
            matrices_full_index = (
                np.array([
                    [1, 2, 4],
                    [2, 3, 5],
                    [4, 5, 6]
                ]) - 1
            )
            matrices_full_index = (
                matrices_full_index[:len(phases_vector), :len(phases_vector)]
            )
            resistance_matrix = resistance_matrix[matrices_full_index]
            reactance_matrix = reactance_matrix[matrices_full_index]
            capacitance_matrix = capacitance_matrix[matrices_full_index]

            # Construct line series admittance matrix.
            series_admittance_matrix = (
                np.linalg.inv(
                    (resistance_matrix + 1j * reactance_matrix)
                    * line['length']
                )
            )

            # Construct line shunt admittance.
            # Note: nF to Ω with X = 1 / (2π * f * C)
            # TODO: Check line shunt admittance.
            shunt_admittance_matrix = (
                capacitance_matrix
                * 2 * np.pi * electric_grid_data.electric_grid.at['base_frequency'] * 1e-9
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
                    phase=phases_vector
                )
            )
            node_index_2 = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=line['node_2_name'],
                    phase=phases_vector
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
        for transformer_index, transformer in electric_grid_data.electric_grid_transformers.iterrows():
            # Calculate transformer admittance.
            admittance = (
                (
                    (
                        2 * transformer.at['resistance_percentage'] / 100
                        + 1j * transformer.at['reactance_percentage'] / 100
                    )
                    * (
                        electric_grid_data.electric_grid_nodes.at[transformer.at['node_2_name'], 'voltage'] ** 2
                        / transformer.at['apparent_power']
                    )
                ) ** -1
            )

            # Calculate turn ratio.
            turn_ratio = (
                (
                    1.0  # TODO: Replace `1.0` with actual tap position.
                    * electric_grid_data.electric_grid_nodes.at[transformer.at['node_1_name'], 'voltage']
                )
                / (
                    1.0  # TODO: Replace `1.0` with actual tap position.
                    * electric_grid_data.electric_grid_nodes.at[transformer.at['node_2_name'], 'voltage']
                )
            )

            # Construct transformer element admittance matrices according to:
            # https://doi.org/10.1109/TPWRS.2017.2728618
            if transformer.at['connection'] == "wye-wye":
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
            elif transformer.at['connection'] == "delta-wye":
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
            elif transformer.at['connection'] == "wye-delta":
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
            elif transformer.at['connection'] == "delta-delta":
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
                logger.error(f"Unknown transformer type: {transformer.at['connection']}")
                raise ValueError

            # Obtain phases vector.
            phases_vector = fledge.utils.get_element_phases_array(transformer)

            # Obtain element admittance matrices for correct phases.
            admittance_matrix_11 = (
                admittance_matrix_11[np.ix_(phases_vector - 1, phases_vector - 1)]
            )
            admittance_matrix_12 = (
                admittance_matrix_12[np.ix_(phases_vector - 1, phases_vector - 1)]
            )
            admittance_matrix_21 = (
                admittance_matrix_21[np.ix_(phases_vector - 1, phases_vector - 1)]
            )
            admittance_matrix_22 = (
                admittance_matrix_22[np.ix_(phases_vector - 1, phases_vector - 1)]
            )

            # Obtain indexes for positioning the transformer element
            # matrices in the full matrices.
            node_index_1 = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=transformer.at['node_1_name'],
                    phase=phases_vector
                )
            )
            node_index_2 = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=transformer.at['node_2_name'],
                    phase=phases_vector
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

            # Obtain index for positioning node transformation matrix in full transformation matrix.
            node_index = (
                fledge.utils.get_index(
                    self.nodes,
                    node_name=node['node_name']
                )
            )

            # Add node transformation matrix to full transformation matrix.
            self.node_transformation_matrix[np.ix_(node_index, node_index)] = transformation_matrix

        # Add DERs to der incidence matrix.
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
            (
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

        self.__init__(
            electric_grid_data
        )

    @multimethod
    def __init__(
            self,
            electric_grid_data: fledge.database_interface.ElectricGridData
    ):

        # TODO: Add reset method to ensure correct circuit model is set in OpenDSS when handling multiple models.

        # Obtain electric grid indexes, via `ElectricGridModel.__init__()`.
        super().__init__(electric_grid_data)

        # Construct nominal DER power vector.
        self.der_power_vector_nominal = (
            (
                electric_grid_data.electric_grid_ders.loc[:, 'active_power']
                + 1j * electric_grid_data.electric_grid_ders.loc[:, 'reactive_power']
            ).values
        )

        # Clear OpenDSS.
        opendss_command_string = "clear"
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Obtain source voltage.
        source_voltage = (
            electric_grid_data.electric_grid_nodes.at[
                electric_grid_data.electric_grid.at['source_node_name'],
                'voltage'
            ]
        )

        # Add circuit info to OpenDSS command string.
        opendss_command_string = (
            f"set defaultbasefrequency={electric_grid_data.electric_grid.at['base_frequency']}"
            + f"\nnew circuit.{electric_grid_data.electric_grid.at['electric_grid_name']}"
            + f" phases={len(self.phases)}"
            + f" bus1={electric_grid_data.electric_grid.at['source_node_name']}"
            + f" basekv={source_voltage / 1000}"
            + f" mvasc3=9999999999 9999999999"  # Set near-infinite power limit for source node.
        )

        # Create circuit in OpenDSS.
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Define line codes.
        for line_type_index, line_type in electric_grid_data.electric_grid_line_types.iterrows():
            # Obtain line resistance and reactance matrix entries for the line.
            matrices = (
                electric_grid_data.electric_grid_line_types_matrices.loc[
                    (
                        electric_grid_data.electric_grid_line_types_matrices.loc[:, 'line_type']
                        == line_type.at['line_type']
                    ),
                    ['resistance', 'reactance', 'capacitance']
                ]
            )

            # Obtain number of phases.
            # - Only define as line types for as many phases as needed for current grid.
            n_phases = min(line_type.at['n_phases'], len(self.phases))

            # Add line type name and number of phases to OpenDSS command string.
            opendss_command_string = (
                f"new linecode.{line_type.at['line_type']}"
                + f" nphases={n_phases}"
            )

            # Add resistance and reactance matrix entries to OpenDSS command string,
            # with formatting depending on number of phases.
            if n_phases == 1:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f}]".format(*matrices.loc[:, 'resistance'])
                    + " xmatrix = "
                    + "[{:.8f}]".format(*matrices.loc[:, 'reactance'])
                    + " cmatrix = "
                    + "[{:.8f}]".format(*matrices.loc[:, 'capacitance'])
                )
            elif n_phases == 2:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices.loc[:, 'resistance'])
                    + " xmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices.loc[:, 'reactance'])
                    + " cmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices.loc[:, 'capacitance'])
                )
            elif n_phases == 3:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices.loc[:, 'resistance'])
                    + f" xmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices.loc[:, 'reactance'])
                    + f" cmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices.loc[:, 'capacitance'])
                )

            # Create line code in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define lines.
        for line_index, line in electric_grid_data.electric_grid_lines.iterrows():
            # Obtain number of phases for the line.
            n_phases = len(fledge.utils.get_element_phases_array(line))

            # Add line name, phases, node connections, line type and length
            # to OpenDSS command string.
            opendss_command_string = (
                f"new line.{line['line_name']}"
                + f" phases={n_phases}"
                + f" bus1={line['node_1_name']}{fledge.utils.get_element_phases_string(line)}"
                + f" bus2={line['node_2_name']}{fledge.utils.get_element_phases_string(line)}"
                + f" linecode={line['line_type']}"
                + f" length={line['length']}"
            )

            # Create line in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define transformers.
        for transformer_index, transformer in electric_grid_data.electric_grid_transformers.iterrows():
            # Obtain number of phases.
            n_phases = len(fledge.utils.get_element_phases_array(transformer))

            # Add transformer name, number of phases / windings and reactances to OpenDSS command string.
            opendss_command_string = (
                f"new transformer.{transformer.at['transformer_name']}"
                + f" phases={n_phases}"
                + f" windings=2"
                + f" xscarray=[{transformer.at['reactance_percentage']}]"
            )

            # Add windings to OpenDSS command string.
            windings = [1, 2]
            for winding in windings:
                # Obtain nominal voltage level for each winding.
                voltage = electric_grid_data.electric_grid_nodes.at[transformer.at[f'node_{winding}_name'], 'voltage']

                # Obtain node phases connection string for each winding.
                connection = transformer.at['connection'].split('-')[winding - 1]
                if connection == "wye":
                    node_phases_string = (
                        fledge.utils.get_element_phases_string(transformer)
                        + ".0"  # Enforce wye-grounded connection.
                    )
                elif connection == "delta":
                    node_phases_string = (
                        fledge.utils.get_element_phases_string(transformer)
                    )
                else:
                    logger.error(f"Unknown transformer connection type: {connection}")
                    raise ValueError

                # Add node connection, nominal voltage / power, resistance and maximum / minimum tap level
                # to OpenDSS command string for each winding.
                opendss_command_string += (
                    f" wdg={winding}"
                    + f" bus={transformer.at[f'node_{winding}_name']}" + node_phases_string
                    + f" conn={connection}"
                    + f" kv={voltage / 1000}"
                    + f" kva={transformer.at['apparent_power'] / 1000}"
                    + f" %r={transformer.at['resistance_percentage']}"
                    + f" maxtap="
                    + f"{transformer.at['tap_maximum_voltage_per_unit']}"
                    + f" mintap="
                    + f"{transformer.at['tap_minimum_voltage_per_unit']}"
                )

            # Create transformer in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define DERs.
        # TODO: At the moment, all DERs are modelled as loads in OpenDSS.
        for der_index, der in electric_grid_data.electric_grid_ders.iterrows():
            # Obtain number of phases for the DER.
            n_phases = len(fledge.utils.get_element_phases_array(der))

            # Obtain nominal voltage level for the DER.
            voltage = electric_grid_data.electric_grid_nodes.at[der['node_name'], 'voltage']
            # Convert to line-to-neutral voltage for single-phase DERs, according to:
            # https://sourceforge.net/p/electricdss/discussion/861976/thread/9c9e0efb/
            if n_phases == 1:
                voltage /= np.sqrt(3)

            # Add ground-phase connection for single-phase, wye DERs.
            if (n_phases == 1) and (der['connection'] == 'wye'):
                ground_phase_string = ".0"  # TODO: Check if any difference without ".0" for wye-connected DERs.
            else:
                ground_phase_string = ""

            # Add node connection, model type, voltage, nominal power to OpenDSS command string.
            opendss_command_string = (
                f"new load.{der['der_name']}"
                + f" bus1={der['node_name']}{ground_phase_string}{fledge.utils.get_element_phases_string(der)}"
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

            # Create DER in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Obtain voltage bases.
        voltage_bases = (
            np.unique(
                electric_grid_data.electric_grid_nodes.loc[:, 'voltage'].values / 1000
            ).tolist()
        )

        # Set control mode and voltage bases.
        opendss_command_string = (
            f"set voltagebases={voltage_bases}"
            + f"\nset controlmode=off"
            + f"\ncalcvoltagebases"
        )
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Set solution mode to "single snapshot power flow" according to:
        # OpenDSSComDoc, November 2016, page 1
        opendss_command_string = "set mode=0"
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)


class PowerFlowSolution(object):
    """Power flow solution object."""

    der_power_vector: np.ndarray
    node_voltage_vector: np.ndarray
    branch_power_vector_1: np.ndarray
    branch_power_vector_2: np.ndarray
    loss: np.complex


class PowerFlowSolutionFixedPoint(PowerFlowSolution):
    """Fixed point power flow solution object."""

    node_power_vector_wye: np.ndarray
    node_power_vector_delta: np.ndarray

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            **kwargs
    ):

        # Obtain `electric_grid_model`.
        electric_grid_model = ElectricGridModelDefault(scenario_name)

        self.__init__(
            electric_grid_model,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: ElectricGridModelDefault,
            **kwargs
    ):

        # Obtain `der_power_vector`, assuming nominal power conditions.
        der_power_vector = electric_grid_model.der_power_vector_nominal

        self.__init__(
            electric_grid_model,
            der_power_vector,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: ElectricGridModelDefault,
            der_power_vector: np.ndarray,
            **kwargs
    ):

        # Store DER power vector.
        self.der_power_vector = der_power_vector

        # Obtain node power vectors.
        self.node_power_vector_wye = (
            np.transpose([
                electric_grid_model.der_incidence_wye_matrix
                @ self.der_power_vector
            ])
        )
        self.node_power_vector_delta = (
            np.transpose([
                electric_grid_model.der_incidence_delta_matrix
                @ self.der_power_vector
            ])
        )

        # Obtain voltage solution.
        self.node_voltage_vector = (
            self.get_voltage(
                electric_grid_model,
                self.der_power_vector,
                **kwargs
            )
        )

        # Obtain branch flow solution.
        (
            self.branch_power_vector_1,
            self.branch_power_vector_2
        ) = (
            self.get_branch_power(
                electric_grid_model,
                self.node_voltage_vector
            )
        )

        # Obtain loss solution.
        self.loss = (
            self.get_loss(
                electric_grid_model,
                self.node_voltage_vector
            )
        )

    @staticmethod
    def get_voltage(
        electric_grid_model: ElectricGridModelDefault,
        der_power_vector: np.ndarray,
        voltage_iteration_limit=100,
        voltage_tolerance=1e-2
    ):
        """Get nodal voltage vector by solving with the fixed point algorithm.

        - Initial DER power vector / node voltage vector must be a valid
          solution to te fixed-point equation, e.g., a previous solution from a past
          operation point.
        - Fixed point equation according to: <https://arxiv.org/pdf/1702.03310.pdf>
        """

        # Obtain no-source variables for fixed point equation.
        node_admittance_matrix_no_source = (
            electric_grid_model.node_admittance_matrix[np.ix_(
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            )]
        )
        node_transformation_matrix_no_source = (
            electric_grid_model.node_transformation_matrix[np.ix_(
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            )]
        )
        der_incidence_wye_matrix_no_source = (
            electric_grid_model.der_incidence_wye_matrix[
                np.ix_(
                    fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                    range(len(electric_grid_model.ders))
                )
            ]
        )
        der_incidence_delta_matrix_no_source = (
            electric_grid_model.der_incidence_delta_matrix[
                np.ix_(
                    fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                    range(len(electric_grid_model.ders))
                )
            ]
        )

        node_power_vector_wye_no_source = (
            der_incidence_wye_matrix_no_source
            @ np.transpose([der_power_vector.ravel()])
        )
        node_power_vector_delta_no_source = (
            der_incidence_delta_matrix_no_source
            @ np.transpose([der_power_vector.ravel()])
        )
        node_voltage_vector_no_load_no_source = (
            electric_grid_model.node_voltage_vector_no_load[
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            ]
        )

        # Obtain initial nodal power and voltage vectors, assuming no power conditions.
        node_power_vector_wye_initial_no_source = np.zeros(node_power_vector_wye_no_source.shape, dtype=complex)
        node_power_vector_delta_initial_no_source = np.zeros(node_power_vector_delta_no_source.shape, dtype=complex)
        node_voltage_vector_initial_no_source = node_voltage_vector_no_load_no_source

        # Instantiate fixed point iteration variables.
        voltage_iteration = 1
        voltage_change = np.inf

        while (
                (voltage_iteration < voltage_iteration_limit)
                & (voltage_change > voltage_tolerance)
        ):
            # Calculate fixed point equation.
            node_voltage_vector_solution_no_source = (
                node_voltage_vector_no_load_no_source
                + np.transpose([
                    scipy.sparse.linalg.spsolve(
                        node_admittance_matrix_no_source,
                        (
                            (
                                (
                                    np.conj(node_voltage_vector_initial_no_source) ** -1
                                )
                                * np.conj(node_power_vector_wye_no_source)
                            )
                            + (
                                np.transpose(node_transformation_matrix_no_source)
                                @ (
                                    (
                                        (
                                            node_transformation_matrix_no_source
                                            @ np.conj(node_voltage_vector_initial_no_source)
                                        ) ** -1
                                    )
                                    * np.conj(node_power_vector_delta_no_source)
                                )
                            )
                        )
                    )
                ])
            )

            # Calculate voltage change from previous iteration.
            voltage_change = (
                np.max(abs(
                    node_voltage_vector_solution_no_source
                    - node_voltage_vector_initial_no_source
                ))
            )

            # Set voltage solution as initial voltage for next iteration.
            node_voltage_vector_initial_no_source = (
                node_voltage_vector_solution_no_source
            )

            # Increment voltage iteration counter.
            voltage_iteration += 1

        if voltage_iteration >= voltage_iteration_limit:
            # Reaching the iteration limit is considered undesired and therefore triggers a warning.
            logger.warning(
                f"Fixed point voltage solution algorithm reached maximum limit of {voltage_iteration_limit} iterations."
            )

        # Get full voltage vector by concatenating source and calculated voltage.
        node_voltage_vector = (
            np.vstack([
                electric_grid_model.node_voltage_vector_no_load[
                    fledge.utils.get_index(electric_grid_model.nodes, node_type='source')
                ],
                node_voltage_vector_initial_no_source  # Takes value of `node_voltage_vector_solution_no_source`.
            ])
        )
        return node_voltage_vector

    @staticmethod
    def get_branch_power(
        electric_grid_model: ElectricGridModelDefault,
        node_voltage_vector: np.ndarray
    ):
        """Get branch power vectors by calculating power flow with given nodal voltage.

        - Returns two branch power vectors, where `branch_power_vector_1` represents the
          "from"-direction and `branch_power_vector_2` represents the "to"-direction.
        """

        # Obtain branch admittance and incidence matrices.
        branch_admittance_1_matrix = (
            electric_grid_model.branch_admittance_1_matrix
        )
        branch_admittance_2_matrix = (
            electric_grid_model.branch_admittance_2_matrix
        )
        branch_incidence_1_matrix = (
            electric_grid_model.branch_incidence_1_matrix
        )
        branch_incidence_2_matrix = (
            electric_grid_model.branch_incidence_2_matrix
        )

        # Calculate branch power vectors.
        branch_power_vector_1 = (
            (
                branch_incidence_1_matrix
                @ node_voltage_vector
            )
            * np.conj(
                branch_admittance_1_matrix
                @ node_voltage_vector
            )
        )
        branch_power_vector_2 = (
            (
                branch_incidence_2_matrix
                @ node_voltage_vector
            )
            * np.conj(
                branch_admittance_2_matrix
                @ node_voltage_vector
            )
        )

        return (
            branch_power_vector_1,
            branch_power_vector_2
        )

    @staticmethod
    def get_loss(
        electric_grid_model: ElectricGridModelDefault,
        node_voltage_vector: np.ndarray
    ):
        """Get total electric losses with given nodal voltage."""

        # Calculate total losses.
        # TODO: Validate loss solution.
        loss = (
            np.conj(
                np.transpose(node_voltage_vector)
                @ (
                    electric_grid_model.node_admittance_matrix
                    @ node_voltage_vector
                )
            )
        )

        return loss


class PowerFlowSolutionOpenDSS(PowerFlowSolution):
    """OpenDSS power flow solution object."""

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            **kwargs
    ):

        # Obtain `electric_grid_model`.
        electric_grid_model = ElectricGridModelOpenDSS(scenario_name)

        self.__init__(
            electric_grid_model,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: ElectricGridModelOpenDSS,
            **kwargs
    ):

        # Obtain `der_power_vector`, assuming nominal power conditions.
        der_power_vector = electric_grid_model.der_power_vector_nominal

        self.__init__(
            electric_grid_model,
            der_power_vector,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: ElectricGridModelOpenDSS,
            der_power_vector: np.ndarray,
            **kwargs
    ):

        # Store DER power vector.
        self.der_power_vector = der_power_vector

        # Set DER power vector in OpenDSS model.
        for der_index, der_name in enumerate(electric_grid_model.der_names):
            # TODO: For OpenDSS, all DERs are assumed to be loads.
            opendss_command_string = (
                f"load.{der_name}.kw = {- np.real(self.der_power_vector.ravel()[der_index]) / 1000.0}"
                + f"\nload.{der_name}.kvar = {- np.imag(self.der_power_vector.ravel()[der_index]) / 1000.0}"
            )
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Solve OpenDSS model.
        opendssdirect.run_command("solve")

        # Obtain voltage solution.
        self.node_voltage_vector = (
            self.get_voltage(
                electric_grid_model
            )
        )

        # Obtain branch flow solution.
        (
            self.branch_power_vector_1,
            self.branch_power_vector_2
        ) = (
            self.get_branch_power()
        )

        # Obtain loss solution.
        self.loss = (
            self.get_loss()
        )

    @staticmethod
    def get_voltage(
            electric_grid_model: ElectricGridModelOpenDSS
    ):
        """Get nodal voltage vector by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
        """

        # Extract nodal voltage vector.
        # - Voltages are sorted by node names in the fashion as nodes are sorted in
        #   `nodes` in `ElectricGridModelDefault`.
        node_voltage_vector_solution = (
            np.transpose([
                pd.Series(
                    (
                        np.array(opendssdirect.Circuit.AllBusVolts()[0::2])
                        + 1j * np.array(opendssdirect.Circuit.AllBusVolts()[1::2])
                    ),
                    index=opendssdirect.Circuit.AllNodeNames()
                ).reindex(
                    natsort.natsorted(opendssdirect.Circuit.AllNodeNames())
                ).values
            ])
        )

        # Adjust voltage solution for single-phase grid.
        # TODO: Validate single phase behavior of OpenDSS.
        if len(electric_grid_model.phases) == 1:
            node_voltage_vector_solution /= np.sqrt(3)

        return node_voltage_vector_solution

    @staticmethod
    def get_branch_power():
        """Get branch power vectors by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
        """

        # Solve OpenDSS model.
        opendssdirect.run_command("solve")

        # Instantiate branch vectors.
        branch_power_vector_1 = (
            np.full(((opendssdirect.Lines.Count() + opendssdirect.Transformers.Count()), 3), np.nan, dtype=np.complex)
        )
        branch_power_vector_2 = (
            np.full(((opendssdirect.Lines.Count() + opendssdirect.Transformers.Count()), 3), np.nan, dtype=np.complex)
        )

        # Instantiate iteration variables.
        branch_vector_index = 0
        line_index = opendssdirect.Lines.First()

        # Obtain line branch power vectors.
        while line_index > 0:
            branch_power_opendss = np.array(opendssdirect.CktElement.Powers()) * 1000.0
            branch_phase_count = opendssdirect.CktElement.NumPhases()
            branch_power_vector_1[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0:(branch_phase_count * 2):2]
                + 1.0j * branch_power_opendss[1:(branch_phase_count * 2):2]
            )
            branch_power_vector_2[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0 + (branch_phase_count * 2)::2]
                + 1.0j * branch_power_opendss[1 + (branch_phase_count * 2)::2]
            )

            branch_vector_index += 1
            line_index = opendssdirect.Lines.Next()

        # Obtain transformer branch power vectors.
        transformer_index = opendssdirect.Transformers.First()
        while transformer_index > 0:
            branch_power_opendss = np.array(opendssdirect.CktElement.Powers()) * 1000.0
            branch_phase_count = opendssdirect.CktElement.NumPhases()
            skip_phase = 2 if 0 in opendssdirect.CktElement.NodeOrder() else 0  # Ignore ground nodes.
            branch_power_vector_1[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0:(branch_phase_count * 2):2]
                + 1.0j * branch_power_opendss[1:(branch_phase_count * 2):2]
            )
            branch_power_vector_2[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0 + (branch_phase_count * 2) + skip_phase:-skip_phase:2]
                + 1.0j * branch_power_opendss[1 + (branch_phase_count * 2) + skip_phase:-skip_phase:2]
            )

            branch_vector_index += 1
            transformer_index = opendssdirect.Transformers.Next()

        # Reshape branch power vectors to appropriate size and remove entries for nonexistent phases.
        # TODO: Sort vector by branch name if not in order.
        branch_power_vector_1 = branch_power_vector_1.flatten()
        branch_power_vector_2 = branch_power_vector_2.flatten()
        branch_power_vector_1 = np.transpose([branch_power_vector_1[~np.isnan(branch_power_vector_1)]])
        branch_power_vector_2 = np.transpose([branch_power_vector_2[~np.isnan(branch_power_vector_2)]])

        return (
            branch_power_vector_1,
            branch_power_vector_2
        )

    @staticmethod
    def get_loss():
        """Get total loss by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
        """

        # Solve OpenDSS model.
        opendssdirect.run_command("solve")

        # Obtain loss.
        loss = opendssdirect.Circuit.Losses()[0] + 1.0j * opendssdirect.Circuit.Losses()[1]

        return loss


class LinearElectricGridModel(object):
    """Abstract linear electric model object, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    Note:
        This abstract class only defines the expected variables of linear electric grid model objects,
        but does not implement any functionality.

    Attributes:
        electric_grid_model (ElectricGridModelDefault): Electric grid model object.
        power_flow_solution (PowerFlowSolution): Reference power flow solution object.
        sensitivity_voltage_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active wye power vector.
        sensitivity_voltage_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive wye power vector.
        sensitivity_voltage_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active delta power vector.
        sensitivity_voltage_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive delta power vector.
        sensitivity_voltage_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER active power vector.
        sensitivity_voltage_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER reactive power vector.
        sensitivity_voltage_magnitude_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for voltage
            magnitude vector by active wye power vector.
        sensitivity_voltage_magnitude_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive wye power vector.
        sensitivity_voltage_magnitude_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by active delta power vector.
        sensitivity_voltage_magnitude_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive delta power vector.
        sensitivity_voltage_magnitude_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER active power vector.
        sensitivity_voltage_magnitude_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER reactive power vector.
        sensitivity_branch_power_1_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active wye power vector.
        sensitivity_branch_power_1_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive wye power vector.
        sensitivity_branch_power_1_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active delta power vector.
        sensitivity_branch_power_1_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive delta power vector.
        sensitivity_branch_power_1_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER active power vector.
        sensitivity_branch_power_1_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER reactive power vector.
        sensitivity_branch_power_2_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active wye power vector.
        sensitivity_branch_power_2_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive wye power vector.
        sensitivity_branch_power_2_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active delta power vector.
        sensitivity_branch_power_2_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive delta power vector.
        sensitivity_branch_power_2_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER active power vector.
        sensitivity_branch_power_2_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER reactive power vector.
        sensitivity_loss_active_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active wye power vector.
        sensitivity_loss_active_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive wye power vector.
        sensitivity_loss_active_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active delta power vector.
        sensitivity_loss_active_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive delta power vector.
        sensitivity_loss_active_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER active power vector.
        sensitivity_loss_active_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER reactive power vector.
        sensitivity_loss_reactive_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active wye power vector.
        sensitivity_loss_reactive_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive wye power vector.
        sensitivity_loss_reactive_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active delta power vector.
        sensitivity_loss_reactive_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive delta power vector.
        sensitivity_loss_reactive_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER active power vector.
        sensitivity_loss_reactive_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER reactive power vector.
    """

    electric_grid_model: ElectricGridModelDefault
    power_flow_solution: PowerFlowSolution
    sensitivity_voltage_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_voltage_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_voltage_magnitude_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_branch_power_1_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_branch_power_2_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_loss_active_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_loss_active_by_der_power_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_wye_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_wye_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_delta_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_power_delta_reactive: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_der_power_active: scipy.sparse.spmatrix
    sensitivity_loss_reactive_by_der_power_reactive: scipy.sparse.spmatrix

    def define_optimization_variables(
            self,
            optimization_problem: pyo.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):
        """Define decision variables for given `optimization_problem`."""

        # DERs.
        optimization_problem.der_active_power_vector_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.ders.to_list())
        )
        optimization_problem.der_reactive_power_vector_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.ders.to_list())
        )

        # Voltage.
        optimization_problem.voltage_magnitude_vector_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.nodes.to_list())
        )

        # Branch flows.
        optimization_problem.branch_power_vector_1_squared_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.branches.to_list())
        )
        optimization_problem.branch_power_vector_2_squared_change = (
            pyo.Var(timesteps.to_list(), self.electric_grid_model.branches.to_list())
        )

        # Loss.
        optimization_problem.loss_active_change = pyo.Var(timesteps.to_list())
        optimization_problem.loss_reactive_change = pyo.Var(timesteps.to_list())

    def define_optimization_constraints(
            self,
            optimization_problem: pyo.ConcreteModel,
            timesteps=pd.Index([0], name='timestep'),
            voltage_magnitude_vector_minimum: np.ndarray = None,
            voltage_magnitude_vector_maximum: np.ndarray = None,
            branch_power_vector_squared_maximum: np.ndarray = None
    ):
        """Define constraints to express the linear electric grid model equations for given `optimization_problem`."""

        # Instantiate constraint list.
        optimization_problem.linear_electric_grid_model_constraints = pyo.ConstraintList()

        for timestep in timesteps:

            # Voltage.
            for node_index, node in enumerate(self.electric_grid_model.nodes):
                optimization_problem.linear_electric_grid_model_constraints.add(
                    optimization_problem.voltage_magnitude_vector_change[timestep, node]
                    ==
                    sum(
                        self.sensitivity_voltage_magnitude_by_der_power_active[node_index, der_index]
                        * optimization_problem.der_active_power_vector_change[timestep, der]
                        + self.sensitivity_voltage_magnitude_by_der_power_reactive[node_index, der_index]
                        * optimization_problem.der_reactive_power_vector_change[timestep, der]
                        for der_index, der in enumerate(self.electric_grid_model.ders)
                    )
                )

            # Branch flows.
            for branch_index, branch in enumerate(self.electric_grid_model.branches):
                optimization_problem.linear_electric_grid_model_constraints.add(
                    optimization_problem.branch_power_vector_1_squared_change[timestep, branch]
                    ==
                    sum(
                        self.sensitivity_branch_power_1_by_der_power_active[branch_index, der_index]
                        * optimization_problem.der_active_power_vector_change[timestep, der]
                        + self.sensitivity_branch_power_1_by_der_power_reactive[branch_index, der_index]
                        * optimization_problem.der_reactive_power_vector_change[timestep, der]
                        for der_index, der in enumerate(self.electric_grid_model.ders)
                    )
                )
                optimization_problem.linear_electric_grid_model_constraints.add(
                    optimization_problem.branch_power_vector_2_squared_change[timestep, branch]
                    ==
                    sum(
                        self.sensitivity_branch_power_2_by_der_power_active[branch_index, der_index]
                        * optimization_problem.der_active_power_vector_change[timestep, der]
                        + self.sensitivity_branch_power_2_by_der_power_reactive[branch_index, der_index]
                        * optimization_problem.der_reactive_power_vector_change[timestep, der]
                        for der_index, der in enumerate(self.electric_grid_model.ders)
                    )
                )

            # Loss.
            optimization_problem.linear_electric_grid_model_constraints.add(
                optimization_problem.loss_active_change[timestep]
                ==
                sum(
                    self.sensitivity_loss_active_by_der_power_active[0, der_index]
                    * optimization_problem.der_active_power_vector_change[timestep, der]
                    + self.sensitivity_loss_active_by_der_power_reactive[0, der_index]
                    * optimization_problem.der_reactive_power_vector_change[timestep, der]
                    for der_index, der in enumerate(self.electric_grid_model.ders)
                )
            )
            optimization_problem.linear_electric_grid_model_constraints.add(
                optimization_problem.loss_reactive_change[timestep]
                ==
                sum(
                    self.sensitivity_loss_reactive_by_der_power_active[0, der_index]
                    * optimization_problem.der_active_power_vector_change[timestep, der]
                    + self.sensitivity_loss_reactive_by_der_power_reactive[0, der_index]
                    * optimization_problem.der_reactive_power_vector_change[timestep, der]
                    for der_index, der in enumerate(self.electric_grid_model.ders)
                )
            )

        # Voltage.
        if (voltage_magnitude_vector_minimum is not None) or (voltage_magnitude_vector_maximum is not None):
            voltage_magnitude_vector = (  # Define shorthand.
                lambda node:
                np.abs(
                    self.power_flow_solution.node_voltage_vector.ravel()[
                        self.electric_grid_model.nodes.get_loc(node)
                    ]
                )
            )
        if voltage_magnitude_vector_minimum is not None:
            optimization_problem.voltage_magnitude_vector_minimum_constraint = pyo.Constraint(
                timesteps.to_list(),
                self.electric_grid_model.nodes.to_list(),
                rule=lambda optimization_problem, timestep, *node: (
                    optimization_problem.voltage_magnitude_vector_change[timestep, node]
                    + voltage_magnitude_vector(node)
                    >=
                    voltage_magnitude_vector_minimum.ravel()[self.electric_grid_model.nodes.get_loc(node)]
                )
            )
        if voltage_magnitude_vector_maximum is not None:
            optimization_problem.voltage_magnitude_vector_maximum_constraint = pyo.Constraint(
                timesteps.to_list(),
                self.electric_grid_model.nodes.to_list(),
                rule=lambda optimization_problem, timestep, *node: (
                    optimization_problem.voltage_magnitude_vector_change[timestep, node]
                    + voltage_magnitude_vector(node)
                    <=
                    voltage_magnitude_vector_maximum.ravel()[self.electric_grid_model.nodes.get_loc(node)]
                )
            )

        # Branch flows.
        if branch_power_vector_squared_maximum is not None:
            branch_power_vector_1_squared = (  # Define shorthand.
                lambda branch:
                np.abs(
                    self.power_flow_solution.branch_power_vector_1.ravel()[
                        self.electric_grid_model.branches.get_loc(branch)
                    ] ** 2
                )
            )
            optimization_problem.branch_power_vector_1_squared_maximum_constraint = pyo.Constraint(
                timesteps.to_list(),
                self.electric_grid_model.branches.to_list(),
                rule=lambda optimization_problem, timestep, *branch: (
                    optimization_problem.branch_power_vector_1_squared_change[timestep, branch]
                    + branch_power_vector_1_squared(branch)
                    <=
                    branch_power_vector_squared_maximum.ravel()[self.electric_grid_model.branches.get_loc(branch)]
                )
            )
            branch_power_vector_2_squared = (  # Define shorthand.
                lambda branch:
                np.abs(
                    self.power_flow_solution.branch_power_vector_2.ravel()[
                        self.electric_grid_model.branches.get_loc(branch)
                    ] ** 2
                )
            )
            optimization_problem.branch_power_vector_2_squared_maximum_constraint = pyo.Constraint(
                timesteps.to_list(),
                self.electric_grid_model.branches.to_list(),
                rule=lambda optimization_problem, timestep, *branch: (
                    optimization_problem.branch_power_vector_2_squared_change[timestep, branch]
                    + branch_power_vector_2_squared(branch)
                    <=
                    branch_power_vector_squared_maximum.ravel()[self.electric_grid_model.branches.get_loc(branch)]
                )
            )

    def get_optimization_limits_duals(
            self,
            optimization_problem: pyo.ConcreteModel,
            timesteps=pd.Index([0], name='timestep')
    ):

        # Instantiate dual variables.
        voltage_magnitude_vector_minimum_dual = (
            pd.DataFrame(0.0, columns=self.electric_grid_model.nodes, index=timesteps, dtype=np.float)
        )
        voltage_magnitude_vector_maximum_dual = (
            pd.DataFrame(0.0, columns=self.electric_grid_model.nodes, index=timesteps, dtype=np.float)
        )
        branch_power_vector_1_squared_maximum_dual = (
            pd.DataFrame(0.0, columns=self.electric_grid_model.branches, index=timesteps, dtype=np.float)
        )
        branch_power_vector_2_squared_maximum_dual = (
            pd.DataFrame(0.0, columns=self.electric_grid_model.branches, index=timesteps, dtype=np.float)
        )

        # Obtain duals.
        for timestep in timesteps:

            if optimization_problem.find_component('voltage_magnitude_vector_minimum_constraint') is not None:
                for node_index, node in enumerate(self.electric_grid_model.nodes):
                    voltage_magnitude_vector_minimum_dual.at[timestep, node] = (
                        optimization_problem.dual[
                            optimization_problem.voltage_magnitude_vector_minimum_constraint[timestep, node]
                        ]
                    )

            if optimization_problem.find_component('voltage_magnitude_vector_maximum_constraint') is not None:
                for node_index, node in enumerate(self.electric_grid_model.nodes):
                    voltage_magnitude_vector_maximum_dual.at[timestep, node] = (
                        optimization_problem.dual[
                            optimization_problem.voltage_magnitude_vector_maximum_constraint[timestep, node]
                        ]
                    )

            if optimization_problem.find_component('branch_power_vector_1_squared_maximum_constraint') is not None:
                for branch_index, branch in enumerate(self.electric_grid_model.branches):
                    branch_power_vector_1_squared_maximum_dual.at[timestep, branch] = (
                        optimization_problem.dual[
                            optimization_problem.branch_power_vector_1_squared_maximum_constraint[timestep, branch]
                        ]
                    )

            if optimization_problem.find_component('branch_power_vector_2_squared_maximum_constraint') is not None:
                for branch_index, branch in enumerate(self.electric_grid_model.branches):
                    branch_power_vector_2_squared_maximum_dual.at[timestep, branch] = (
                        optimization_problem.dual[
                            optimization_problem.branch_power_vector_2_squared_maximum_constraint[timestep, branch]
                        ]
                    )

        return (
            voltage_magnitude_vector_minimum_dual,
            voltage_magnitude_vector_maximum_dual,
            branch_power_vector_1_squared_maximum_dual,
            branch_power_vector_2_squared_maximum_dual
        )

    def get_optimization_dlmps(
            self,
            optimization_problem: pyo.ConcreteModel,
            price_timeseries: pd.DataFrame,
            timesteps=pd.Index([0], name='timestep')
    ):

        # Obtain duals.
        (
            voltage_magnitude_vector_minimum_dual,
            voltage_magnitude_vector_maximum_dual,
            branch_power_vector_1_squared_maximum_dual,
            branch_power_vector_2_squared_maximum_dual
        ) = self.get_optimization_limits_duals(
            optimization_problem,
            timesteps
        )

        # Instantiate DLMP variables.
        voltage_magnitude_vector_minimum_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        voltage_magnitude_vector_maximum_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        branch_power_vector_1_squared_maximum_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        branch_power_vector_2_squared_maximum_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        loss_active_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        loss_reactive_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )

        electric_grid_energy_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        electric_grid_voltage_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        electric_grid_congestion_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        electric_grid_loss_dlmp = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )

        # Obtain DLMPs.
        for timestep in timesteps:
            voltage_magnitude_vector_minimum_dlmp.loc[timestep, :] = (
                (
                    self.sensitivity_voltage_magnitude_by_der_power_active.transpose()
                    @ np.transpose([voltage_magnitude_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            voltage_magnitude_vector_maximum_dlmp.loc[timestep, :] = (
                (
                    self.sensitivity_voltage_magnitude_by_der_power_active.transpose()
                    @ np.transpose([voltage_magnitude_vector_maximum_dual.loc[timestep, :].values])
                ).ravel()
            )
            branch_power_vector_1_squared_maximum_dlmp.loc[timestep, :] = (
                (
                    self.sensitivity_branch_power_1_by_der_power_active.transpose()
                    @ np.transpose([branch_power_vector_1_squared_maximum_dual.loc[timestep, :].values])
                ).ravel()
            )
            branch_power_vector_2_squared_maximum_dlmp.loc[timestep, :] = (
                (
                    self.sensitivity_branch_power_2_by_der_power_active.transpose()
                    @ np.transpose([branch_power_vector_2_squared_maximum_dual.loc[timestep, :].values])
                ).ravel()
            )
            loss_active_dlmp.loc[timestep, :] = (
                self.sensitivity_loss_active_by_der_power_active.ravel()
                * price_timeseries.at[timestep, 'price_value']
            )
            loss_reactive_dlmp.loc[timestep, :] = (
                -1.0
                * self.sensitivity_loss_reactive_by_der_power_active.ravel()
                * price_timeseries.at[timestep, 'price_value']
            )

            electric_grid_energy_dlmp.loc[timestep, :] = (
                price_timeseries.at[timestep, 'price_value']
            )
        electric_grid_voltage_dlmp = (
            voltage_magnitude_vector_minimum_dlmp
            + voltage_magnitude_vector_maximum_dlmp
        )
        electric_grid_congestion_dlmp = (
            branch_power_vector_1_squared_maximum_dlmp
            + branch_power_vector_2_squared_maximum_dlmp
        )
        electric_grid_loss_dlmp = (
            loss_active_dlmp
            + loss_reactive_dlmp
        )

        return (
            voltage_magnitude_vector_minimum_dlmp,
            voltage_magnitude_vector_maximum_dlmp,
            branch_power_vector_1_squared_maximum_dlmp,
            branch_power_vector_2_squared_maximum_dlmp,
            loss_active_dlmp,
            loss_reactive_dlmp,
            electric_grid_energy_dlmp,
            electric_grid_voltage_dlmp,
            electric_grid_congestion_dlmp,
            electric_grid_loss_dlmp
        )

    def get_optimization_results(
            self,
            optimization_problem: pyo.ConcreteModel,
            power_flow_solution: PowerFlowSolution = None,
            timesteps=pd.Index([0], name='timestep'),
            in_per_unit=False,
            with_mean=False,
    ) -> fledge.utils.ResultsDict:

        # Instantiate results variables.
        der_active_power_vector = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        der_reactive_power_vector = (
            pd.DataFrame(columns=self.electric_grid_model.ders, index=timesteps, dtype=np.float)
        )
        voltage_magnitude_vector = (
            pd.DataFrame(columns=self.electric_grid_model.nodes, index=timesteps, dtype=np.float)
        )
        branch_power_vector_1_squared = (
            pd.DataFrame(columns=self.electric_grid_model.branches, index=timesteps, dtype=np.float)
        )
        branch_power_vector_2_squared = (
            pd.DataFrame(columns=self.electric_grid_model.branches, index=timesteps, dtype=np.float)
        )
        loss_active = pd.DataFrame(columns=['total'], index=timesteps, dtype=np.float)
        loss_reactive = pd.DataFrame(columns=['total'], index=timesteps, dtype=np.float)

        # Obtain results.
        for timestep in timesteps:
            for der_index, der in enumerate(self.electric_grid_model.ders):
                der_active_power_vector.at[timestep, der] = (
                    optimization_problem.der_active_power_vector_change[timestep, der].value
                    + np.real(self.power_flow_solution.der_power_vector[der_index])
                )
                der_reactive_power_vector.at[timestep, der] = (
                    optimization_problem.der_reactive_power_vector_change[timestep, der].value
                    + np.imag(self.power_flow_solution.der_power_vector[der_index])
                )
            for node_index, node in enumerate(self.electric_grid_model.nodes):
                voltage_magnitude_vector.at[timestep, node] = (
                    optimization_problem.voltage_magnitude_vector_change[timestep, node].value
                    + np.abs(self.power_flow_solution.node_voltage_vector[node_index])
                )
            for branch_index, branch in enumerate(self.electric_grid_model.branches):
                branch_power_vector_1_squared.at[timestep, branch] = (
                    optimization_problem.branch_power_vector_1_squared_change[timestep, branch].value
                    + np.abs(self.power_flow_solution.branch_power_vector_1[branch_index] ** 2)
                )
                branch_power_vector_2_squared.at[timestep, branch] = (
                    optimization_problem.branch_power_vector_2_squared_change[timestep, branch].value
                    + np.abs(self.power_flow_solution.branch_power_vector_2[branch_index] ** 2)
                )
            loss_active.at[timestep, 'total'] = (
                optimization_problem.loss_active_change[timestep].value
                + np.real(self.power_flow_solution.loss)
            )
            loss_reactive.at[timestep, 'total'] = (
                optimization_problem.loss_reactive_change[timestep].value
                + np.imag(self.power_flow_solution.loss)
            )

        # Convert in per-unit values.
        if in_per_unit:
            power_flow_solution = self.power_flow_solution if power_flow_solution is None else power_flow_solution
            der_active_power_vector = (
                der_active_power_vector
                / np.real(self.electric_grid_model.der_power_vector_nominal.ravel())
            )
            der_reactive_power_vector = (
                der_reactive_power_vector
                / np.imag(self.electric_grid_model.der_power_vector_nominal.ravel())
            )
            voltage_magnitude_vector = (
                voltage_magnitude_vector
                / abs(power_flow_solution.node_voltage_vector.ravel())
            )
            branch_power_vector_1_squared = (
                branch_power_vector_1_squared
                / abs(power_flow_solution.branch_power_vector_1.ravel() ** 2)
            )
            branch_power_vector_2_squared = (
                branch_power_vector_2_squared
                / abs(power_flow_solution.branch_power_vector_2.ravel() ** 2)
            )
            loss_active = (
                loss_active
                / np.real(power_flow_solution.loss)
            )
            loss_reactive = (
                loss_reactive
                / np.imag(power_flow_solution.loss)
            )

        # Add mean column.
        if with_mean:
            der_active_power_vector['mean'] = der_active_power_vector.mean(axis=1)
            der_reactive_power_vector['mean'] = der_reactive_power_vector.mean(axis=1)
            voltage_magnitude_vector['mean'] = voltage_magnitude_vector.mean(axis=1)
            branch_power_vector_1_squared['mean'] = branch_power_vector_1_squared.mean(axis=1)
            branch_power_vector_2_squared['mean'] = branch_power_vector_2_squared.mean(axis=1)

        return fledge.utils.ResultsDict(
            der_active_power_vector=der_active_power_vector,
            der_reactive_power_vector=der_reactive_power_vector,
            voltage_magnitude_vector=voltage_magnitude_vector,
            branch_power_vector_1_squared=branch_power_vector_1_squared,
            branch_power_vector_2_squared=branch_power_vector_2_squared,
            loss_active=loss_active,
            loss_reactive=loss_reactive
        )


class LinearElectricGridModelGlobal(LinearElectricGridModel):
    """Linear electric grid model object based on global approximations, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    :syntax:
        - ``LinearElectricGridModelGlobal(electric_grid_model, power_flow_solution)``: Instantiate linear electric grid
          model object for given `electric_grid_model` and `power_flow_solution`.
        - ``LinearElectricGridModelGlobal(scenario_name)``: Instantiate linear electric grid model for given
          `scenario_name`. The required `electric_grid_model` is obtained for given `scenario_name` and the
          `power_flow_solution` is obtained for nominal power conditions.

    Parameters:
        electric_grid_model (ElectricGridModelDefault): Electric grid model object.
        power_flow_solution (PowerFlowSolution): Power flow solution object.
        scenario_name (str): FLEDGE scenario name.

    Attributes:
        electric_grid_model (ElectricGridModelDefault): Electric grid model object.
        power_flow_solution (PowerFlowSolution): Reference power flow solution object.
        sensitivity_voltage_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active wye power vector.
        sensitivity_voltage_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive wye power vector.
        sensitivity_voltage_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage vector
            by active delta power vector.
        sensitivity_voltage_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive delta power vector.
        sensitivity_voltage_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER active power vector.
        sensitivity_voltage_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            complex voltage vector by DER reactive power vector.
        sensitivity_voltage_magnitude_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for voltage
            magnitude vector by active wye power vector.
        sensitivity_voltage_magnitude_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive wye power vector.
        sensitivity_voltage_magnitude_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by active delta power vector.
        sensitivity_voltage_magnitude_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive delta power vector.
        sensitivity_voltage_magnitude_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER active power vector.
        sensitivity_voltage_magnitude_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER reactive power vector.
        sensitivity_branch_power_1_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active wye power vector.
        sensitivity_branch_power_1_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive wye power vector.
        sensitivity_branch_power_1_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by active delta power vector.
        sensitivity_branch_power_1_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 ('from' direction) by reactive delta power vector.
        sensitivity_branch_power_1_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER active power vector.
        sensitivity_branch_power_1_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 1 by DER reactive power vector.
        sensitivity_branch_power_2_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active wye power vector.
        sensitivity_branch_power_2_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive wye power vector.
        sensitivity_branch_power_2_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by active delta power vector.
        sensitivity_branch_power_2_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 ('to' direction) by reactive delta power vector.
        sensitivity_branch_power_2_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER active power vector.
        sensitivity_branch_power_2_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            squared branch flow power vector 2 by DER reactive power vector.
        sensitivity_loss_active_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active wye power vector.
        sensitivity_loss_active_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive wye power vector.
        sensitivity_loss_active_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by active delta power vector.
        sensitivity_loss_active_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by reactive delta power vector.
        sensitivity_loss_active_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER active power vector.
        sensitivity_loss_active_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            active loss by DER reactive power vector.
        sensitivity_loss_reactive_by_power_wye_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active wye power vector.
        sensitivity_loss_reactive_by_power_wye_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive wye power vector.
        sensitivity_loss_reactive_by_power_delta_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by active delta power vector.
        sensitivity_loss_reactive_by_power_delta_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by reactive delta power vector.
        sensitivity_loss_reactive_by_der_power_active (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER active power vector.
        sensitivity_loss_reactive_by_der_power_reactive (scipy.sparse.spmatrix): Sensitivity matrix for
            reactive loss by DER reactive power vector.
    """

    @multimethod
    def __init__(
            self,
            scenario_name: str,
    ):

        # Obtain electric grid model.
        electric_grid_model = (
            ElectricGridModelDefault(scenario_name)
        )

        # Obtain der power vector.
        der_power_vector = (
            electric_grid_model.der_power_vector_nominal
        )

        # Obtain power flow solution.
        power_flow_solution = (
            PowerFlowSolutionFixedPoint(
                electric_grid_model,
                der_power_vector
            )
        )

        self.__init__(
            electric_grid_model,
            power_flow_solution
        )

    @multimethod
    def __init__(
            self,
            electric_grid_model: ElectricGridModelDefault,
            power_flow_solution: PowerFlowSolution
    ):
        # TODO: Validate linear model with delta DERs.

        # Store power flow solution.
        self.power_flow_solution = power_flow_solution

        # Store electric grid model.
        self.electric_grid_model = electric_grid_model

        # Obtain shorthands for no-source matrices and vectors.
        node_admittance_matrix_no_source = (
            electric_grid_model.node_admittance_matrix[np.ix_(
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            )]
        )
        node_transformation_matrix_no_source = (
            electric_grid_model.node_transformation_matrix[np.ix_(
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            )]
        )
        node_voltage_no_source = (
            self.power_flow_solution.node_voltage_vector[
                fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
            ]
        )

        # Instantiate voltage sensitivity matrices.
        self.sensitivity_voltage_by_power_wye_active = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_wye_reactive = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_delta_active = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )
        self.sensitivity_voltage_by_power_delta_reactive = (
            scipy.sparse.dok_matrix(
                (len(electric_grid_model.nodes), len(electric_grid_model.nodes)),
                dtype=complex
            )
        )

        # Calculate voltage sensitivity matrices.
        # TODO: Document the change in sign in the reactive part compared to Hanif.
        self.sensitivity_voltage_by_power_wye_active[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                node_admittance_matrix_no_source.tocsc(),
                scipy.sparse.diags(np.conj(node_voltage_no_source).ravel() ** -1, format='csc')
            )
        )
        self.sensitivity_voltage_by_power_wye_reactive[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                1.0j * node_admittance_matrix_no_source.tocsc(),
                scipy.sparse.diags(np.conj(node_voltage_no_source).ravel() ** -1, format='csc')
            )
        )
        self.sensitivity_voltage_by_power_delta_active[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                node_admittance_matrix_no_source.tocsc(),
                np.transpose(node_transformation_matrix_no_source)
            )
            @ scipy.sparse.diags(
                (
                    (
                        node_transformation_matrix_no_source
                        @ np.conj(node_voltage_no_source)
                    ) ** -1
                ).ravel()
            )
        )
        self.sensitivity_voltage_by_power_delta_reactive[np.ix_(
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
            fledge.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        )] = (
            scipy.sparse.linalg.spsolve(
                1.0j * node_admittance_matrix_no_source.tocsc(),
                np.transpose(node_transformation_matrix_no_source)
            )
            @ scipy.sparse.diags(
                (
                    (
                        node_transformation_matrix_no_source
                        * np.conj(node_voltage_no_source)
                    ) ** -1
                ).ravel()
            )
        )

        self.sensitivity_voltage_by_der_power_active = (
            self.sensitivity_voltage_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_by_der_power_reactive = (
            self.sensitivity_voltage_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        self.sensitivity_voltage_magnitude_by_power_wye_active = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_wye_active
            )
        )
        self.sensitivity_voltage_magnitude_by_power_wye_reactive = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_wye_reactive
            )
        )
        self.sensitivity_voltage_magnitude_by_power_delta_active = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_delta_active
            )
        )
        self.sensitivity_voltage_magnitude_by_power_delta_reactive = (
            scipy.sparse.diags(abs(self.power_flow_solution.node_voltage_vector).ravel() ** -1)
            @ np.real(
                scipy.sparse.diags(np.conj(self.power_flow_solution.node_voltage_vector).ravel())
                @ self.sensitivity_voltage_by_power_delta_reactive
            )
        )

        self.sensitivity_voltage_magnitude_by_der_power_active = (
            self.sensitivity_voltage_magnitude_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_magnitude_by_der_power_reactive = (
            self.sensitivity_voltage_magnitude_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Caculate branch flow sensitivity matrices.
        # TODO: Document the removed factor two compared to Hanif.
        sensitivity_branch_power_1_by_voltage = (
            scipy.sparse.diags(np.conj(
                electric_grid_model.branch_admittance_1_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ electric_grid_model.branch_incidence_1_matrix
            + scipy.sparse.diags((
                electric_grid_model.branch_incidence_1_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ np.conj(electric_grid_model.branch_admittance_1_matrix)
        )
        sensitivity_branch_power_2_by_voltage = (
            scipy.sparse.diags(np.conj(
                electric_grid_model.branch_admittance_2_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ electric_grid_model.branch_incidence_2_matrix
            + scipy.sparse.diags((
                electric_grid_model.branch_incidence_2_matrix
                @ self.power_flow_solution.node_voltage_vector
            ).ravel())
            @ np.conj(electric_grid_model.branch_admittance_2_matrix)
        )

        self.sensitivity_branch_power_1_by_power_wye_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                )
            ])
        )
        self.sensitivity_branch_power_1_by_power_wye_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                )
            ])
        )
        self.sensitivity_branch_power_1_by_power_delta_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                )
            ])
        )
        self.sensitivity_branch_power_1_by_power_delta_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_1).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_1).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_1_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_wye_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_active)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_wye_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_wye_reactive)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_delta_active = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_active)
                )
            ])
        )
        self.sensitivity_branch_power_2_by_power_delta_reactive = (
            scipy.sparse.hstack([
                scipy.sparse.diags(np.real(self.power_flow_solution.branch_power_vector_2).ravel()),
                scipy.sparse.diags(np.imag(self.power_flow_solution.branch_power_vector_2).ravel())
            ])
            @ scipy.sparse.vstack([
                np.real(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                ),
                np.imag(
                    sensitivity_branch_power_2_by_voltage
                    @ np.conj(self.sensitivity_voltage_by_power_delta_reactive)
                )
            ])
        )

        self.sensitivity_branch_power_1_by_der_power_active = (
            self.sensitivity_branch_power_1_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_1_by_der_power_reactive = (
            self.sensitivity_branch_power_1_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_active = (
            self.sensitivity_branch_power_2_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_reactive = (
            self.sensitivity_branch_power_2_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Caculate loss sensitivity matrices.
        # TODO: Document the inverted real / imag parts compared to Hanif.
        sensitivity_loss_by_voltage = (
            np.transpose(self.power_flow_solution.node_voltage_vector)
            @ np.conj(electric_grid_model.node_admittance_matrix)
            + np.transpose(
                electric_grid_model.node_admittance_matrix
                @ self.power_flow_solution.node_voltage_vector
            )
        )

        self.sensitivity_loss_active_by_power_wye_active = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_active
            )
        )
        self.sensitivity_loss_active_by_power_wye_reactive = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_active_by_power_delta_active = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_active
            )
        )
        self.sensitivity_loss_active_by_power_delta_reactive = (
            np.imag(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_reactive
            )
        )

        self.sensitivity_loss_reactive_by_power_wye_active = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_active
            )
        )
        self.sensitivity_loss_reactive_by_power_wye_reactive = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_active = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_active
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_reactive = (
            np.real(
                sensitivity_loss_by_voltage
                @ self.sensitivity_voltage_by_power_delta_reactive
            )
        )

        self.sensitivity_loss_active_by_der_power_active = (
            self.sensitivity_loss_active_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_active_by_der_power_reactive = (
            self.sensitivity_loss_active_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_active = (
            self.sensitivity_loss_reactive_by_power_wye_active
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_reactive = (
            self.sensitivity_loss_reactive_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
