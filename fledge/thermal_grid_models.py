"""Thermal grid models module."""

from multimethod import multimethod
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.database_interface

logger = fledge.config.get_logger(__name__)


class ThermalGridModel(object):
    """Thermal grid model object."""

    node_names: pd.Index
    line_names: pd.Index
    der_names: pd.Index
    nodes: pd.Index
    branches: pd.Index
    ders = pd.Index
    branch_node_incidence_matrix: scipy.sparse.spmatrix
    der_node_incidence_matrix: scipy.sparse.spmatrix
    der_power_vector_nominal: np.ndarray

    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid data.
        thermal_grid_data = fledge.database_interface.ThermalGridData(scenario_name)

        # Obtain node / line / DER names.
        self.node_names = pd.Index(thermal_grid_data.thermal_grid_nodes['node_name'])
        self.line_names = pd.Index(thermal_grid_data.thermal_grid_lines['line_name'])
        self.der_names = pd.Index(thermal_grid_data.thermal_grid_ders['der_name'])

        # Obtain node / branch / DER index set.
        self.nodes = pd.MultiIndex.from_frame(thermal_grid_data.thermal_grid_nodes[['node_name', 'node_type']])
        self.branches = self.line_names
        self.ders = self.der_names

        # Define branch to node incidence matrix.
        self.branch_node_incidence_matrix = (
            scipy.sparse.dok_matrix((len(self.nodes), len(self.branches)), dtype=np.int)
        )
        for node_index, node in enumerate(self.nodes):
            for branch_index, branch in enumerate(self.branches):
                if node[0] == thermal_grid_data.thermal_grid_lines.at[branch, 'node_1_name']:
                    self.branch_node_incidence_matrix[node_index, branch_index] += -1.0
                elif node[0] == thermal_grid_data.thermal_grid_lines.at[branch, 'node_2_name']:
                    self.branch_node_incidence_matrix[node_index, branch_index] += +1.0
        self.branch_node_incidence_matrix = self.branch_node_incidence_matrix.tocsr()

        # Define DER to node incidence matrix.
        self.der_node_incidence_matrix = (
            scipy.sparse.dok_matrix((len(self.nodes), len(self.ders)), dtype=np.int)
        )
        for node_index, node in enumerate(self.nodes):
            for der_index, der in enumerate(self.ders):
                if node[0] == thermal_grid_data.thermal_grid_ders.at[der, 'node_name']:
                    self.der_node_incidence_matrix[node_index, der_index] = -1.0

        # Obtain DER nominal thermal power vector.
        self.der_power_vector_nominal = thermal_grid_data.thermal_grid_ders.loc[:, 'thermal_power_nominal'].values
