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
    node: pd.Index
    line_incidence_matrix: scipy.sparse.spmatrix

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

        # Obtain node index set.
        self.nodes = pd.MultiIndex.from_frame(thermal_grid_data.thermal_grid_nodes[['node_name', 'node_type']])

        # Define line incidence_matrix.
        self.line_incidence_matrix = (
            scipy.sparse.dok_matrix((len(self.line_names), len(self.node_names)), dtype=np.int)
        )
        for node_index, node_name in enumerate(self.node_names):
            for line_index, line_name in enumerate(self.line_names):
                if node_name == thermal_grid_data.thermal_grid_lines.at[line_name, 'node_1_name']:
                    self.line_incidence_matrix[line_index, node_index] += -1
                elif node_name == thermal_grid_data.thermal_grid_lines.at[line_name, 'node_2_name']:
                    self.line_incidence_matrix[line_index, node_index] += +1
        self.line_incidence_matrix = self.line_incidence_matrix.tocsr()
