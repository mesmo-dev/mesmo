"""Plots module."""

from multimethod import multimethod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import fledge.config
import fledge.data_interface
import fledge.utils

logger = fledge.config.get_logger(__name__)


class ElectricGridGraph(nx.DiGraph):
    """Electric grid graph object."""

    edge_by_line_name: pd.Series
    transformer_nodes: list
    node_positions: dict
    node_labels: dict

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain electric grid data.
        electric_grid_data = fledge.data_interface.ElectricGridData(scenario_name)

        self.__init__(
            electric_grid_data
        )

    @multimethod
    def __init__(
            self,
            electric_grid_data: fledge.data_interface.ElectricGridData
    ):

        # Create electric grid graph.
        super().__init__()
        self.add_nodes_from(
            electric_grid_data.electric_grid_nodes.loc[:, 'node_name'].tolist()
        )
        self.add_edges_from(
            electric_grid_data.electric_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False)
        )

        # Obtain edges indexed by line name.
        self.edge_by_line_name = (
            pd.Series(
                electric_grid_data.electric_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False),
                index=electric_grid_data.electric_grid_lines.loc[:, 'line_name']
            )
        )

        # Obtain transformer nodes (secondary nodes of transformers).
        self.transformer_nodes = (
            electric_grid_data.electric_grid_transformers.loc[:, 'node_2_name'].tolist()
        )

        # Obtain node positions / labels.
        self.node_positions = (
            electric_grid_data.electric_grid_nodes.loc[:, ['longitude', 'latitude']].T.to_dict('list')
        )
        self.node_labels = (
            electric_grid_data.electric_grid_nodes.loc[:, 'node_name'].to_dict()
        )


class ThermalGridGraph(nx.DiGraph):
    """Electric grid graph object."""

    node_positions: dict
    node_labels: dict

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid data.
        thermal_grid_data = fledge.data_interface.ThermalGridData(scenario_name)

        self.__init__(
            thermal_grid_data
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_data: fledge.data_interface.ThermalGridData
    ):

        # Create thermal grid graph.
        super().__init__()
        self.add_nodes_from(
            thermal_grid_data.thermal_grid_nodes.loc[:, 'node_name'].tolist()
        )
        self.add_edges_from(
            thermal_grid_data.thermal_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False)
        )

        # Obtain node positions / labels.
        self.node_positions = (
            thermal_grid_data.thermal_grid_nodes.loc[:, ['longitude', 'latitude']].T.to_dict('list')
        )
        self.node_labels = (
            thermal_grid_data.thermal_grid_nodes.loc[:, 'node_name'].to_dict()
        )
