"""Plots module."""

import cv2
from multimethod import multimethod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re

import fledge.config
import fledge.data_interface
import fledge.electric_grid_models
import fledge.utils

if fledge.config.config['plots']['add_basemap']:
    # Basemap requires `contextily`, which is an optional dependency, due to needing installation through `conda`.
    import contextily as ctx

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


def plot_electric_grid_transformer_utilization(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
        electric_grid_graph: ElectricGridGraph,
        branch_power_vector_magnitude_per_unit: pd.DataFrame,
        results_path: str,
        make_video: bool = False
):

    for timestep in branch_power_vector_magnitude_per_unit.index:
        vmin = 20.0
        vmax = 120.0
        plt.figure(
            figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
            dpi=300
        )
        plt.title(
            f"Substation utilization: {timestep.strftime('%H:%M:%S') if type(timestep) is pd.Timestamp else timestep}"
        )
        # Plot nodes all nodes, but with node size 0.0, just to get appropriate map extent.
        nx.draw(
            electric_grid_graph,
            edgelist=[],
            pos=electric_grid_graph.node_positions,
            node_size=0.0
        )
        nx.draw(
            electric_grid_graph,
            nodelist=electric_grid_graph.transformer_nodes,
            edgelist=[],
            pos=electric_grid_graph.node_positions,
            node_size=200.0,
            node_color=(
                100.0
                * branch_power_vector_magnitude_per_unit.loc[timestep, electric_grid_model.transformers].mean(
                    level='branch_name'  # Take mean across all phases.
                )
            ).tolist(),
            vmin=vmin,
            vmax=vmax,
            edgecolors='black',
            # Uncomment below to print utilization as node labels.
            # labels=node_utilization.loc[timestep, :].round().astype(np.int).to_dict(),
            # font_size=7.0,
            # font_color='white',
            # font_family='Arial'
        )
        # Adjust axis limits, to get a better view of surrounding map.
        xlim = plt.xlim()
        xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        plt.xlim(xlim)
        ylim = plt.ylim()
        ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
        plt.ylim(ylim)
        # Add colorbar.
        sm = (
            plt.cm.ScalarMappable(
                norm=plt.Normalize(
                    vmin=vmin,
                    vmax=vmax
                )
            )
        )
        cb = plt.colorbar(sm, shrink=0.9)
        cb.set_label('Utilization [%]')
        if fledge.config.config['plots']['add_basemap']:
            # Add contextual basemap layer for orientation.
            ctx.add_basemap(
                plt.gca(),
                crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
                source=ctx.providers.CartoDB.Positron,
                attribution=False  # Do not show copyright notice.
            )
        name_string = re.sub(r'\W+', '-', f'{timestep}')
        plt.savefig(os.path.join(results_path, f'transformer_utilization_{name_string}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

    # Stitch images to video.
    if make_video:
        images = []
        for timestep in branch_power_vector_magnitude_per_unit.index:
            if type(timestep) is pd.Timestamp:
                name_string = re.sub(r'\W+', '-', f'{timestep}')
                images.append(cv2.imread(os.path.join(results_path, f'transformer_utilization_{name_string}.png')))
        video_writer = (
            cv2.VideoWriter(
                os.path.join(results_path, 'transformer_utilization.avi'),  # Filename.
                cv2.VideoWriter_fourcc(*'XVID'),  # Format.
                2.0,  # FPS.
                images[0].shape[1::-1]  # Size.
            )
        )
        for image in images:
            video_writer.write(image)
        video_writer.release()
        cv2.destroyAllWindows()


def plot_electric_grid_line_utilization(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
        electric_grid_graph: ElectricGridGraph,
        branch_power_vector_magnitude_per_unit: pd.DataFrame,
        results_path: str,
        make_video: bool = False
):

    for timestep in branch_power_vector_magnitude_per_unit.index:
        vmin = 20.0
        vmax = 120.0
        plt.figure(
            figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
            dpi=300
        )
        plt.title(
            f"Line utilization: {timestep.strftime('%H:%M:%S') if type(timestep) is pd.Timestamp else timestep}"
        )
        nx.draw(
            electric_grid_graph,
            nodelist=electric_grid_graph.transformer_nodes,
            edgelist=[],
            pos=electric_grid_graph.node_positions,
            node_size=100.0,
            node_color='red'
        )
        nx.draw(
            electric_grid_graph,
            edgelist=electric_grid_graph.edge_by_line_name.loc[electric_grid_model.line_names].tolist(),
            pos=electric_grid_graph.node_positions,
            node_size=10.0,
            node_color='black',
            arrows=False,
            width=5.0,
            edge_vmin=vmin,
            edge_vmax=vmax,
            edge_color=(
                100.0
                * branch_power_vector_magnitude_per_unit.loc[timestep, electric_grid_model.lines].mean(
                    level='branch_name'  # Take mean across all phases.
                )
            ).tolist(),
        )
        # Adjust axis limits, to get a better view of surrounding map.
        xlim = plt.xlim()
        xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        plt.xlim(xlim)
        ylim = plt.ylim()
        ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
        plt.ylim(ylim)
        # Add colorbar.
        sm = (
            plt.cm.ScalarMappable(
                norm=plt.Normalize(
                    vmin=vmin,
                    vmax=vmax
                )
            )
        )
        cb = plt.colorbar(sm, shrink=0.9)
        cb.set_label('Utilization [%]')
        if fledge.config.config['plots']['add_basemap']:
            # Add contextual basemap layer for orientation.
            ctx.add_basemap(
                plt.gca(),
                crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
                source=ctx.providers.CartoDB.Positron,
                attribution=False  # Do not show copyright notice.
            )
        name_string = re.sub(r'\W+', '-', f'{timestep}')
        plt.savefig(os.path.join(results_path, f'line_utilization_{name_string}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

    # Stitch images to video.
    if make_video:
        images = []
        for timestep in branch_power_vector_magnitude_per_unit.index:
            if type(timestep) is pd.Timestamp:
                name_string = re.sub(r'\W+', '-', f'{timestep}')
                images.append(cv2.imread(os.path.join(results_path, f'line_utilization_{name_string}.png')))
        video_writer = (
            cv2.VideoWriter(
                os.path.join(results_path, 'line_utilization.avi'),  # Filename.
                cv2.VideoWriter_fourcc(*'XVID'),  # Format.
                2.0,  # FPS.
                images[0].shape[1::-1]  # Size.
            )
        )
        for image in images:
            video_writer.write(image)
        video_writer.release()
        cv2.destroyAllWindows()


def plot_electric_grid_node_voltage_drop(
        electric_grid_model: fledge.electric_grid_models.ElectricGridModel,
        electric_grid_graph: ElectricGridGraph,
        node_voltage_vector_magnitude_per_unit: pd.DataFrame,
        results_path: str,
        make_video: bool = False
):

    for timestep in node_voltage_vector_magnitude_per_unit.index:
        vmin = 0.0
        vmax = 10.0
        plt.figure(
            figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
            dpi=300
        )
        plt.title(
            f"Node voltage drop: {timestep.strftime('%H:%M:%S') if type(timestep) is pd.Timestamp else timestep}"
        )
        nx.draw(
            electric_grid_graph,
            nodelist=electric_grid_graph.transformer_nodes,
            edgelist=[],
            pos=electric_grid_graph.node_positions,
            node_size=100.0,
            node_color='red'
        )
        nx.draw(
            electric_grid_graph,
            nodelist=electric_grid_model.node_names.tolist(),
            pos=electric_grid_graph.node_positions,
            node_size=50.0,
            arrows=False,
            vmin=vmin,
            vmax=vmax,
            node_color=(
                -100.0
                * (node_voltage_vector_magnitude_per_unit.loc[timestep, :] - 1.0).mean(
                    level='node_name'  # Take mean across all phases.
                )
            ).tolist(),
            edgecolors='black',
        )
        # Adjust axis limits, to get a better view of surrounding map.
        xlim = plt.xlim()
        xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
        plt.xlim(xlim)
        ylim = plt.ylim()
        ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
        plt.ylim(ylim)
        # Add colorbar.
        sm = (
            plt.cm.ScalarMappable(
                norm=plt.Normalize(
                    vmin=vmin,
                    vmax=vmax
                )
            )
        )
        cb = plt.colorbar(sm, shrink=0.9)
        cb.set_label('Voltage drop [%]')
        if fledge.config.config['plots']['add_basemap']:
            # Add contextual basemap layer for orientation.
            ctx.add_basemap(
                plt.gca(),
                crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
                source=ctx.providers.CartoDB.Positron,
                attribution=False  # Do not show copyright notice.
            )
        name_string = re.sub(r'\W+', '-', f'{timestep}')
        plt.savefig(os.path.join(results_path, f'node_voltage_drop_{name_string}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

    # Stitch images to video.
    if make_video:
        images = []
        for timestep in node_voltage_vector_magnitude_per_unit.index:
            if type(timestep) is pd.Timestamp:
                name_string = re.sub(r'\W+', '-', f'{timestep}')
                images.append(cv2.imread(os.path.join(results_path, f'node_voltage_drop_{name_string}.png')))
        video_writer = (
            cv2.VideoWriter(
                os.path.join(results_path, 'node_voltage_drop.avi'),  # Filename.
                cv2.VideoWriter_fourcc(*'XVID'),  # Format.
                2.0,  # FPS.
                images[0].shape[1::-1]  # Size.
            )
        )
        for image in images:
            video_writer.write(image)
        video_writer.release()
        cv2.destroyAllWindows()
