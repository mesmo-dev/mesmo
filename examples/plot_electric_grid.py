"""Script to generate an network plot of the electric grid for given scenario.

- This script depends on `contextily`, which is not included in the package dependencies, but can be installed
  via `conda install -c conda-forge contextily`.
"""

import contextily as ctx  # TODO: Document contextily dependency.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd

import fledge.data_interface
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_all'
    results_path = fledge.utils.get_results_path('plot_electric_grid', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid data.
    electric_grid_data = fledge.data_interface.ElectricGridData(scenario_name)

    # Create electric grid graph.
    electric_grid_graph = nx.DiGraph()
    electric_grid_graph.add_edges_from(
        electric_grid_data.electric_grid_lines.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False)
    )
    electric_grid_graph.add_edges_from(
        electric_grid_data.electric_grid_transformers.loc[:, ['node_1_name', 'node_2_name']].itertuples(index=False)
    )

    # Remove nodes without latitude / longitude & obtain node positions / labels.
    electric_grid_graph.remove_nodes_from(
        electric_grid_data.electric_grid_nodes.index[
            electric_grid_data.electric_grid_nodes.loc[:, ['longitude', 'latitude']].isnull().any(axis='columns')
        ]
    )
    node_positions = (
        electric_grid_data.electric_grid_nodes.loc[:, ['longitude', 'latitude']].T.to_dict('list')
    )
    node_labels = (
        electric_grid_data.electric_grid_nodes.loc[:, 'node_name'].to_dict()
    )

    # Plot electric grid graph.
    plt.figure(
        figsize=[33.1, 23.4],  # A1 paper size.
        dpi=300
    )
    nx.draw(
        electric_grid_graph,
        pos=node_positions,
        labels=node_labels,
        arrows=False,
        node_size=1.0,
        width=0.25,
        font_size=0.25
    )
    ctx.add_basemap(
        plt.gca(),
        crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
        source=ctx.providers.CartoDB.Positron,
        zoom=14
    )
    plt.savefig(os.path.join(results_path, 'electric_grid.pdf'), bbox_inches='tight')
    plt.show()
    plt.close()
    os.startfile(os.path.join(results_path, 'electric_grid.pdf'))

    # Print results path.
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
