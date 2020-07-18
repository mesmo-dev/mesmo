"""Script to generate an network plot of the electric grid for given scenario.

- This script depends on `contextily`, which is not included in the package dependencies, but can be installed
  under Anaconda via `conda install -c conda-forge contextily`.
"""

import contextily as ctx  # TODO: Document contextily dependency.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd

import fledge.data_interface
import fledge.plots
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_all'
    results_path = fledge.utils.get_results_path('plot_electric_grid', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid data / graph.
    electric_grid_data = fledge.data_interface.ElectricGridData(scenario_name)
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)

    # Obtain substation nodes.
    # - This identification is based on the assumption that all nodes with no DER connected are substation nodes,
    #   which is true for the Singapore synthetic grid test case.
    nodes_substation = (
        electric_grid_data.electric_grid_nodes.loc[
            ~electric_grid_data.electric_grid_nodes.loc[:, 'node_name'].isin(
                electric_grid_data.electric_grid_ders.loc[:, 'node_name']
            ),
            'node_name'
        ].tolist()
    )

    # Plot electric grid graph.
    plt.figure(
        figsize=[33.1, 23.4],  # A1 paper size.
        dpi=300
    )
    nx.draw(
        electric_grid_graph,
        pos=electric_grid_graph.node_positions,
        nodelist=nodes_substation,
        edgelist=[],
        node_color='red',
        node_size=10.0
    )
    nx.draw(
        electric_grid_graph,
        pos=electric_grid_graph.node_positions,
        labels=electric_grid_graph.node_labels,
        arrows=False,
        node_size=1.0,
        width=0.25,
        font_size=0.25
    )
    ctx.add_basemap(
        plt.gca(),
        crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
        source=ctx.providers.CartoDB.Positron,
        zoom=14,
        attribution=False
    )
    plt.savefig(os.path.join(results_path, 'electric_grid.pdf'), bbox_inches='tight')
    plt.show()
    plt.close()

    # Print results path.
    os.startfile(os.path.join(results_path))
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
