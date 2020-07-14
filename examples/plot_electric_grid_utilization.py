"""Script to generate an utilization plot of the electric grid for given scenario.

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
    results_path = fledge.utils.get_results_path('plot_electric_grid_utilization', scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain electric grid data / graph.
    electric_grid_data = fledge.data_interface.ElectricGridData(scenario_name)
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)

    # Obtain substation nodes / utilization.
    nodes_substation = (
        electric_grid_data.electric_grid_nodes.loc[
            ~electric_grid_data.electric_grid_nodes.loc[:, 'node_name'].isin(
                electric_grid_data.electric_grid_ders.loc[:, 'node_name']
            ),
            'node_name'
        ].tolist()
    )
    node_utilization = (
        pd.Series(
            (np.random.rand(len(nodes_substation)) * 100.0 + 20.0).tolist(),
            index=nodes_substation
        )
    )

    # Plot electric grid substation utilization as nodes.
    plt.figure(
        figsize=[12.0, 6.0],  # Arbitrary convenient figure size.
        dpi=300
    )
    nx.draw(
        electric_grid_graph,
        nodelist=nodes_substation,
        edgelist=[],
        pos=electric_grid_graph.node_positions,
        node_size=200.0,
        node_color=node_utilization.tolist(),
        vmin=20.0,
        vmax=120.0,
        edgecolors='black',
        # Uncomment below to print utilization as node labels.
        # labels=node_utilization.round().astype(np.int).to_dict(),
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
                vmin=20.0,
                vmax=120.0
            )
        )
    )
    cb = plt.colorbar(sm, shrink=0.9)
    cb.set_label('Utilization [%]')
    # Add basemap / open street map for better orientation.
    ctx.add_basemap(
        plt.gca(),
        crs='EPSG:4326',  # Use 'EPSG:4326' for latitude / longitude coordinates.
        source=ctx.providers.CartoDB.Positron,
        attribution=False  # Do not show copyright notice.
    )
    plt.savefig(os.path.join(results_path, 'nodes_utilization.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    # Print results path.
    os.startfile(os.path.join(results_path))
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
