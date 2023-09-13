"""Script to generate an interactive network plot of the electric grid for given scenario."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mesmo


def main():
    # Settings.
    scenario_name = "singapore_all"
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain electric grid data / graph.
    electric_grid_data = mesmo.data_interface.ElectricGridData(scenario_name)
    electric_grid_graph = mesmo.legacy_plots.ElectricGridGraph(scenario_name)

    # Obtain substation nodes.
    # - This identification is based on the assumption that all nodes with no DER connected are substation nodes,
    #   which is true for the Singapore synthetic grid test case.
    nodes_substation = electric_grid_data.electric_grid_transformers.loc[
        electric_grid_data.electric_grid_transformers.loc[:, "transformer_name"].str.contains("66kV"), "node_2_name"
    ].values

    # Obtain lat/lon values.
    nodes_values = pd.DataFrame(electric_grid_graph.node_positions).T.rename(
        {0: "longitude", 1: "latitude"}, axis="columns"
    )

    # Create lines.
    lines = pd.DataFrame(electric_grid_graph.edges)
    lines_values = pd.DataFrame(index=range(3 * len(lines)), columns=["longitude", "latitude"])
    # Define start node of line
    lines_values.loc[range(0, 3 * len(lines), 3), :] = nodes_values.reindex(lines.iloc[:, 0]).values
    # Define end node of line
    lines_values.loc[range(1, 3 * len(lines), 3), :] = nodes_values.reindex(lines.iloc[:, 1]).values
    # Define intermediate invalid node which is not plotted, such that lines are only connected from start to end.
    # - This enables us to define all lines as a single "virtual" line, which improves performance.
    lines_values.loc[range(2, 3 * len(lines), 3), :] = np.full((len(lines), 2), np.nan)

    # Obtain zoom / center for interactive plot.
    zoom, center = mesmo.utils.get_plotly_mapbox_zoom_center(
        nodes_values.loc[:, "longitude"].dropna().tolist(), nodes_values.loc[:, "latitude"].dropna().tolist()
    )

    # Create interactive plot.
    figure = go.Figure()
    figure.add_trace(
        go.Scattermapbox(
            lon=lines_values.loc[:, "longitude"],
            lat=lines_values.loc[:, "latitude"],
            line=dict(color="black", width=0.5),
            hoverinfo="none",
            mode="lines",
        )
    )
    figure.add_trace(
        go.Scattermapbox(
            lon=nodes_values.loc[:, "longitude"],
            lat=nodes_values.loc[:, "latitude"],
            text=("Node: " + np.array(electric_grid_graph.nodes, dtype=object)),
            mode="markers",
            hoverinfo="text",
            marker=dict(color="royalblue"),
        )
    )
    figure.add_trace(
        go.Scattermapbox(
            lon=nodes_values.loc[nodes_substation, "longitude"],
            lat=nodes_values.loc[nodes_substation, "latitude"],
            text=("Substation: " + nodes_substation),
            mode="markers",
            hoverinfo="text",
            marker=dict(color="crimson"),
        )
    )
    figure.update(
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            mapbox=go.layout.Mapbox(style="carto-positron", zoom=zoom, center=center),
            xaxis=go.layout.XAxis(showgrid=False, zeroline=False, showticklabels=False, ticks=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks=""),
        )
    )
    figure.write_html(results_path / "electric_grid.html")
    mesmo.utils.launch(results_path / "electric_grid.html")
    figure.write_json(results_path / "electric_grid.json", pretty=True)
    mesmo.utils.launch(results_path / "electric_grid.json")

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
