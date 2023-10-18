"""Graph-based plotting functions."""

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from multimethod import multimethod
from netgraph import Graph

import mesmo.data_interface
from mesmo import data_models
from mesmo.plots import constants


class ElectricGridGraph(nx.DiGraph):
    """Electric grid graph object."""

    line_edges: dict[str, tuple[str, str]]
    transformer_edges: dict[str, tuple[str, str]]
    node_positions: dict[str, tuple[float, float]]

    @multimethod
    def __init__(self, scenario_name: str):
        # Obtain electric grid data.
        electric_grid_data = mesmo.data_interface.ElectricGridData(scenario_name)

        self.__init__(electric_grid_data)

    @multimethod
    def __init__(self, electric_grid_data: mesmo.data_interface.ElectricGridData):
        # Create graph
        super().__init__()
        self.add_nodes_from(electric_grid_data.electric_grid_nodes.loc[:, "node_name"].tolist(), layer=0)
        self.add_edges_from(
            electric_grid_data.electric_grid_lines.loc[:, ["node_1_name", "node_2_name"]].itertuples(index=False)
        )
        self.add_edges_from(
            electric_grid_data.electric_grid_transformers.loc[:, ["node_1_name", "node_2_name"]].itertuples(index=False)
        )

        # Obtain edges indexed by line name
        self.line_edges = pd.Series(
            electric_grid_data.electric_grid_lines.loc[:, ["node_1_name", "node_2_name"]].itertuples(index=False),
            index=electric_grid_data.electric_grid_lines.loc[:, "line_name"],
        ).to_dict()

        # Obtain edges indexed by transformer name
        self.transformer_edges = pd.Series(
            electric_grid_data.electric_grid_transformers.loc[:, ["node_1_name", "node_2_name"]].itertuples(
                index=False
            ),
            index=electric_grid_data.electric_grid_transformers.loc[:, "transformer_name"],
        ).to_dict()

        # Apply graph layout for node positions
        graph_layout = Graph(self, node_layout="dot")
        self.node_positions = graph_layout.node_positions


def electric_grid_asset_layout(figure: go.Figure, results: data_models.RunResults) -> go.Figure:
    graph = ElectricGridGraph(results.scenario_name)

    # Plot lines
    line_edge_x = []
    line_edge_y = []
    line_edge_x_label = []
    line_edge_y_label = []
    line_name = []
    for name, edge in graph.line_edges.items():
        x0, y0 = graph.node_positions[edge[0]]
        x1, y1 = graph.node_positions[edge[1]]
        line_edge_x.append(x0)
        line_edge_x.append(x1)
        line_edge_x.append(None)
        line_edge_y.append(y0)
        line_edge_y.append(y1)
        line_edge_y.append(None)
        line_edge_x_label.append((x0 + x1) / 2)
        line_edge_y_label.append((y0 + y1) / 2)
        line_name.append(name)
    figure.add_trace(
        go.Scatter(
            x=line_edge_x,
            y=line_edge_y,
            line=go.scatter.Line(width=2, color="grey"),
            mode="lines",
            name="lines",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=line_edge_x_label,
            y=line_edge_y_label,
            hoverinfo="text",
            hovertext=line_name,
            mode="markers",
            showlegend=False,
            marker=go.scatter.Marker(opacity=0),
        )
    )

    # Plot transformers
    transformer_edge_x = []
    transformer_edge_y = []
    transformer_edge_x_label = []
    transformer_edge_y_label = []
    transformer_name = []
    for name, edge in graph.transformer_edges.items():
        x0, y0 = graph.node_positions[edge[0]]
        x1, y1 = graph.node_positions[edge[1]]
        transformer_edge_x.append(x0)
        transformer_edge_x.append(x1)
        transformer_edge_x.append(None)
        transformer_edge_y.append(y0)
        transformer_edge_y.append(y1)
        transformer_edge_y.append(None)
        transformer_edge_x_label.append((x0 + x1) / 2)
        transformer_edge_y_label.append((y0 + y1) / 2)
        transformer_name.append(name)
    figure.add_trace(
        go.Scatter(
            x=transformer_edge_x,
            y=transformer_edge_y,
            line=go.scatter.Line(width=2, color="black", dash="dot"),
            mode="lines",
            name="transformers",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=transformer_edge_x_label,
            y=transformer_edge_y_label,
            hoverinfo="text",
            hovertext=transformer_name,
            mode="markers",
            showlegend=False,
            marker=go.scatter.Marker(opacity=0),
        )
    )

    node_x = []
    node_y = []
    node_name = []
    for node in graph.nodes():
        x, y = graph.node_positions[node]
        node_x.append(x)
        node_y.append(y)
        node_name.append(node)
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            hovertext=node_name,
            marker=go.scatter.Marker(
                # showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                # colorscale="YlGnBu",
                # reversescale=True,
                color="teal",
                size=10,
                # colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right"),
                # line_width=2,
            ),
            name="nodes",
        )
    )

    title = "Electric grid asset layout"
    legend_title = constants.ValueLabels.ASSETS

    figure.update_layout(
        title=title,
        xaxis=go.layout.XAxis(showgrid=False, visible=False),
        yaxis=go.layout.YAxis(showgrid=False, visible=False),
        legend=go.layout.Legend(title=legend_title, x=0.99, xanchor="auto", y=0.01, yanchor="auto"),
    )
    return figure
