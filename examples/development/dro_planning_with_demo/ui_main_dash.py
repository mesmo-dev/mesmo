# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
import time
import mesmo
import os
import numpy as np
import scipy.sparse as sp

from dash import Dash, html, dcc, dash_table
import dash_auth
import plotly.express as px
import pandas as pd

from module_optimal_battery_sizing_placement.data_interface import data_battery_sizing_placement
from module_optimal_battery_sizing_placement.deterministic_acopf_planning import \
    deterministic_acopf_battery_placement_sizing

VALID_USERNAME_PASSWORD_PAIRS = {
    'mesmo': 'zhang'
}

def generate_table(dataframe, max_rows=26):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns]) ] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
        style={'display': 'table-cell', 'border': '2px black', 'fontSize': '4'}
    )

class create_result_webpage(object):
    def __init__(
            self,
            results,
            linear_electric_grid_model_set
    ):
        self.app = Dash(__name__)

        auth = dash_auth.BasicAuth(
            self.app,
            VALID_USERNAME_PASSWORD_PAIRS
        )

        data_set = data_battery_sizing_placement(os.path.join(os.path.dirname(os.path.normpath(__file__)),
                                                              'module_optimal_battery_sizing_placement',
                                                              'test_case_customized'))

        temp = list(linear_electric_grid_model_set.linear_electric_grid_models.values())
        loss_sensitivity = temp[0].sensitivity_loss_active_by_power_wye_active

        results_battery_capacity = results['battery_capacity']
        results_battery_placement = results['battery_placement_binary']
        results_energy_root_node = results['energy_root_node']
        import_price = data_set.annual_average_energy_price[0:results_energy_root_node.size]

        node_index_with_bess_placement = results_battery_placement.values.nonzero()[0]

        temp = results_battery_placement.iloc[node_index_with_bess_placement]['battery_placement_binary']
        node_name_placement = temp.keys().get_level_values('node')
        battery_capacity_placement = results_battery_capacity.iloc[
            node_index_with_bess_placement]['battery_capacity'].values

        node_name_placement_list = list()
        for i in range(node_name_placement.values.size):
            node_name_placement_list.append(str(node_name_placement.values[i]))

        self.df = pd.DataFrame({
            "node": node_name_placement_list,
            "Battery Capacity (kWh)": battery_capacity_placement,
        })

        self.df_energy_import = pd.DataFrame({
            "timestep": results_energy_root_node.index.to_list(),
            "Energy Root Node (kWh)": np.ravel(results_energy_root_node.values),
        })

        self.df_energy_price = pd.DataFrame({
            "timestep": results_energy_root_node.index.to_list(),
            "Energy Price ($/kWh)": import_price.values
        })

        self.df_loss_sensitivity = pd.DataFrame({
            "node index": [str(x) for x in linear_electric_grid_model_set.electric_grid_model.nodes.values],
            "loss sensitivity": loss_sensitivity.toarray()[0]
        })

        self.fig = px.bar(self.df, x="node", y="Battery Capacity (kWh)", barmode="group", width=10)

        self.fig_2 = px.bar(self.df_energy_import, x="timestep", y="Energy Root Node (kWh)", barmode="group")

        self.fig_3 = px.bar(self.df_energy_price, x="timestep", y="Energy Price ($/kWh)", barmode="group")

        self.fig_4 = px.bar(self.df_loss_sensitivity, x="node index", y="loss sensitivity", barmode="group")

        self.app.layout = html.Div(children=[
            html.H1(children='Battery Placement Results', style={'color': 'black', 'fontSize': 40, 'textAlign': 'center'},
                    ),

            html.Div(children=''' Input Data: '''),

            generate_table(data_set.battery_data),

            html.Div(children=''' Battery Capacity Results: '''),

            dcc.Graph(
                id='example-graph',
                figure=self.fig,
                style={'height': '40vh', "border":{"width":"2px", "color":"black"}}
            ),

            dcc.Graph(
                id='example-graph_loss',
                figure=self.fig_4,
                style={'height': '40vh', "border": {"width": "2px", "color": "black"}}
            ),

            dcc.Graph(
                id='example-graph_2',
                figure=self.fig_2,
                style={'height': '40vh', "border": {"width": "2px", "color": "black"}}
            ),

            dcc.Graph(
                id='example-graph_3',
                figure=self.fig_3,
                style={'height': '40vh', "border": {"width": "2px", "color": "black"}}
            )
        ])

        self.app = self.app.run_server(debug=True)



# TO-DO
class plot_grid_topology(object):
    # input: linear_electric_grid_model_set.electric_grid_model.nodes /lines / branches
    def __init__(
            self,
    ):
        print()


class plot_general_results_bess_placement(object):
    def __init__(
            self,
            result,
    ):
        print()



if __name__ == '__main__':
    # Settings.
    scenario_name = 'paper_2021_zhang_dro'
    mesmo.data_interface.recreate_database()
    linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)

    # Obtain data.
    data_set = data_battery_sizing_placement(
        os.path.join(os.path.dirname(os.path.normpath(__file__)),
                     'module_optimal_battery_sizing_placement',
                     'test_case_customized')
    )

    # Get results path.
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Get standard form of stage 1.
    optimal_sizing_problem = deterministic_acopf_battery_placement_sizing(scenario_name, data_set)

    optimal_sizing_problem.optimization_problem.solve()
    results = optimal_sizing_problem.optimization_problem.get_results()

    result_web = create_result_webpage(results, linear_electric_grid_model_set)
    result_web.app.run_server(debug=True)
    print()



