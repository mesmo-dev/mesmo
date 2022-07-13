# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash import Dash, dash_table, dcc, html, Input, Output, callback
import dash_auth
import base64
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import mesmo
import os

from module_optimal_battery_sizing_placement.data_interface import data_battery_sizing_placement
from module_optimal_battery_sizing_placement.deterministic_acopf_planning import \
    deterministic_acopf_battery_placement_sizing

import numpy as np
import scipy.sparse as sp
import plotly.express as px
import plotly.graph_objects as go

# data
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
VALID_USERNAME_PASSWORD_PAIRS = {
    'mesmo': 'astar'
}

# function

app = Dash(__name__, suppress_callback_exceptions=True)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

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

class create_figures_battery_sitting_sizing_simple(object):
    def __init__(
            self,
            results,
            linear_electric_grid_model_set
    ):

        self.data_set = data_battery_sizing_placement(os.path.join(os.path.dirname(os.path.normpath(__file__)),
                                                              'module_optimal_battery_sizing_placement',
                                                              'test_case_customized'))

        temp = list(linear_electric_grid_model_set.linear_electric_grid_models.values())
        loss_sensitivity = temp[0].sensitivity_loss_active_by_power_wye_active

        results_battery_capacity = results['battery_capacity']
        results_battery_placement = results['battery_placement_binary']
        results_energy_root_node = results['energy_root_node']
        import_price = self.data_set.annual_average_energy_price[0:results_energy_root_node.size]

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



app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.H1(children='MESMO Demo', style={'color': 'black', 'fontSize': 40, 'textAlign': 'center'},),
    html.Img(src='/assets/mesmo_logo.png', style={'width': '180vh'}, ),
    html.Br(),
    dcc.Link('Go to Demo Page', href='/page-1', style={'color': 'black', 'fontSize': 18, 'textAlign': 'center'}),
    html.Br(),
    #dcc.Link('Go to Page 2', href='/page-2'),
])

page_1_layout = html.Div([
    html.H1('Battery Siting & Sizing Optimization', style={'color': 'black', 'fontSize': 40, 'textAlign': 'center'}),
    html.Div(children=''' Please select your optimization config data file: '''),
    dcc.Upload(
        id='datatable-upload',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
    ),
    dash_table.DataTable(id='datatable-upload-container'),
    html.Div(children=''' Please select your optimization results display mode: '''),
    dcc.Dropdown(['Basic', 'Grande', 'Venti'], 'LA', id='page-1-dropdown'),
    html.Div(id='page-1-content'),
    html.Br(),
    html.Button('Start Optimization', id='submit-val', n_clicks=0,  style={
                    'display': 'inline-block', 'vertical-align': 'middle', 'horizontal-align': 'middle',
                    "min-width": "150px",
                    'height': "25px",
                    "margin-top": "0px",
                    "margin-left": "5px"}
                ),
    html.Br(),
    html.Div(id='container-button-basic',
             children='Enter a value and press submit')
    # dcc.Link('Go to Page 2', href='/page-2'),
    # html.Br(),
    # dcc.Link('Go back to home', href='/'),
])

@callback(Output('page-1-content', 'children'),
              [Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return f'You have selected {value}'


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(io.BytesIO(decoded))

@app.callback(Output('datatable-upload-container', 'data'),
              Output('datatable-upload-container', 'columns'),
              Input('datatable-upload', 'contents'),
              State('datatable-upload', 'filename'))
def update_output(contents, filename):
    if contents is None:
        return [{}], []
    df = parse_contents(contents, filename)
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]

@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
   # State('input-on-submit', 'value')
)
def update_output(n_clicks):

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

    optimization_result_figures = create_figures_battery_sitting_sizing_simple(results, linear_electric_grid_model_set)

    app.result_layout = html.Div([
        html.H1('Results'),
        html.H1(children='Battery Placement Results', style={'color': 'black', 'fontSize': 40, 'textAlign': 'center'},
                ),

        html.Div(children=''' Input Data: '''),

        generate_table(optimization_result_figures.data_set.battery_data),

        html.Div(children=''' Battery Capacity Results: '''),

        dcc.Graph(
            id='example-graph',
            figure=optimization_result_figures.fig,
            style={'height': '40vh', "border": {"width": "2px", "color": "black"}}
        ),

        dcc.Graph(
            id='example-graph_loss',
            figure=optimization_result_figures.fig_4,
            style={'height': '40vh', "border": {"width": "2px", "color": "black"}}
        ),

        dcc.Graph(
            id='example-graph_2',
            figure=optimization_result_figures.fig_2,
            style={'height': '40vh', "border": {"width": "2px", "color": "black"}}
        ),

        dcc.Graph(
            id='example-graph_3',
            figure=optimization_result_figures.fig_3,
            style={'height': '40vh', "border": {"width": "2px", "color": "black"}}
        )
    ])
    return app.result_layout
    # return 'optimization finished, the button has been clicked {} times'.format(
    #     n_clicks
    # )

page_2_layout = html.Div([
    html.H1('Results'),
])


# Update the index
@callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)