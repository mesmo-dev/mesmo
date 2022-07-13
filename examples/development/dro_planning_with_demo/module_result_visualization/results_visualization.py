# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

class create_result_webpage(object):
    def __init__(
            self,
    ):
        self.app = Dash(__name__)

        # assume you have a "long-form" data frame
        # see https://plotly.com/python/px-arguments/ for more options
        self.df = pd.DataFrame({
            "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
            "Amount": [4, 1, 2, 2, 4, 5],
            "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
        })

        self.fig = px.bar(self.df, x="Fruit", y="Amount", color="City", barmode="group")

        self.app.layout = html.Div(children=[
            html.H1(children='Hello MESMO'),

            html.Div(children='''
                Dash: A web application framework for your data.
            '''),

            dcc.Graph(
                id='example-graph',
                figure=self.fig
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
    result_web = create_result_webpage()
    print()



