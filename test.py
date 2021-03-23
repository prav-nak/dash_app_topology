from __future__ import division

from jupyter_dash import JupyterDash
import dash_html_components as html
import dash_core_components as dcc
from plotly import graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np

# Initialize the app
app = JupyterDash(__name__)
app.config.suppress_callback_exceptions = True


def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list


app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('Parameters'),

                                    html.H2('Parameters for optimization'),
                                    html.Div([html.H1('nx: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='nx', type='text')],style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([html.H1('ny: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='ny', type='text')],style={'display': 'inline-block', 'width': '25%'}),

                                    html.Hr(),
                                    html.Div([html.Button('Run', id='run-data', n_clicks=0)])
                                ]
                             )
                              ])
        ]

)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
