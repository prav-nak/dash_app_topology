from __future__ import division

import dash
from jupyter_dash import JupyterDash
import dash_html_components as html
import dash_core_components as dcc
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px
import xarray as xr
from dash.dependencies import Input, Output, State
import numpy as np
#import cupy as cp
import timeit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt

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
                                #  html.P('Visualising time series with Plotly - Dash.'),
                                #  html.P('Pick one or more stocks from the dropdown below.'),
                                #  html.Div(
                                #      className='div-for-dropdown',
                                #      children=[
                                #          dcc.Dropdown(id='stockselector', options=get_options(df['stock'].unique()),
                                #                       multi=True, value=[df['stock'].sort_values()[0]],
                                #                       style={'backgroundColor': '#1E1E1E'},
                                #                       className='stockselector'
                                #                       ),
                                #      ],
                                #     style={'color': '#1E1E1E'}),
                                    html.H2('Parameters for optimization'),
                                    html.Div([html.H1('nx: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='nx', type='text')],style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([html.H1('ny: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='ny', type='text')],style={'display': 'inline-block', 'width': '25%'}),

                                    # html.Div([html.H1('Target gas: ')], style={'display': 'inline-block', 'width': '49%'}),
                                    # html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='gas_lb', type='text')],style={'display': 'inline-block', 'width': '25%'}),
                                    # html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='gas_ub', type='text')],style={'display': 'inline-block', 'width': '25%'}),

                                    # html.Div([html.H1('Target water: ')], style={'display': 'inline-block', 'width': '49%'}),
                                    # html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='water_lb', type='text')],style={'display': 'inline-block', 'width': '25%'}),
                                    # html.Div([dcc.Input(style={'width': 80, 'textAlign':'center'}, id='water_ub', type='text')],style={'display': 'inline-block', 'width': '25%'}), 
                                    html.Hr(),
                                    html.Div([html.Button('Run', id='run-data', n_clicks=0)])
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True)
                             ])
                              ])
        ]

)

@app.callback(Output('timeseries', 'figure'),
              [Input('run-data', 'n_clicks')], 
              [State('nx', 'value'), State('ny', 'value')])
def update_output_run(nclicks, nx, ny):

    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # print("button-id = ", button_id)

    if (nclicks or 0) and (button_id == "run-data"):
        #df = pd.read_json(jsonified_cleaned_data, orient='split')
        #df['Produce'] = 1
        nelx=180
        nely=60
        volfrac=0.4
        rmin=5.4
        penal=3.0
        ft=1 # ft==0 -> sens, ft==1 -> dens
        #run_topopt(nelx,nely,volfrac,penal,rmin,ft)

        figure = {'data': [],
                'layout': go.Layout(
                    colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                    template='plotly_dark',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    margin={'b': 15},
                    hovermode='x',
                    autosize=True,
                    title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
                    # xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
                ),

                }
        airtemps = xr.tutorial.open_dataset('air_temperature').air.isel(time=500)
        colorbar_title = airtemps.attrs['var_desc'] + '<br>(%s)'%airtemps.attrs['units']
        print(airtemps)
        fig = px.imshow(airtemps, color_continuous_scale='RdBu_r', aspect='equal')
        print("Returning a new figure.")
        return fig
    else:
        return {'data' : [], 'layout' : go.Layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        margin={'b': 15},
                        hovermode='x',
                        autosize=True,
                        title={'text': 'Production', 'font': {'color': 'white', 'size': 30}, 'x': 0.5},
                )}

if __name__ == '__main__':
    app.run_server(debug=False, port=8050, mode="external")
