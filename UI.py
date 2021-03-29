from __future__ import division

# System imports
import math
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from plotly import graph_objs as go
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
from plotly.graph_objects import Layout
from dash.dependencies import Input, Output, State

# Custom imports
from topCode import main

# Initialize the app
app = JupyterDash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

myGlobalStr = ""

app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.H1("Parameters for optimization"),
                        html.Div(
                            [html.H1("Number of elements in each direction: ")],
                            style={"display": "inline-block", "width": "100%"},
                        ),
                        html.Div(
                            [html.H1("nx: ")],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    placeholder=180,
                                    style={"width": 80, "textAlign": "center"},
                                    id="nx",
                                    min=0,
                                    max=500,
                                    step=1,
                                    type="number",
                                )
                            ],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Div(
                            [html.H1("ny: ")],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    placeholder=60,
                                    style={"width": 80, "textAlign": "center"},
                                    id="ny",
                                    min=0,
                                    max=500,
                                    step=1,
                                    type="number",
                                )
                            ],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            [html.H1("Point force location: ")],
                            style={"display": "inline-block", "width": "100%"},
                        ),
                        dcc.Dropdown(
                            id="force-dropdown",
                            options=[
                                {"label": "Top right", "value": "tr"},
                                {"label": "Top left", "value": "tl"},
                                {"label": "Bottom right", "value": "br"},
                                {"label": "Bottom left", "value": "bl"},
                            ],
                            value="tr",
                        ),
                        # html.Div(id='force-output-container'),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            [html.H1("Point force vector: ")],
                            style={"display": "inline-block", "width": "100%"},
                        ),
                        html.Div(
                            [html.H1("fx: ")],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    placeholder=0.0,
                                    style={"width": 80, "textAlign": "center"},
                                    id="fx",
                                    min=-50,
                                    max=50,
                                    step=0.1,
                                    type="number",
                                )
                            ],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Div(
                            [html.H1("fy: ")],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    placeholder=-1.0,
                                    style={"width": 80, "textAlign": "center"},
                                    id="fy",
                                    min=-50,
                                    max=50,
                                    step=0.1,
                                    type="number",
                                )
                            ],
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            [html.H1("Fixed boundary: ")],
                            style={"display": "inline-block", "width": "100%"},
                        ),
                        dcc.Dropdown(
                            id="bc-dropdown",
                            options=[
                                {"label": "Top", "value": "t"},
                                {"label": "Bottom", "value": "b"},
                                {"label": "Right", "value": "r"},
                                {"label": "Left", "value": "l"},
                            ],
                            value="l",
                        ),
                        html.Div(id="bc-output-container"),
                        #
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            [html.H1("Solver library: ")],
                            style={"display": "inline-block", "width": "100%"},
                        ),
                        dcc.RadioItems(
                            id="radio-solver",
                            options=[
                                {"label": "scipy", "value": "scipy"},
                                {"label": "cupy", "value": "cupy"},
                                {"label": "pycuda", "value": "pycuda"},
                            ],
                            value="scipy",
                        ),
                        html.Hr(),
                        html.Div(
                            [html.Button("Draw", id="draw-data", n_clicks=0)],
                            style={"display": "inline-block", "width": "50%"},
                        ),
                        html.Div(
                            [html.Button("Run", id="run-data", n_clicks=0)],
                            style={"display": "inline-block", "width": "50%"},
                        ),
                    ],
                ),
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="timeseries-initial",
                                            config={"displayModeBar": False},
                                        )
                                    ]
                                )
                            ),
                            style={"height": "25%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="timeseries",
                                            config={"displayModeBar": False},
                                        )
                                    ]
                                )
                            ),
                            style={"height": "25%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    id="textarea-example-output",
                                    style={
                                        "whiteSpace": "pre-line",
                                        "maxHeight": "50%",
                                        "font-size": "20px",
                                    },
                                )
                            ),
                            style={"height": "45%"},
                        ),
                        dcc.Interval(id="my-interval", interval=1000),  # one tick each 5 seconds
                    ],
                ),
            ],
        )
    ]
)


@app.callback(
    Output("timeseries-initial", "figure"),
    [
        Input("draw-data", "n_clicks"),
        dash.dependencies.Input("bc-dropdown", "value"),
        dash.dependencies.Input("force-dropdown", "value"),
    ],
    [
        State("nx", "value"),
        State("ny", "value"),
        State("fx", "value"),
        State("fy", "value"),
    ],
)
def update_plot(nclicks, bcloc, floc, nx, ny, fx, fy):

    if nclicks == 0:
        layout = Layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig = go.Figure(
            go.Scatter(x=[0, 0, 0, 0, 0], y=[0, 0, 0, 0, 0], fill="toself"),
            layout=layout,
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        )
        fig.update_layout(
            title="Domain",
            legend_title="Optimal topology",
            font=dict(family="Courier New, monospace", size=18, color="White"),
        )
        fig.update_layout(
            # width=800,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    layout = Layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig = go.Figure(
        go.Scatter(x=[0, 0, nx, nx, 0], y=[0, ny, ny, 0, 0], fill="toself"),
        layout=layout,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title="Domain",
        legend_title="Optimal topology",
        font=dict(family="Courier New, monospace", size=18, color="White"),
    )

    if fx == 0:
        ay = ny + np.sign(fy) * 5
        # ax=nx
    else:
        theta = math.atan2(fy, fx)
        ay = ny - 10 * math.sin(theta)
        # ax = nx

    theta = math.atan2(fy, fx)
    if floc == "tr" or floc == "br":
        ax = nx - 10 * math.cos(theta)
    else:
        ax = -10 * math.cos(theta)
    # ay = ny

    if floc == "tr":
        xloc = nx
        yloc = ny
    elif floc == "tl":
        xloc = 0
        yloc = ny
    elif floc == "br":
        xloc = nx
        yloc = 0
    elif floc == "bl":
        xloc = 0
        yloc = 0

    fig.add_annotation(
        x=xloc,  # arrows' head
        y=yloc,  # arrows' head
        ax=ax,  # arrows' tail
        ay=ay,  # arrows' tail
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
    )

    if bcloc == "l":
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=ny + 1,
            line=dict(
                color="Red",
            ),
            xref="x",
            yref="y",
        )
    elif bcloc == "r":
        fig.add_shape(
            type="line",
            x0=nx + 1,
            y0=0,
            x1=nx + 1,
            y1=ny + 1,
            line=dict(
                color="Red",
            ),
            xref="x",
            yref="y",
        )
    elif bcloc == "b":
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=nx + 1,
            y1=0,
            line=dict(
                color="Red",
            ),
            xref="x",
            yref="y",
        )
    elif bcloc == "t":
        fig.add_shape(
            type="line",
            x0=0,
            y0=ny + 1,
            x1=nx + 1,
            y1=ny + 1,
            line=dict(
                color="Red",
            ),
            xref="x",
            yref="y",
        )

    return fig


@app.callback(Output("textarea-example-output", "children"), Input("my-interval", "n_intervals"))
def callback_func(interval):
    with open("filename.txt", "r") as f:
        val = f.readlines()
    return html.Div(val)


@app.callback(
    Output("timeseries", "figure"),
    [
        Input("run-data", "n_clicks"),
        dash.dependencies.Input("force-dropdown", "value"),
        dash.dependencies.Input("bc-dropdown", "value"),
        Input("radio-solver", "value"),
    ],
    [
        State("nx", "value"),
        State("ny", "value"),
        State("fx", "value"),
        State("fy", "value"),
    ],
)
def update_output_run(nclicks, floc, bcloc, solver, nx, ny, fx, fy):

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "No clicks yet"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (nclicks or 0) and (button_id == "run-data"):
        nelx = nx
        nely = ny
        volfrac = 0.4
        rmin = 5.4
        penal = 3.0
        ft = 1  # ft==0 -> sens, ft==1 -> dens
        f = open("filename.txt", "w+")
        xPhys = main(nelx, nely, volfrac, penal, rmin, ft, floc, fx, fy, bcloc, f, solver)
        f.close()
        fig = px.imshow(xPhys.reshape((nelx, nely)).T)
        # fig.update_layout(
        fig.update_layout(
            # width=800,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update(layout_coloraxis_showscale=False)
        fig.update(layout_showlegend=False)
        return fig

    else:

        layout = Layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        fig = go.Figure(
            go.Scatter(x=[0, 0, 0, 0, 0], y=[0, 0, 0, 0, 0], fill="toself"),
            layout=layout,
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        )
        fig.update_layout(
            title="Optimal topology",
            legend_title="Optimal topology",
            font=dict(family="Courier New, monospace", size=18, color="White"),
        )

        fig.update_layout(
            # width=800,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig


# The real main driver
if __name__ == "__main__":
    # # Default input parameters
    f = open("filename.txt", "w+")
    f.close()
    app.run_server(debug=True, port=8060)
    # app._terminate_server_for_port("localhost", 8060)
