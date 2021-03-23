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
import cupy as cp
import timeit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/stockdata2.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df['Date'])

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

def run_topopt(nelx,nely,volfrac,penal,rmin,ft):
  # A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
  # MAIN DRIVER

  print("Minimum compliance problem with OC")
  print("ndes: " + str(nelx) + " x " + str(nely))
  print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
  print("Filter method: " + ["Sensitivity based","Density based"][ft])
  # Max and min stiffness
  Emin=1e-9
  Emax=1.0
  # dofs:
  ndof = 2*(nelx+1)*(nely+1)
  # Allocate design variables (as array), initialize and allocate sens.
  x=volfrac * np.ones(nely*nelx,dtype=float)
  xcu=volfrac * cp.ones(nely*nelx,dtype=float)
  xold=x.copy()
  xPhys=x.copy()
  xcuold = xcu.copy()
  xcuPhys = xPhys.copy()
  g=0 # must be initialized to use the NGuyen/Paulino OC approach
  dc=np.zeros((nely,nelx), dtype=float)
  dccu = cp.zeros((nely,nelx), dtype=float)
  # FE: Build the index vectors for the for coo matrix format.
  KE=lk()
  KEcu = lkcu()
  print("KE = ", KE)
  print("KEcu = ", KEcu)
  edofMat=np.zeros((nelx*nely,8),dtype=int)
  edofcuMat=cp.zeros((nelx*nely,8),dtype=int)
  for elx in range(nelx):
    for ely in range(nely):
      el = ely+elx*nely
      n1=(nely+1)*elx+ely
      n2=(nely+1)*(elx+1)+ely
      edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
      edofcuMat[el,:]=cp.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
  # Construct the index pointers for the coo format
  iK = np.kron(edofMat,np.ones((8,1))).flatten()
  jK = np.kron(edofMat,np.ones((1,8))).flatten()    
  iKcu = cp.kron(edofcuMat,cp.ones((8,1))).flatten()
  jKcu = cp.kron(edofcuMat,cp.ones((1,8))).flatten()    
  # Filter: Build (and assemble) the index+data vectors for the coo matrix format
  nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
  iH = np.zeros(nfilter)
  jH = np.zeros(nfilter)
  sH = np.zeros(nfilter)
  iHcu = cp.zeros(nfilter)
  jHcu = cp.zeros(nfilter)
  sHcu = cp.zeros(nfilter)
  cc=0
  for i in range(nelx):
    for j in range(nely):
      row=i*nely+j
      kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
      kk2=int(np.minimum(i+np.ceil(rmin),nelx))
      ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
      ll2=int(np.minimum(j+np.ceil(rmin),nely))
      for k in range(kk1,kk2):
        for l in range(ll1,ll2):
          col=k*nely+l
          fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
          iH[cc]=row
          jH[cc]=col
          sH[cc]=np.maximum(0.0,fac)
          iHcu[cc]=row
          jHcu[cc]=col
          sHcu[cc]=np.maximum(0.0,fac)
          cc=cc+1
  # Finalize assembly and convert to csc format
  H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()
  # Hcu = cupyx.scipy.sparse.coo_matrix((sHcu,(iHcu,jHcu)),shape=(nelx*nely,nelx*nely)).tocsc()
  Hs=H.sum(1)
  # BC's and support
  dofs=np.arange(2*(nelx+1)*(nely+1))
  fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
  free=np.setdiff1d(dofs,fixed)
  # Solution and RHS vectors
  f=np.zeros((ndof,1))
  u=np.zeros((ndof,1))
  # Set load
  f[1,0]=-1
  # Initialize plot and plot the initial design
  plt.ion() # Ensure that redrawing is possible
  fig,ax = plt.subplots()
  im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
  interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
  fig.show()
  # Set loop counter and gradient vectors 
  loop=0
  change=1
  dv = np.ones(nely*nelx)
  dc = np.ones(nely*nelx)
  ce = np.ones(nely*nelx)
  elapsed_normal = 0
  elapsed_cuda = 0
  while change>0.01 and loop<50:
    loop=loop+1
    KEcu.flatten()
    # Setup and solve FE problem
    sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
    K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
    # Remove constrained dofs from matrix
    K = K[free,:][:,free]
    Kcu = cupyx.scipy.sparse.coo_matrix(K)

    # Solve system 
    start_time_normal = timeit.default_timer()
    u[free,0]=spsolve(K,f[free,0])  
    elapsed_normal = elapsed_normal + timeit.default_timer() - start_time_normal
    fcu = cp.array(f[free,0])
    start_time_cuda = timeit.default_timer()
    #sol = cp.sparse.linalg.lsqr(Kcu, fcu)
    sol = cupyx.scipy.sparse.linalg.lsqr(Kcu, fcu)
    elapsed_cuda = elapsed_cuda + timeit.default_timer() - start_time_cuda

  
    # Objective and sensitivity
    ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1)
    obj=( (Emin+xPhys**penal*(Emax-Emin))*ce ).sum()
    dc[:]=(-penal*xPhys**(penal-1)*(Emax-Emin))*ce
    dv[:] = np.ones(nely*nelx)
    # Sensitivity filtering:
    if ft==0:
      dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
    elif ft==1:
      dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
      dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
    # Optimality criteria
    xold[:]=x
    (x[:],g)=oc(nelx,nely,x,volfrac,dc,dv,g)
    # Filter design variables
    if ft==0:   xPhys[:]=x
    elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
    # Compute the change by the inf. norm
    change=np.linalg.norm(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1),np.inf)
    # Plot to screen
    im.set_array(-xPhys.reshape((nelx,nely)).T)
    fig.canvas.draw()
    # Write iteration history to screen (req. Python 2.6 or newer)
    print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
        loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
  # Make sure the plot stays and that the shell remains	
  plt.show()
  print("Normal time = ", elapsed_normal)
  print("Cuda time = ", elapsed_cuda)
  input("Press any key...")
#element stiffness matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
	return (KE)
#element stiffness matrix
def lkcu():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*cp.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
	return (KE)
# Optimality criterion
def oc(nelx,nely,x,volfrac,dc,dv,g):
	l1=0
	l2=1e9
	move=0.2
	# reshape to perform vector operations
	xnew=np.zeros(nelx*nely)
	while (l2-l1)/(l1+l2)>1e-3:
		lmid=0.5*(l2+l1)
		xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
		gt=g+np.sum((dv*(xnew-x)))
		if gt>0 :
			l1=lmid
		else:
			l2=lmid
	return (xnew,gt)


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

# # Callback for timeseries price
# @app.callback(Output('timeseries', 'figure'),
#               [Input('stockselector', 'value')])
# def update_graph(selected_dropdown_value):
#     trace1 = []
#     df_sub = df
#     for stock in selected_dropdown_value:
#         trace1.append(go.Scatter(x=df_sub[df_sub['stock'] == stock].index,
#                                  y=df_sub[df_sub['stock'] == stock]['value'],
#                                  mode='lines',
#                                  opacity=0.7,
#                                  name=stock,
#                                  textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(
#                   colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
#                   template='plotly_dark',
#                   paper_bgcolor='rgba(0, 0, 0, 0)',
#                   plot_bgcolor='rgba(0, 0, 0, 0)',
#                   margin={'b': 15},
#                   hovermode='x',
#                   autosize=True,
#                   title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
#                   xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
#               ),

#               }

#     return figure


if __name__ == '__main__':
    app.run_server(debug=False, port=8050, mode="external")
