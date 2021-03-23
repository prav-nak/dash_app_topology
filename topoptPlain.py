# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
import dash
#from jupyter_dash import JupyterDash
import dash_html_components as html
import dash_core_components as dcc
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px
#import xarray as xr
from dash.dependencies import Input, Output, State
import numpy as np
#import cupy as cp
import timeit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt

# Initialize the app
app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True


myGlobalStr = ""
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
                                    html.H1('Parameters for optimization'),
                                    
									html.Div([html.H1('Number of elements in each direction: ')], style={'display': 'inline-block', 'width': '100%'}),
									html.Div([html.H1('nx: ')], style={'display': 'inline-block', 'width': '10%'}),
                                    html.Div([dcc.Input(placeholder=180, style={'width': 80, 'textAlign':'center'}, id='nx', min=0, max=500, step=1, type='number')],style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([html.H1('ny: ')], style={'display': 'inline-block', 'width': '10%'}),
                                    html.Div([dcc.Input(placeholder=60, style={'width': 80, 'textAlign':'center'}, id='ny', min=0, max=500, step=1, type='number')],style={'display': 'inline-block', 'width': '25%'}),
									html.Div([html.Button('Draw', id='draw-data', n_clicks=0)], style={'display': 'inline-block', 'width': '30%'}),
									html.Br(),
									html.Br(),
									html.Br(),
									html.Div([html.H1('Point force location: ')], style={'display': 'inline-block', 'width': '100%'}),
									dcc.Dropdown(id='force-dropdown',
        											options=[
														{'label': 'Top right', 'value': 'tr'},
														{'label': 'Top left', 'value': 'tl'},
														{'label': 'Bottom right', 'value': 'br'},
														{'label': 'Bottom left', 'value': 'bl'}
													],
													value='tr'
										),
									html.Div(id='force-output-container'),
									html.Br(),
									html.Br(),
									html.Br(),
									html.Div([html.H1('Force vector: ')], style={'display': 'inline-block', 'width': '100%'}),
                                    html.Div([html.H1('fx: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(placeholder=0.0, style={'width': 80, 'textAlign':'center'}, id='fx', min=0, max=50, step=0.1, type='number')],style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([html.H1('fy: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(placeholder=-1.0, style={'width': 80, 'textAlign':'center'}, id='fy', min=0, max=50, step=0.1, type='number')],style={'display': 'inline-block', 'width': '25%'}),
									html.Br(),
									html.Br(),
									html.Br(),
									html.Div([html.H1('Boundary condition: ')], style={'display': 'inline-block', 'width': '100%'}),
									dcc.Dropdown(id='bc-dropdown',
        											options=[
														{'label': 'Top right', 'value': 'tr'},
														{'label': 'Top left', 'value': 'tl'},
														{'label': 'Bottom right', 'value': 'br'},
														{'label': 'Bottom left', 'value': 'bl'}
													],
													value='tr'
										),
									html.Div(id='bc-output-container'),
                                    html.Hr(),
                                    html.Div([html.Button('Run', id='run-data', n_clicks=0)])
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                dcc.Graph(id='timeseries-initial', config={'displayModeBar': False}), 
								dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True), 
    							html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line',"maxHeight": '100%', "overflow-y": "scroll"})]),
								dcc.Interval(id='my-interval', interval=1000)  # one tick each 5 seconds
                              ])
        ]

)

@app.callback(Output('timeseries-initial', 'figure'),
              [Input('draw-data', 'n_clicks')], 
              [State('nx', 'value'), State('ny', 'value')])
def update_plot(nclicks, nx, ny):
	fig = go.Figure(go.Scatter(x=[0,0,nx,nx,0], y=[0,ny,ny,0,0], fill="toself"))
	#fig.update_layout(width=nx)
	fig.update_xaxes()
	fig.update_yaxes(    scaleanchor = "x",
    scaleratio = 1,)
	return fig


@app.callback(
    Output('textarea-example-output', 'children'),
    Input('my-interval', 'n_intervals')
)
def callback_func(interval):
	with open('filename.txt', 'r') as f:
		val = f.readlines()
	return html.Div(val)

@app.callback(
    dash.dependencies.Output('force-output-container', 'children'),
    [dash.dependencies.Input('force-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('bc-output-container', 'children'),
    [dash.dependencies.Input('bc-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

@app.callback(Output('timeseries', 'figure'),
              [Input('run-data', 'n_clicks')], 
              [State('nx', 'value'), State('ny', 'value')])
def update_output_run(nclicks, nx, ny):

	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = 'No clicks yet'
	else:
		button_id = ctx.triggered[0]['prop_id'].split('.')[0]

	if (nclicks or 0) and (button_id == "run-data"):
		#df = pd.read_json(jsonified_cleaned_data, orient='split')
		#df['Produce'] = 1
		nelx=nx
		nely=ny
		volfrac=0.4
		rmin=5.4
		penal=3.0
		ft=1 # ft==0 -> sens, ft==1 -> dens
		f = open('filename.txt', 'w+')
		xPhys = main(nelx,nely,volfrac,penal,rmin,ft,f)
		#print(type(xPhys))
		f.close()
		# df = px.data.gapminder().query("country=='Canada'")
		# fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
		# return fig

		img_rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                   ])
		fig = px.imshow(xPhys.reshape((nelx, nely)).T)
		# fig = px.imshow(img_rgb)
		return fig

	else:
		return {'data' : [], 'layout' : go.Layout(
                        template='plotly_dark',
                        paper_bgcolor='rgb(233,233,233)',
                        plot_bgcolor='rgb(233,233,233)',
                        margin={'b': 15},
                        hovermode='x',
                        autosize=True,
                        title={'text': 'Topology optimization', 'font': {'color': 'black', 'size': 30}, 'x': 0.5},
                )}
		# fig = go.Figure(go.Scatter(x=[0,0,180,180,0], y=[0,60,60,0,0], fill="toself"))
		# return fig


def main(nelx,nely,volfrac,penal,rmin,ft, fout):
	myGlobalStr = ""
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
	xold=x.copy()
	xPhys=x.copy()
	g=0 # must be initialized to use the NGuyen/Paulino OC approach
	dc=np.zeros((nely,nelx), dtype=float)
	# FE: Build the index vectors for the for coo matrix format.
	KE=lk()
	edofMat=np.zeros((nelx*nely,8),dtype=int)
	for elx in range(nelx):
		for ely in range(nely):
			el = ely+elx*nely
			n1=(nely+1)*elx+ely
			n2=(nely+1)*(elx+1)+ely
			edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
	# Construct the index pointers for the coo format
	iK = np.kron(edofMat,np.ones((8,1))).flatten()
	jK = np.kron(edofMat,np.ones((1,8))).flatten()    
	# Filter: Build (and assemble) the index+data vectors for the coo matrix format
	nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
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
					cc=cc+1
	# Finalize assembly and convert to csc format
	H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
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
	# plt.ion() # Ensure that redrawing is possible
	# fig,ax = plt.subplots()
	# im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
	# interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
	# fig.show()
   	# Set loop counter and gradient vectors 
	loop=0
	change=1
	dv = np.ones(nely*nelx)
	dc = np.ones(nely*nelx)
	ce = np.ones(nely*nelx)
	while change>0.01 and loop<200:
		loop=loop+1
		# Setup and solve FE problem
		sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
		K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
		# Remove constrained dofs from matrix
		K = K[free,:][:,free]
		# Solve system 
		u[free,0]=spsolve(K,f[free,0])    
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
		# im.set_array(-xPhys.reshape((nelx,nely)).T)
		# fig.canvas.draw()
		# Write iteration history to screen (req. Python 2.6 or newer)
		print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
					loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
		objVal = "%.2f" % obj
		volVal = "%.2f" % ((g+volfrac*nelx*nely)/(nelx*nely))
		changeVal = "%.2f" % change
		myGlobalStr = "iteration : {it} , \t objective: {obj}, \t Volume fraction: {vol}, \t change in volume: {change}".format(it=loop,obj=objVal, vol=volVal, change=changeVal)
		print(myGlobalStr, file=fout, flush=True)
		# print("yoyoma", file=f, flush=True)
	
	return xPhys
	# Make sure the plot stays and that the shell remains	
	# plt.show()
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
# The real main driver    
if __name__ == "__main__":
	# # Default input parameters
	# nelx=180
	# nely=60
	# volfrac=0.4
	# rmin=5.4
	# penal=3.0
	# ft=1 # ft==0 -> sens, ft==1 -> dens
	# import sys
	# if len(sys.argv)>1: nelx   =int(sys.argv[1])
	# #if len(sys.argv)>2: nely   =int(sys.argv[2])
	# #if len(sys.argv)>3: volfrac=float(sys.argv[3])
	# #if len(sys.argv)>4: rmin   =float(sys.argv[4])
	# #if len(sys.argv)>5: penal  =float(sys.argv[5])
	# #if len(sys.argv)>6: ft     =int(sys.argv[6])
	# #main(nelx,nely,volfrac,penal,rmin,ft)
	f = open('filename.txt', 'w+')
	f.close()
	app.run_server(debug=True, port=8050)
