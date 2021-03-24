# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from time import sleep
# MAIN DRIVER
import dash
#from jupyter_dash import JupyterDash
import dash_html_components as html
import dash_bootstrap_components as dbc
from plotly.graph_objects import Layout
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
									# html.Div(id='force-output-container'),
									html.Br(),
									html.Br(),
									html.Br(),
									html.Div([html.H1('Point force vector: ')], style={'display': 'inline-block', 'width': '100%'}),
                                    html.Div([html.H1('fx: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(placeholder=0.0, style={'width': 80, 'textAlign':'center'}, id='fx', min=-50, max=50, step=0.1, type='number')],style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([html.H1('fy: ')], style={'display': 'inline-block', 'width': '25%'}),
                                    html.Div([dcc.Input(placeholder=-1.0, style={'width': 80, 'textAlign':'center'}, id='fy', min=-50, max=50, step=0.1, type='number')],style={'display': 'inline-block', 'width': '25%'}),
									html.Br(),
									html.Br(),
									html.Br(),
									html.Div([html.H1('Fixed boundary: ')], style={'display': 'inline-block', 'width': '100%'}),
									dcc.Dropdown(id='bc-dropdown',
        											options=[
														{'label': 'Top', 'value': 't'},
														{'label': 'Bottom', 'value': 'b'},
														{'label': 'Right', 'value': 'r'},
														{'label': 'Left', 'value': 'l'}
													],
													value='l'
										),
									html.Div(id='bc-output-container'),
                                    html.Hr(),
                                    html.Div([html.Button('Run', id='run-data', n_clicks=0)])
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
																dbc.Row(dbc.Col(html.Div([dcc.Graph(id='timeseries-initial', config={'displayModeBar': False})])), style={'height': '25%'}),
																html.Br(),
																html.Br(),
																html.Br(),
																html.Br(),
																dbc.Row(dbc.Col(html.Div([dcc.Graph(id='timeseries', config={'displayModeBar': False})])), style={'height': '25%'}), 
																html.Br(),
																html.Br(),
																html.Br(),
																html.Br(),
																html.Br(),
																html.Br(),
																html.Br(),
																html.Br(),																																
							    							dbc.Row(dbc.Col(html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line', 'maxHeight': '50%', 'font-size': '20px'})), style={'height': '45%'}),
																dcc.Interval(id='my-interval', interval=1000)  # one tick each 5 seconds
                              ])
        ]

)])

@app.callback(Output('timeseries-initial', 'figure'),
              [Input('draw-data', 'n_clicks'), dash.dependencies.Input('bc-dropdown', 'value'), dash.dependencies.Input('force-dropdown', 'value')], 
              [State('nx', 'value'), State('ny', 'value'), State('fx', 'value'), State('fy', 'value')])
def update_plot(nclicks, bcloc, floc, nx, ny, fx, fy):

	if nclicks == 0:
		layout = Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
		fig = go.Figure(go.Scatter(x=[0,0,0,0,0], y=[0,0,0,0,0], fill="toself"), layout=layout)
		fig.update_xaxes(showgrid=False, zeroline=False)
		fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor = "x", scaleratio = 1,)
		fig.update_layout(
				title="Domain",
    		legend_title="Optimal topology",
    		font=dict(family="Courier New, monospace", size=18, color="White")
			)

		return fig


	layout = Layout(    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')
	fig = go.Figure(go.Scatter(x=[0,0,nx,nx,0], y=[0,ny,ny,0,0], fill="toself"), layout=layout)
	fig.update_xaxes(showgrid=False, zeroline=False)
	fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor = "x", scaleratio = 1,)
	fig.update_layout(
    title="Domain",
    legend_title="Optimal topology",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="White"
    		)
		)


	if fx == 0:
		ay = ny + np.sign(fy)*5
		# ax=nx
	else:
		theta = math.atan2(fy, fx)
		ay = ny-10*math.sin(theta)
		# ax = nx

	theta = math.atan2(fy, fx)
	if floc == 'tr' or floc == 'br':
		ax = nx-10*math.cos(theta)
	else:
		ax = -10*math.cos(theta)
	# ay = ny

	if floc == 'tr':
		xloc=nx
		yloc=ny	
	elif floc == 'tl':
		xloc=0
		yloc=ny	
	elif floc == 'br':
		xloc=nx
		yloc=0	
	elif floc == 'bl':
		xloc=0
		yloc=0	

	fig.add_annotation(
  x=xloc,  # arrows' head
  y=yloc,  # arrows' head
  ax=ax,  # arrows' tail
  ay=ay,  # arrows' tail
  xref='x',
  yref='y',
  axref='x',
  ayref='y',
  text='',  # if you want only the arrow
  showarrow=True,
  arrowhead=3,
  arrowsize=1,
  arrowwidth=2,
  arrowcolor='red'
	)

	if bcloc == 'l':
		fig.add_shape(type='line', x0=0, y0=0, x1=0, y1=ny+1, line=dict(color='Red',), xref='x', yref='y')
	elif bcloc == 'r':
		fig.add_shape(type='line', x0=nx+1, y0=0, x1=nx+1, y1=ny+1, line=dict(color='Red',), xref='x', yref='y')
	elif bcloc == 'b':
		fig.add_shape(type='line', x0=0, y0=0, x1=nx+1, y1=0, line=dict(color='Red',), xref='x', yref='y')
	elif bcloc == 't':
		fig.add_shape(type='line', x0=0, y0=ny+1, x1=nx+1, y1=ny+1, line=dict(color='Red',), xref='x', yref='y')

	return fig


@app.callback(
    Output('textarea-example-output', 'children'),
    Input('my-interval', 'n_intervals')
)
def callback_func(interval):
	with open('filename.txt', 'r') as f:
		val = f.readlines()
	return html.Div(val)

@app.callback(Output('timeseries', 'figure'),
              [Input('run-data', 'n_clicks'), dash.dependencies.Input('force-dropdown', 'value'), dash.dependencies.Input('bc-dropdown', 'value')], 
              [State('nx', 'value'), State('ny', 'value'), State('fx', 'value'), State('fy', 'value')])
def update_output_run(nclicks, floc, bcloc, nx, ny, fx, fy):

	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = 'No clicks yet'
	else:
		button_id = ctx.triggered[0]['prop_id'].split('.')[0]

	if (nclicks or 0) and (button_id == "run-data"):
		nelx=nx
		nely=ny
		volfrac=0.4
		rmin=5.4
		penal=3.0
		ft=1 # ft==0 -> sens, ft==1 -> dens
		f = open('filename.txt', 'w+')
		xPhys = main(nelx,nely,volfrac,penal,rmin,ft,floc,fx,fy,bcloc,f)
		f.close()
		fig = px.imshow(xPhys.reshape((nelx, nely)).T)
		# fig.update_layout(
		return fig

	else:

		layout = Layout(    paper_bgcolor='rgba(0,0,0,0)',
    										plot_bgcolor='rgba(0,0,0,0)')

		fig = go.Figure(go.Scatter(x=[0,0,0,0,0], y=[0,0,0,0,0], fill="toself"), layout=layout)
		fig.update_xaxes(showgrid=False, zeroline=False)
		fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor = "x", scaleratio = 1,)
		fig.update_layout(
    title="Plot Title",
    legend_title="Optimal topology",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="White"
    		)
		)

		return fig

def main(nelx,nely,volfrac,penal,rmin,ft,floc,fx,fy,bcloc,fout):
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
	# fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
	# fixed=dofs[0:2*(nely+1):2]
	# print("Original fixed = ", fixed)
	fixed=[]
	if bcloc == 'l':
		for i in range(nely+1):
			ind = 2*i
			fixed.append(dofs[ind])
			fixed.append(dofs[ind+1])
	elif bcloc == 'r':
		for i in range(nely+1):
			ind = 2*i + 2*(nelx)*(nely+1)
			fixed.append(dofs[ind])
			fixed.append(dofs[ind+1])
	elif bcloc == 't':
		for i in range(nelx+1):
			ind = 2*i*(nely+1) 
			fixed.append(dofs[ind])
			fixed.append(dofs[ind+1])
	elif bcloc == 'b':
		for i in range(nelx+1):
			ind = (nely+1) + 2*i*(nely+1) 
			fixed.append(dofs[ind])
			fixed.append(dofs[ind+1])
	else:
		fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
	print("new fixed = ", fixed)

	free=np.setdiff1d(dofs,fixed)
	# Solution and RHS vectors
	f=np.zeros((ndof,1))
	u=np.zeros((ndof,1))
	# Set load
	if floc == 'tr':
		f[2*(nelx)*(nely+1),0]=fx # tr
		f[2*(nelx)*(nely+1)+1,0]=fy # tr
	elif floc == 'br':
		f[2*(nelx+1)*(nely+1)-2,0]=fx # br
		f[2*(nelx+1)*(nely+1)-1,0]=fy # br
	elif floc == 'tl':
		f[0,0]=fx # tl
		f[1,0]=fy # tl
	elif floc == 'bl':
		f[2*(nely+1)-2,0]=fx # bl
		f[2*(nely+1)-1,0]=fy # bl
	else:
		f[1,0]=-1 # tl
	# Initialize plot and plot the initial design
  # Set loop counter and gradient vectors 
	loop=0
	change=1
	dv = np.ones(nely*nelx)
	dc = np.ones(nely*nelx)
	ce = np.ones(nely*nelx)
	TotalLoops = 100
	nPrint = 20
	PrintFreq = int(TotalLoops/nPrint)
	print("PrintFreq = ", PrintFreq)
	while change>0.001 and loop<TotalLoops:
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
		# Write iteration history to screen (req. Python 2.6 or newer)
		print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
					loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
		objVal = "%.2f" % obj
		volVal = "%.2f" % ((g+volfrac*nelx*nely)/(nelx*nely))
		changeVal = "%.2f" % change
		myGlobalStr = "iteration : {it} , \t objective: {obj}, \t Volume fraction: {vol}, \t change in volume: {change}".format(it=loop,obj=objVal, vol=volVal, change=changeVal)
		if loop % PrintFreq == 0:
			print(myGlobalStr, file=fout, flush=True)
	
	return xPhys

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
	f = open('filename.txt', 'w+')
	f.close()
	app.run_server(debug=False, port=8050)
