import os
from scipy.stats import norm
from dash import Dash, html, dcc
import numpy as np
import pandas as pd
from dash import dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from statsmodels.nonparametric.kernel_regression import KernelReg

########## Load data - ASD ##########
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\ACD_results\ACD_measures')
df_dist_control_asd_vector = pd.read_csv('dist_control_ACD_vector.csv', header=None)
dist_control_asd_vector = df_dist_control_asd_vector.to_numpy()
dist_control_asd_vector = dist_control_asd_vector.flatten()

df_dist_control_control_vector_asd = pd.read_csv('dist_control_control_vector.csv', header=None)
dist_control_control_vector_asd = df_dist_control_control_vector_asd.to_numpy()
dist_control_control_vector_asd = dist_control_control_vector_asd.flatten()

df_idoa_control_asd_vector = pd.read_csv('idoa_control_ACD_vector.csv', header=None)
idoa_control_asd_vector = df_idoa_control_asd_vector.to_numpy()
idoa_control_asd_vector = idoa_control_asd_vector.flatten()

df_idoa_control_control_vector_asd = pd.read_csv('idoa_control_control_vector.csv', header=None)
idoa_control_control_vector_asd = df_idoa_control_control_vector_asd.to_numpy()
idoa_control_control_vector_asd = idoa_control_control_vector_asd.flatten()

df_DOC_control_asd = pd.read_csv('Doc_mat_control.csv', header=None)
DOC_control_asd = df_DOC_control_asd.to_numpy()
df_DOC_asd = pd.read_csv('Doc_mat_ACD.csv', header=None)
DOC_asd = df_DOC_asd.to_numpy()

########## Fit normal distribution to the ASD data ##########
mu_dist_control_asd_vector, std_dist_control_asd_vector = norm.fit(dist_control_asd_vector)
mu_dist_control_control_vector_asd, std_dist_control_control_vector_asd = norm.fit(dist_control_control_vector_asd)
mu_idoa_control_asd_vector, std_idoa_control_asd_vector = norm.fit(idoa_control_asd_vector)
mu_idoa_control_control_vector_asd, std_idoa_control_control_vector_asd = norm.fit(idoa_control_control_vector_asd)

########################################################################################################################
# Plot the PDF of the fitted normal distribution
app = Dash(__name__)
# Fitted distribution Control <--> CD distances.
#hist_dist_control_asd_vector = go.Histogram(x=dist_control_asd_vector, histnorm='probability density', nbinsx=50)
x_min_dist_control_asd_vector, x_max_dist_control_asd_vector = np.min(dist_control_asd_vector),\
    np.max(dist_control_asd_vector)
#x_dist_control_asd_vector = np.linspace(x_min_dist_control_asd_vector, x_max_dist_control_asd_vector, 100)
x_dist_control_asd_vector = np.linspace(mu_dist_control_asd_vector-(1-mu_dist_control_asd_vector)
                                       , mu_dist_control_asd_vector+(1-mu_dist_control_asd_vector), 100)
y_dist_control_asd_vector = norm.pdf(x_dist_control_asd_vector, mu_dist_control_asd_vector, std_dist_control_asd_vector)
fitted_curve_dist_control_asd_vector = go.Scatter(x=x_dist_control_asd_vector,
                                                 y=y_dist_control_asd_vector,
                                                 line=dict(color='blue', width=2),
                                                 name='Distances Control-ASD')

hist_dist_control_asd_vector = go.Histogram(x=dist_control_asd_vector,
                                           histnorm='probability density',
                                           marker=dict(color='blue'),
                                           opacity=0.5,
                                           name='Distances Control-ASD histogram',
                                           nbinsx=10)

# Fitted distribution Control <--> Control distances.
x_min_dist_control_control_vector_asd, x_max_dist_control_control_vector_asd = np.min(dist_control_control_vector_asd),\
    np.max(dist_control_control_vector_asd)
#x_dist_control_control_vector_asd = np.linspace(x_min_dist_control_control_vector_asd, x_max_dist_control_control_vector_asd, 100)
x_dist_control_control_vector_asd = np.linspace(mu_dist_control_control_vector_asd-(1-mu_dist_control_control_vector_asd),
                                            mu_dist_control_control_vector_asd+(1-mu_dist_control_control_vector_asd), 100)
y_dist_control_control_vector_asd = norm.pdf(x_dist_control_control_vector_asd,
                                         mu_dist_control_control_vector_asd, std_dist_control_control_vector_asd)
fitted_curve_dist_control_control_vector_asd = go.Scatter(x=x_dist_control_control_vector_asd,
                                                      y=y_dist_control_control_vector_asd,
                                                      line=dict(color='red', width=2),
                                                      name='Distances Control-Control')

hist_dist_control_control_vector_asd = go.Histogram(x=dist_control_control_vector_asd,
                                           histnorm='probability density',
                                           marker=dict(color='red'),
                                           opacity=0.5,
                                           name='Distances Control-control histogram',
                                           nbinsx=8)

# Fitted distribution Control <--> ASD IDOA.
x_min_idoa_control_asd_vector = np.min(idoa_control_asd_vector)
x_max_idoa_control_asd_vector = np.max(idoa_control_asd_vector)
#x_idoa_control_asd_vector = np.linspace(x_min_idoa_control_asd_vector, x_max_idoa_control_asd_vector, 100)
x_idoa_control_asd_vector = np.linspace(mu_idoa_control_asd_vector+1.25, mu_idoa_control_asd_vector-1.25, 100)
y_idoa_control_asd_vector = norm.pdf(x_idoa_control_asd_vector, mu_idoa_control_asd_vector, std_idoa_control_asd_vector)
fitted_curve_idoa_control_asd_vector = go.Scatter(x=x_idoa_control_asd_vector,
                                                 y=y_idoa_control_asd_vector,
                                                 line=dict(color='blue', width=2),
                                                 name='IDOA Control-ASD')

hist_idoa_control_asd_vector = go.Histogram(x=idoa_control_asd_vector,
                                           histnorm='probability density',
                                           marker=dict(color='blue'),
                                           opacity=0.5,
                                           name='IDOA Control-ASD histogram',
                                           nbinsx=10)

# Fitted distribution Control <--> Control IDOA.
x_min_idoa_control_control_vector_asd, x_max_idoa_control_control_vector_asd = np.min(idoa_control_control_vector_asd),\
    np.max(idoa_control_control_vector_asd)
#x_idoa_control_control_vector_asd = np.linspace(x_min_idoa_control_control_vector_asd, x_max_idoa_control_control_vector_asd, 100)
x_idoa_control_control_vector_asd = np.linspace(mu_idoa_control_control_vector_asd-1.25,
                                            mu_idoa_control_control_vector_asd+1.25, 100)
y_idoa_control_control_vector_asd = norm.pdf(x_idoa_control_control_vector_asd, mu_idoa_control_control_vector_asd,
                                         std_idoa_control_control_vector_asd)
fitted_curve_idoa_control_control_vector_asd = go.Scatter(x=x_idoa_control_control_vector_asd,
                                                      y=y_idoa_control_control_vector_asd,
                                                      line=dict(color='red', width=2),
                                                      name='IDOA Control-Control')

hist_idoa_control_control_vector_asd = go.Histogram(x=idoa_control_control_vector_asd,
                                               histnorm='probability density',
                                               marker=dict(color='red'),
                                               opacity=0.5,
                                               name='IDOA Control-control histogram',
                                               nbinsx=8)

scatter_DOC_control_asd = go.Scatter(x=DOC_control_asd[1][:],
                                     y=DOC_control_asd[0][:],
                                     marker={"color": "red", "size": 0.8},
                                     name='Control',
                                     mode="markers"
)

scatter_DOC_asd = go.Scatter(x=DOC_asd[1][:],
                             y=DOC_asd[0][:],
                             marker={"color": "blue", "size": 0.8},
                             name='Control',
                             mode="markers"
)

# Combination of IDOA and Distances.
scatter_asd = go.Scatter(x=dist_control_asd_vector,
                         y=idoa_control_asd_vector,
                         marker={"color": "blue"},
                         name='ASD',
                         mode="markers")

scatter_control_asd = go.Scatter(x=dist_control_control_vector_asd,
                             y=idoa_control_control_vector_asd,
                             marker={"color": "red"},
                             name='Control',
                             mode="markers",
                             opacity=0.8)

line_asd = go.Scattergl(
    x=[0.2, 0.9],
    y=[0, 0],
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary - IDOA'
)

line_horizontal_asd = go.Scattergl(
    x=[mu_dist_control_control_vector_asd, mu_dist_control_control_vector_asd],
    y=[-2, 2],
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary - Distances'
)

graph_layout_asd = go.Layout(
    xaxis={"title": 'Distance w.r.t control'},
    yaxis={"title": 'IDOA w.r.t control'},
    title='Combination IDOA - distances',
)
########################################################################################################################

########## Load data - CD ##########
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\CD_results\Distance_vectors')
df_dist_control_cd_vector = pd.read_csv('dist_control_cd_vector.csv', header=None)
dist_control_cd_vector = df_dist_control_cd_vector.to_numpy()
dist_control_cd_vector = dist_control_cd_vector.flatten()

df_dist_control_control_vector = pd.read_csv('dist_control_control_vector.csv', header=None)
dist_control_control_vector = df_dist_control_control_vector.to_numpy()
dist_control_control_vector = dist_control_control_vector.flatten()

df_idoa_control_cd_vector = pd.read_csv('idoa_control_cd_vector.csv', header=None)
idoa_control_cd_vector = df_idoa_control_cd_vector.to_numpy()
idoa_control_cd_vector = idoa_control_cd_vector.flatten()

df_idoa_control_control_vector = pd.read_csv('idoa_control_control_vector.csv', header=None)
idoa_control_control_vector = df_idoa_control_control_vector.to_numpy()
idoa_control_control_vector = idoa_control_control_vector.flatten()

df_DOC_control = pd.read_csv('Doc_mat.csv', header=None)
DOC_control = df_DOC_control.to_numpy()
df_DOC_cd = pd.read_csv('Doc_mat_cd.csv', header=None)
DOC_cd = df_DOC_cd.to_numpy()

# Fit the nonparametric regression model using the KernelReg class
kr = KernelReg(endog=DOC_control[0][:], exog=DOC_control[1][:], var_type='c')
#x_pred_control_cd = np.linspace(DOC_control[0][:].min(), DOC_control[1][:].max(), 1000)  # Generate a set of prediction
x_pred_control_cd = np.linspace(0.4, 0.9, 1000)  # Generate a set of prediction
y_pred_control_cd, _ = kr.fit(x_pred_control_cd)  # Predict the response at the prediction points

########## Fit normal distribution to the CD data ##########
mu_dist_control_cd_vector, std_dist_control_cd_vector = norm.fit(dist_control_cd_vector)
mu_dist_control_control_vector, std_dist_control_control_vector = norm.fit(dist_control_control_vector)
mu_idoa_control_cd_vector, std_idoa_control_cd_vector = norm.fit(idoa_control_cd_vector)
mu_idoa_control_control_vector, std_idoa_control_control_vector = norm.fit(idoa_control_control_vector)

########## Plots ##########
# Plot the PDF of the fitted normal distribution
#app = Dash(__name__)
# Fitted distribution Control <--> CD distances.
#hist_dist_control_cd_vector = go.Histogram(x=dist_control_cd_vector, histnorm='probability density', nbinsx=50)
x_min_dist_control_cd_vector, x_max_dist_control_cd_vector = np.min(dist_control_cd_vector),\
    np.max(dist_control_cd_vector)
#x_dist_control_cd_vector = np.linspace(x_min_dist_control_cd_vector, x_max_dist_control_cd_vector, 100)
x_dist_control_cd_vector = np.linspace(mu_dist_control_cd_vector-(1-mu_dist_control_cd_vector)
                                       , mu_dist_control_cd_vector+(1-mu_dist_control_cd_vector), 100)
y_dist_control_cd_vector = norm.pdf(x_dist_control_cd_vector, mu_dist_control_cd_vector, std_dist_control_cd_vector)
fitted_curve_dist_control_cd_vector = go.Scatter(x=x_dist_control_cd_vector,
                                                 y=y_dist_control_cd_vector,
                                                 line=dict(color='blue', width=2),
                                                 name='Distances Control-CD')

hist_dist_control_cd_vector = go.Histogram(x=dist_control_cd_vector,
                                           histnorm='probability density',
                                           marker=dict(color='blue'),
                                           opacity=0.5,
                                           name='Distances Control-CD histogram',
                                           nbinsx=10)

# Fitted distribution Control <--> Control distances.
x_min_dist_control_control_vector, x_max_dist_control_control_vector = np.min(dist_control_control_vector),\
    np.max(dist_control_control_vector)
#x_dist_control_control_vector = np.linspace(x_min_dist_control_control_vector, x_max_dist_control_control_vector, 100)
x_dist_control_control_vector = np.linspace(mu_dist_control_control_vector-(1-mu_dist_control_control_vector),
                                            mu_dist_control_control_vector+(1-mu_dist_control_control_vector), 100)
y_dist_control_control_vector = norm.pdf(x_dist_control_control_vector,
                                         mu_dist_control_control_vector, std_dist_control_control_vector)
fitted_curve_dist_control_control_vector = go.Scatter(x=x_dist_control_control_vector,
                                                      y=y_dist_control_control_vector,
                                                      line=dict(color='red', width=2),
                                                      name='Distances Control-Control')

hist_dist_control_control_vector = go.Histogram(x=dist_control_control_vector,
                                           histnorm='probability density',
                                           marker=dict(color='red'),
                                           opacity=0.5,
                                           name='Distances Control-control histogram',
                                           nbinsx=8)

# Fitted distribution Control <--> CD IDOA.
x_min_idoa_control_cd_vector = np.min(idoa_control_cd_vector)
x_max_idoa_control_cd_vector = np.max(idoa_control_cd_vector)
#x_idoa_control_cd_vector = np.linspace(x_min_idoa_control_cd_vector, x_max_idoa_control_cd_vector, 100)
x_idoa_control_cd_vector = np.linspace(mu_idoa_control_cd_vector+1.25, mu_idoa_control_cd_vector-1.25, 100)
y_idoa_control_cd_vector = norm.pdf(x_idoa_control_cd_vector, mu_idoa_control_cd_vector, std_idoa_control_cd_vector)
fitted_curve_idoa_control_cd_vector = go.Scatter(x=x_idoa_control_cd_vector,
                                                 y=y_idoa_control_cd_vector,
                                                 line=dict(color='blue', width=2),
                                                 name='IDOA Control-CD')

hist_idoa_control_cd_vector = go.Histogram(x=idoa_control_cd_vector,
                                           histnorm='probability density',
                                           marker=dict(color='blue'),
                                           opacity=0.5,
                                           name='IDOA Control-cd histogram',
                                           nbinsx=10)

# Fitted distribution Control <--> Control IDOA.
x_min_idoa_control_control_vector, x_max_idoa_control_control_vector = np.min(idoa_control_control_vector),\
    np.max(idoa_control_control_vector)
#x_idoa_control_control_vector = np.linspace(x_min_idoa_control_control_vector, x_max_idoa_control_control_vector, 100)
x_idoa_control_control_vector = np.linspace(mu_idoa_control_control_vector-1.25,
                                            mu_idoa_control_control_vector+1.25, 100)
y_idoa_control_control_vector = norm.pdf(x_idoa_control_control_vector, mu_idoa_control_control_vector,
                                         std_idoa_control_control_vector)
fitted_curve_idoa_control_control_vector = go.Scatter(x=x_idoa_control_control_vector,
                                                      y=y_idoa_control_control_vector,
                                                      line=dict(color='red', width=2),
                                                      name='IDOA Control-Control')

hist_idoa_control_control_vector = go.Histogram(x=idoa_control_control_vector,
                                           histnorm='probability density',
                                           marker=dict(color='red'),
                                           opacity=0.5,
                                           name='IDOA Control-control histogram',
                                           nbinsx=8)

scatter_DOC_control = go.Scatter(x=DOC_control[1][:],
                                 y=DOC_control[0][:],
                                 marker={"color": "red", "size": 2},
                                 name='Control',
                                 mode="markers"
)

######
nonparam_reg_control_cd = go.Scattergl(
           x=x_pred_control_cd,
           y=y_pred_control_cd,
           line={"color": "black", "dash": 'solid'},
)
######
scatter_DOC_cd = go.Scatter(x=DOC_cd[1][:],
                            y=DOC_cd[0][:],
                            marker={"color": "blue", "size": 0.5},
                            name='Control',
                            mode="markers"
)

# Combination of IDOA and Distances.
scatter_cd = go.Scatter(x=dist_control_cd_vector,
                        y=idoa_control_cd_vector,
                        marker={"color": "blue"},
                        name='CD',
                        mode="markers")

scatter_control = go.Scatter(x=dist_control_control_vector,
                             y=idoa_control_control_vector,
                             marker={"color": "red"},
                             name='Control',
                             mode="markers",
                             opacity=0.8)

line = go.Scattergl(
    x=[0.45, 1.17],
    y=[0, 0],
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary - IDOA'
)

line_horizontal = go.Scattergl(
    x=[mu_dist_control_control_vector, mu_dist_control_control_vector],
    y=[-2, 2],
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary - Distances'
)

graph_layout = go.Layout(
    xaxis={"title": 'Distance w.r.t control'},
    yaxis={"title": 'IDOA w.r.t control'},
    title='Combination IDOA - distances',
)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'IDOA and Distances methods - CD', 'value': 'CD_graphs'},
            {'label': 'IDOA and Distances methods - ASD', 'value': 'ASD_graphs'},
        ],
        value='IDOA and Distances methods - CD'
    ),
    html.Div(id='page-content')]),
])

@app.callback(Output('page-content', 'children'),
              [Input('dropdown', 'value')])
def display_page(value):
    if value == 'CD_graphs':
        return html.Div([
            html.H1(children='IDOA and Distances methods - CD'),
            html.P(children='This is a paragraph.'),
            html.Div([
                dcc.Graph(
                    id='Normal distribution fit distances',
                    figure={
                        'data': [fitted_curve_dist_control_cd_vector, fitted_curve_dist_control_control_vector,
                                 hist_dist_control_cd_vector, hist_dist_control_control_vector],
                        'layout': go.Layout(title='Fitted Normal Distributions for Distances.',
                                            xaxis={'title': 'Distance w.r.t control'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Normal distribution fit IDOA',
                    figure={
                        'data': [fitted_curve_idoa_control_cd_vector, fitted_curve_idoa_control_control_vector,
                                 hist_idoa_control_cd_vector, hist_idoa_control_control_vector],
                        'layout': go.Layout(title='Fitted Normal Distributions for IDOA.',
                                            xaxis={'title': 'IDOA w.r.t control'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Combination IDOA - distances',
                    figure={
                        'data': [line, line_horizontal, scatter_cd, scatter_control],
                        'layout': graph_layout
                    }
                )], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div([
                dcc.Graph(
                    id='DOC - Control',
                    figure={
                        'data': [scatter_DOC_control, nonparam_reg_control_cd],
                        'layout': go.Layout(title='DOC - Control',
                                            xaxis={'title': 'Overlap'},
                                            yaxis={'title': 'Dissimilarity'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='DOC - CD',
                    figure={
                        'data': [scatter_DOC_cd],
                        'layout': go.Layout(title='DOC - CD',
                                            xaxis={'title': 'Overlap'},
                                            yaxis={'title': 'Dissimilarity'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'})
        ])
    elif value == 'ASD_graphs':
        return html.Div([
            html.H1(children='IDOA and Distances methods - ASD'),
            html.P(children='This is a paragraph.'),
            html.Div([
                dcc.Graph(
                    id='Normal distribution fit distances',
                    figure={
                        'data': [fitted_curve_dist_control_asd_vector, fitted_curve_dist_control_control_vector_asd,
                                 hist_dist_control_asd_vector, hist_dist_control_control_vector_asd],
                        'layout': go.Layout(title='Fitted Normal Distributions for Distances.',
                                            xaxis={'title': 'Distance w.r.t control'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Normal distribution fit IDOA',
                    figure={
                        'data': [fitted_curve_idoa_control_asd_vector, fitted_curve_idoa_control_control_vector_asd,
                                 hist_idoa_control_asd_vector, hist_idoa_control_control_vector_asd],
                        'layout': go.Layout(title='Fitted Normal Distributions for IDOA.',
                                            xaxis={'title': 'IDOA w.r.t control'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Combination IDOA - distances',
                    figure={
                        'data': [line_asd, line_horizontal_asd, scatter_asd, scatter_control_asd],
                        'layout': graph_layout_asd
                    }
                )], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div([
                dcc.Graph(
                    id='DOC - Control',
                    figure={
                        'data': [scatter_DOC_control_asd],
                        'layout': go.Layout(title='DOC - Control',
                                            xaxis={'title': 'Overlap', 'range': [0.95, 1]},
                                            yaxis={'title': 'Dissimilarity', 'range': [0, 0.8]})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='DOC - ASD',
                    figure={
                        'data': [scatter_DOC_asd],
                        'layout': go.Layout(title='DOC - ASD',
                                            xaxis={'title': 'Overlap', 'range': [0.95, 1]},
                                            yaxis={'title': 'Dissimilarity', 'range': [0, 0.8]})
                    }
                )], style={'width': '48%', 'display': 'inline-block'})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)