import os
from scipy.stats import norm
from dash import Dash, html, dcc
import numpy as np
import pandas as pd
from dash import dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output

########## Load data - ACD ##########
os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\ACD_results\ACD_measures')
df_dist_control_acd_vector = pd.read_csv('dist_control_ACD_vector.csv', header=None)
dist_control_acd_vector = df_dist_control_acd_vector.to_numpy()
dist_control_acd_vector = dist_control_acd_vector.flatten()

df_dist_control_control_vector_acd = pd.read_csv('dist_control_control_vector.csv', header=None)
dist_control_control_vector_acd = df_dist_control_control_vector_acd.to_numpy()
dist_control_control_vector_acd = dist_control_control_vector_acd.flatten()

df_idoa_control_acd_vector = pd.read_csv('idoa_control_ACD_vector.csv', header=None)
idoa_control_acd_vector = df_idoa_control_acd_vector.to_numpy()
idoa_control_acd_vector = idoa_control_acd_vector.flatten()

df_idoa_control_control_vector_acd = pd.read_csv('idoa_control_control_vector.csv', header=None)
idoa_control_control_vector_acd = df_idoa_control_control_vector_acd.to_numpy()
idoa_control_control_vector_acd = idoa_control_control_vector_acd.flatten()

df_DOC_control_acd = pd.read_csv('Doc_mat_control.csv', header=None)
DOC_control_acd = df_DOC_control_acd.to_numpy()
df_DOC_acd = pd.read_csv('Doc_mat_ACD.csv', header=None)
DOC_acd = df_DOC_acd.to_numpy()

########## Fit normal distribution to the ACD data ##########
mu_dist_control_acd_vector, std_dist_control_acd_vector = norm.fit(dist_control_acd_vector)
mu_dist_control_control_vector_acd, std_dist_control_control_vector_acd = norm.fit(dist_control_control_vector_acd)
mu_idoa_control_acd_vector, std_idoa_control_acd_vector = norm.fit(idoa_control_acd_vector)
mu_idoa_control_control_vector_acd, std_idoa_control_control_vector_acd = norm.fit(idoa_control_control_vector_acd)

########################################################################################################################
# Plot the PDF of the fitted normal distribution
app = Dash(__name__)
# Fitted distribution Control <--> CD distances.
#hist_dist_control_acd_vector = go.Histogram(x=dist_control_acd_vector, histnorm='probability density', nbinsx=50)
x_min_dist_control_acd_vector, x_max_dist_control_acd_vector = np.min(dist_control_acd_vector),\
    np.max(dist_control_acd_vector)
#x_dist_control_acd_vector = np.linspace(x_min_dist_control_acd_vector, x_max_dist_control_acd_vector, 100)
x_dist_control_acd_vector = np.linspace(mu_dist_control_acd_vector-(1-mu_dist_control_acd_vector)
                                       , mu_dist_control_acd_vector+(1-mu_dist_control_acd_vector), 100)
y_dist_control_acd_vector = norm.pdf(x_dist_control_acd_vector, mu_dist_control_acd_vector, std_dist_control_acd_vector)
fitted_curve_dist_control_acd_vector = go.Scatter(x=x_dist_control_acd_vector,
                                                 y=y_dist_control_acd_vector,
                                                 line=dict(color='blue', width=2),
                                                 name='Distances Control-ACD')

hist_dist_control_acd_vector = go.Histogram(x=dist_control_acd_vector,
                                           histnorm='probability density',
                                           marker=dict(color='blue'),
                                           opacity=0.5,
                                           name='Distances Control-ACD histogram',
                                           nbinsx=10)

# Fitted distribution Control <--> Control distances.
x_min_dist_control_control_vector_acd, x_max_dist_control_control_vector_acd = np.min(dist_control_control_vector_acd),\
    np.max(dist_control_control_vector_acd)
#x_dist_control_control_vector_acd = np.linspace(x_min_dist_control_control_vector_acd, x_max_dist_control_control_vector_acd, 100)
x_dist_control_control_vector_acd = np.linspace(mu_dist_control_control_vector_acd-(1-mu_dist_control_control_vector_acd),
                                            mu_dist_control_control_vector_acd+(1-mu_dist_control_control_vector_acd), 100)
y_dist_control_control_vector_acd = norm.pdf(x_dist_control_control_vector_acd,
                                         mu_dist_control_control_vector_acd, std_dist_control_control_vector_acd)
fitted_curve_dist_control_control_vector_acd = go.Scatter(x=x_dist_control_control_vector_acd,
                                                      y=y_dist_control_control_vector_acd,
                                                      line=dict(color='red', width=2),
                                                      name='Distances Control-Control')

hist_dist_control_control_vector_acd = go.Histogram(x=dist_control_control_vector_acd,
                                           histnorm='probability density',
                                           marker=dict(color='red'),
                                           opacity=0.5,
                                           name='Distances Control-control histogram',
                                           nbinsx=8)

# Fitted distribution Control <--> ACD IDOA.
x_min_idoa_control_acd_vector = np.min(idoa_control_acd_vector)
x_max_idoa_control_acd_vector = np.max(idoa_control_acd_vector)
#x_idoa_control_acd_vector = np.linspace(x_min_idoa_control_acd_vector, x_max_idoa_control_acd_vector, 100)
x_idoa_control_acd_vector = np.linspace(mu_idoa_control_acd_vector+1.25, mu_idoa_control_acd_vector-1.25, 100)
y_idoa_control_acd_vector = norm.pdf(x_idoa_control_acd_vector, mu_idoa_control_acd_vector, std_idoa_control_acd_vector)
fitted_curve_idoa_control_acd_vector = go.Scatter(x=x_idoa_control_acd_vector,
                                                 y=y_idoa_control_acd_vector,
                                                 line=dict(color='blue', width=2),
                                                 name='IDOA Control-ACD')

hist_idoa_control_acd_vector = go.Histogram(x=idoa_control_acd_vector,
                                           histnorm='probability density',
                                           marker=dict(color='blue'),
                                           opacity=0.5,
                                           name='IDOA Control-ACD histogram',
                                           nbinsx=10)

# Fitted distribution Control <--> Control IDOA.
x_min_idoa_control_control_vector_acd, x_max_idoa_control_control_vector_acd = np.min(idoa_control_control_vector_acd),\
    np.max(idoa_control_control_vector_acd)
#x_idoa_control_control_vector_acd = np.linspace(x_min_idoa_control_control_vector_acd, x_max_idoa_control_control_vector_acd, 100)
x_idoa_control_control_vector_acd = np.linspace(mu_idoa_control_control_vector_acd-1.25,
                                            mu_idoa_control_control_vector_acd+1.25, 100)
y_idoa_control_control_vector_acd = norm.pdf(x_idoa_control_control_vector_acd, mu_idoa_control_control_vector_acd,
                                         std_idoa_control_control_vector_acd)
fitted_curve_idoa_control_control_vector_acd = go.Scatter(x=x_idoa_control_control_vector_acd,
                                                      y=y_idoa_control_control_vector_acd,
                                                      line=dict(color='red', width=2),
                                                      name='IDOA Control-Control')

hist_idoa_control_control_vector_acd = go.Histogram(x=idoa_control_control_vector_acd,
                                               histnorm='probability density',
                                               marker=dict(color='red'),
                                               opacity=0.5,
                                               name='IDOA Control-control histogram',
                                               nbinsx=8)

scatter_DOC_control_acd = go.Scatter(x=DOC_control_acd[1][:],
                                     y=DOC_control_acd[0][:],
                                     marker={"color": "red", "size": 0.8},
                                     name='Control',
                                     mode="markers"
)

scatter_DOC_acd = go.Scatter(x=DOC_acd[1][:],
                             y=DOC_acd[0][:],
                             marker={"color": "blue", "size": 0.8},
                             name='Control',
                             mode="markers"
)

# Combination of IDOA and Distances.
scatter_acd = go.Scatter(x=dist_control_acd_vector,
                         y=idoa_control_acd_vector,
                         marker={"color": "blue"},
                         name='ACD',
                         mode="markers")

scatter_control_acd = go.Scatter(x=dist_control_control_vector_acd,
                             y=idoa_control_control_vector_acd,
                             marker={"color": "red"},
                             name='Control',
                             mode="markers",
                             opacity=0.8)

line_acd = go.Scattergl(
    x=[0.2, 0.9],
    y=[0, 0],
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary - IDOA'
)

line_horizontal_acd = go.Scattergl(
    x=[mu_dist_control_control_vector_acd, mu_dist_control_control_vector_acd],
    y=[-2, 2],
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary - Distances'
)

graph_layout_acd = go.Layout(
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
            {'label': 'IDOA and Distances methods - ACD', 'value': 'ACD_graphs'},
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
                        'data': [scatter_DOC_control],
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
    elif value == 'ACD_graphs':
        return html.Div([
            html.H1(children='IDOA and Distances methods - ACD'),
            html.P(children='This is a paragraph.'),
            html.Div([
                dcc.Graph(
                    id='Normal distribution fit distances',
                    figure={
                        'data': [fitted_curve_dist_control_acd_vector, fitted_curve_dist_control_control_vector_acd,
                                 hist_dist_control_acd_vector, hist_dist_control_control_vector_acd],
                        'layout': go.Layout(title='Fitted Normal Distributions for Distances.',
                                            xaxis={'title': 'Distance w.r.t control'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Normal distribution fit IDOA',
                    figure={
                        'data': [fitted_curve_idoa_control_acd_vector, fitted_curve_idoa_control_control_vector_acd,
                                 hist_idoa_control_acd_vector, hist_idoa_control_control_vector_acd],
                        'layout': go.Layout(title='Fitted Normal Distributions for IDOA.',
                                            xaxis={'title': 'IDOA w.r.t control'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='Combination IDOA - distances',
                    figure={
                        'data': [line_acd, line_horizontal_acd, scatter_acd, scatter_control_acd],
                        'layout': graph_layout_acd
                    }
                )], style={'display': 'flex', 'flexDirection': 'row'}),
            html.Div([
                dcc.Graph(
                    id='DOC - Control',
                    figure={
                        'data': [scatter_DOC_control_acd],
                        'layout': go.Layout(title='DOC - Control',
                                            xaxis={'title': 'Overlap'},
                                            yaxis={'title': 'Dissimilarity'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='DOC - ACD',
                    figure={
                        'data': [scatter_DOC_acd],
                        'layout': go.Layout(title='DOC - ACD',
                                            xaxis={'title': 'Overlap'},
                                            yaxis={'title': 'Dissimilarity'})
                    }
                )], style={'width': '48%', 'display': 'inline-block'})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)