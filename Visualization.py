from dash import Dash, html, dcc
import plotly.express as px
from plotly.io import to_json
import pandas as pd
import dash
from dash import dcc
import plotly.graph_objects as go
import functions as fun
import numpy as np
import matplotlib.pyplot as plt
from functions import normalize_data
from data_filter import DataFilter
from functions import calc_bray_curtis_dissimilarity, create_PCoA, plot_confusion_matrix
from IDOA_class import IDOA
import os
from functions import Confusion_matrix, Confusion_matrix_comb
from scipy.spatial.distance import cdist

os.chdir(r'C:\Users\shaya\OneDrive\Desktop\IDOA\CD data')

# Filter the data.
total_data = DataFilter('CD_data.xlsx', 'Control_(CD).xlsx', two_data_sets=True)
total_data.remove_less_one_data()
total_data.split_two_sets()
cd_data = total_data.first_set
control_data = total_data.second_set

cd_data = cd_data.to_numpy()
control_data = control_data.to_numpy()

# Normalization of the data.
cd_data = normalize_data(cd_data)
control_data = normalize_data(control_data)

########## IDOA ##########
idoa_cd_cd = IDOA(cd_data.T, cd_data.T, self_cohort=True)
idoa_cd_cd_vector = idoa_cd_cd.calc_idoa_vector()
idoa_control_control = IDOA(control_data.T, control_data.T, self_cohort=True)
idoa_control_control_vector = idoa_control_control.calc_idoa_vector()
idoa_cd_control = IDOA(cd_data.T, control_data.T)
idoa_cd_control_vector = idoa_cd_control.calc_idoa_vector()
idoa_control_cd = IDOA(control_data.T, cd_data.T)
idoa_control_cd_vector = idoa_control_cd.calc_idoa_vector()

x1 = idoa_control_cd_vector
y1 = idoa_cd_cd_vector

x2 = idoa_control_control_vector
y2 = idoa_cd_control_vector

x3 = [-3, 3]
y3 = [-3, 3]

########## Distances ##########
dist_cd_control_vector = calc_bray_curtis_dissimilarity(cd_data.T, control_data.T, median=False)
dist_control_cd_vector = calc_bray_curtis_dissimilarity(control_data.T, cd_data.T, median=False)
dist_control_control_vector = calc_bray_curtis_dissimilarity(control_data.T, control_data.T, self_cohort=True,
                                                             median=False)
dist_cd_cd_vector = calc_bray_curtis_dissimilarity(cd_data.T, cd_data.T, self_cohort=True, median=False)

x1_dist = dist_control_cd_vector
y1_dist = dist_cd_cd_vector

x2_dist = dist_control_control_vector
y2_dist = dist_cd_control_vector

x3_dist = [-3, 3]
y3_dist = [-3, 3]

########## Confusion matrices ##########
#con_mat_distances, y_exp_dist, y_pred_dist = Confusion_matrix(
#    dist_cd_control_vector, dist_control_control_vector, dist_control_cd_vector, dist_cd_cd_vector)
#con_mat_IDOA, y_exp_IDOA, y_pred_IDOA = Confusion_matrix(
#    idoa_cd_control_vector, idoa_control_control_vector, idoa_control_cd_vector, idoa_cd_cd_vector)
#[fig, ax] = plot_confusion_matrix(con_mat_distances, 'Confusion matrix - distances', labels=('NCD', 'CD'))
#[fig0, ax0] = plot_confusion_matrix(con_mat_IDOA, 'Confusion matrix - IDOA', labels=('NCD', 'CD'))

#plot_conf_dist = to_json(fig)
#plot_conf_idoa = to_json(fig0)

########## PCoA graph ##########
#combined_data = np.concatenate((cd_data.T, control_data.T), axis=0)
#dist_mat = cdist(combined_data, combined_data, 'braycurtis')
#[fig, ax] = create_PCoA(dist_mat, np.size(cd_data, axis=1), 'PCoA graph', 'CD', 'Control')

#plot_pcoa = to_json(fig)
app = Dash(__name__)
scatter_plot_1_dist = go.Scatter(
    x=x1_dist,
    y=y1_dist,
    marker={"color": "blue"},
    name='CD',
    mode="markers"
)

scatter_plot_2_dist = go.Scatter(
    x=x2_dist,
    y=y2_dist,
    marker={"color": "red"},
    name='Control',
    mode="markers"
)

line_plot_dist = go.Scattergl(
    x=x3_dist,
    y=y3_dist,
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary'
)

layout_dist = go.Layout(
    xaxis={"title": 'Mean distance to control cohort', "range": [0.6, 1]},
    yaxis={"title": 'Mean distance to CD cohort', "range": [0.6, 1]},
    scene={'aspectmode': 'cube'}
)


scatter_plot_1 = go.Scatter(
    x=x1,
    y=y1,
    marker={"color": "blue"},
    name='CD',
    mode="markers"
)

scatter_plot_2 = go.Scatter(
    x=x2,
    y=y2,
    marker={"color": "red"},
    name='Control',
    mode="markers"
)

line_plot = go.Scattergl(
    x=x3,
    y=y3,
    line={"color": "black", "dash": 'dash'},
    name='Decision boundary'
)

layout = go.Layout(
    xaxis={"title": 'IDOA w.r.t control', "range": [-1.2, 1]},
    yaxis={"title": 'IDOA w.r.t CD', "range": [-1.2, 1]},
    scene={'aspectmode': 'cube'}
)

app.layout = html.Div([
    html.Div([
    dcc.Graph(
        id="scatter-plot",
        figure={
            "data": [scatter_plot_1, scatter_plot_2, line_plot],
            "layout": layout,
        },
    ),
    dcc.Graph(
        id="scatter-plot_dist",
        figure={
            "data": [scatter_plot_1_dist, scatter_plot_2_dist, line_plot_dist],
            "layout": layout_dist,
        },
    ),
    #dcc.Graph(
    #    id='plot',
    #    figure=plot_pcoa
    #)
    ], style={'display': 'flex', 'flexDirection': 'row'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
