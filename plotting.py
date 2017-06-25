import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import set_credentials_file
from sklearn.manifold import TSNE


<<<<<<< HEAD
def gen_plotly_specs(datas, names, cat):
    data = []

    for name in np.unique(cat):

        i = np.where(cat == name)
        index = datas[i]

        # creating scatter plot for a topic
        trace = go.Scatter3d(
            x=index[:, 0],
            y=index[:, 1],
            z=index[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                line=dict(
                    width=0.0
                ),
                opacity=0.8
            ),
            name=name
        )

        data.append(trace)

    return data

=======
<<<<<<< HEAD
>>>>>>> master
def tsne_plotly(data, cat, labels, source, username, api_key, seed=0):
=======
def tsne_plotly(data, cat, labels, username, api_key, seed=0, max_points_per_category=250):
>>>>>>> 95d22955835727cff8baa328c6dacba8c98ac5b5
    print("Plotting data...")
    set_credentials_file(username=username, api_key=api_key)
<<<<<<< HEAD
    model = TSNE(n_components=3, random_state=seed, verbose=1)
=======
    # subsample points before tsne / plotting
    new_data = []
    new_cats = []
    for n in np.unique(cat):
        idx = np.where(cat == n)[0]
        idx = idx[:max_points_per_category]
        new_data.append(data[idx])
        new_cats.append(np.reshape(cat[idx], [cat[idx].size, 1]))
    data = np.vstack(new_data)
    cat = np.vstack(new_cats)
    cat = np.reshape(cat, (cat.size,))

>>>>>>> master
    # creating the model --- I am not sure if you can pickle TSNE need to run experiments. Had problems when i pickeled it

    model = TSNE(n_components=3, random_state=seed, verbose=1)
    reduced = model.fit_transform(data)
    plot_params = [[reduced, labels, cat, True], [reduced, labels, source, False]]

    # generating figures
    figures = []
    for i in plot_params:
        if i[3]:
            fname = 'topics-scatter.html'
        else:
            fname = 'source-scatter.html'

<<<<<<< HEAD
        fig = gen_plotly_specs(i[0], i[1], i[2])
        figures.append([fig, fname])
=======
        # creating scatter plot for a topic
<<<<<<< HEAD

=======
        trace = go.Scatter3d(
            x=index[:, 0],
            y=index[:, 1],
            z=index[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                line=dict(width=0.0),
                opacity=0.8
            ),
            name=labels[n]
        )
>>>>>>> 95d22955835727cff8baa328c6dacba8c98ac5b5
        data.append(trace)
>>>>>>> master

    # general figure layouts these are default values
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    for n in figures:
        plotly.offline.plot({
            "data":n[0],
            "layout":layout
        }, filename=n[1])

