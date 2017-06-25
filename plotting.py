import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.tools import set_credentials_file
from sklearn.manifold import TSNE


def gen_plotly_specs(datas, hover_text, cat):
    data = []

    for cur_cat in np.unique(cat):

        idx = np.where(cat == cur_cat)[0]
        cur_pts = datas[idx]

        # creating scatter plot for a topic
        trace = go.Scatter3d(
            x=cur_pts[:, 0],
            y=cur_pts[:, 1],
            z=cur_pts[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                line=dict(width=0.0),
                opacity=0.8
            ),
            name=cur_cat,
            text=hover_text[idx]
        )

        data.append(trace)

    return data


def tsne_plotly(data, cat, labels, source, username, api_key, seed=0,
                max_points_per_category=250, max_label_length=64):
    print("Plotting data...")
    set_credentials_file(username=username, api_key=api_key)
    model = TSNE(n_components=3, random_state=seed, verbose=1)
    reduced = model.fit_transform(data)

    # subsample points before tsne / plotting
    if False:
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
    labels = np.asarray([lbl[:max_label_length] for lbl in labels])
    plot_params = [
        [reduced, source, labels[cat], 'topics-scatter.html'],
        [reduced, labels[cat], source, 'source-scatter.html']]

    # general figure layouts these are default values
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    # generating figures
    figures = []
    for data, hover_text, cats, fname in plot_params:
        fig = gen_plotly_specs(datas=data, hover_text=hover_text, cat=cats)

        plotly.offline.plot({
            "data": fig,
            "layout": layout
        }, filename=fname)

        figures.append([fig, fname])
