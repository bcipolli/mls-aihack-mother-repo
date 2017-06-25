import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import set_credentials_file
from sklearn.manifold import TSNE


<<<<<<< HEAD
def tsne_plotly(data, cat, labels, source, username, api_key, seed=0):
=======
def tsne_plotly(data, cat, labels, username, api_key, seed=0, max_points_per_category=250):
>>>>>>> 95d22955835727cff8baa328c6dacba8c98ac5b5
    print("Plotting data...")
    set_credentials_file(username=username, api_key=api_key)
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

    # creating the model --- I am not sure if you can pickle TSNE need to run experiments. Had problems when i pickeled it
    model = TSNE(n_components=3, random_state=seed, verbose=1)
    reduced = model.fit_transform(data)

    # scratter plot info for topics
    data = []

    for n in np.unique(cat):
        i = np.where(cat == n)
        index = reduced[i]

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

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')


# def gen_plotly_specs(data, name):
#     trace = go.Scatter3d(
#             x=data[:, 0],
#             y=data[:, 1],
#             z=data[:, 2],
#             mode='markers',
#             marker=dict(
#                 size=12,
#                 line=dict(
#                     width=0.0
#                 ),
#                 opacity=0.8
#             ),
#             name=name
#         )

    return trace

if __name__ == '__main__':
    # need to go into the main py
    # set_credentials_file(username='bakeralex664', api_key='hWwBstLnNCX5CsDZpOSU')
    data = np.random.rand(50, 20)
    cat = np.arange(1, 10)
    print cat.shape
    # lda_labels = ['poltics', 'sports']
    # topics = np.arange(20)
    # model = TSNE(n_components=3, random_state=0)
    # tsne(data, ['one', 'two', 'three'], model)

    # To make your color choice reproducible, uncomment the following line:
    # random.seed(10)

    # colors = []
    #
    # for i in range(0, 10):
    #     colors.append(generate_new_color(colors, pastel_factor=0.9))
    #
    # print 'Your colors:',colors
