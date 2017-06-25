import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import set_credentials_file
from sklearn.manifold import TSNE


def tsne_plotly(data, cat, labels):

    # creating the model --- I am not sure if you can pickle TSNE need to run experiments. Had problems when i pickeled it
    set_credentials_file(username='bakeralex664', api_key='hWwBstLnNCX5CsDZpOSU')
    model = TSNE(n_components=3, random_state=0)
    reduced = model.fit_transform(data)

    # scratter plot info for topics
    data = []

    for n in np.unique(cat):
        i = np.where(cat == n)
        index = reduced[i]

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
            name=labels[n]
        )
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
