import os.path as op
import pickle as pkl
import re

import numpy as np
import pandas as pd

from nlp_demo import do_lda, do_lemmatize, do_vectorize
from plotting import tsne_plotly
from registry_data import fetch_event_articles


def get_srcs(df):
    srcs = []
    for source in df.source:
        try:
            title = eval(source)['title']
        except:
            continue
        srcs.append(title)
    return srcs


def get_src(source):
    try:
        title = eval(source)['title']
    except:
        return
    return title


def clean_text(txt):
    txt = re.sub(r'[^\sa-zA-Z]', '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt


def reduce_by_source(df, thresh=0.75):
    df['org_title'] = df['source'].map(get_src)
    n_events = len(df['eventUri'].unique())

    total_srcs = get_srcs(df)
    all_srcs = []
    for group in df.groupby(by='eventUri'):
        srcs = get_srcs(group[1])
        all_srcs.append(srcs)
    srcs_dict = {src: 0 for src in total_srcs}
    for group in df.groupby(by='eventUri'):
        event_srcs = list(set(get_srcs(group[1])))
        for src in event_srcs:
            srcs_dict[src] += 1
    srcs_with_cover = {key: value for key, value in srcs_dict.items()
                       if value > n_events * thresh}
    print('total sources: {}\nsource with cover (>{}): {}'.format(
        len(np.unique(total_srcs)), thresh, len(srcs_with_cover)))

    cover_articles = {}
    dfs = []
    for event_uri, df in df.groupby(by='eventUri'):
        cover_articles[event_uri] = {}
        for src in srcs_with_cover.keys():
            dfs.append(df.loc[df.org_title == src, :])
    article_df = pd.concat(dfs)

    return article_df


def load_and_clean(csv_file, n_events=2, min_article_length=250, source_thresh=0.75, force=False,
                   eventregistry_api_key=None, min_articles=500):
    print("Loading and cleaning text...")

    clean_csv_file = csv_file.replace('.csv', '-ev%s-minlen%d-thresh%.3f.clean.csv' % (
        n_events if n_events < np.inf else 'All', min_article_length, source_thresh))
    if not force and op.exists(clean_csv_file):
        df = pd.read_csv(clean_csv_file)
        event_uris = df['eventUri'].unique()
        return df, event_uris

    # Need to download data.
    fetch_event_articles(
        api_key=eventregistry_api_key, min_articles=min_articles,
        csv_file=csv_file)

    df = pd.read_csv(csv_file)
    df['body'] = df['body'].apply(clean_text)
    event_uris = df['eventUri'].unique()

    # Reduce by article length
    good_idx = df['body'].apply(
        lambda b: not pd.isnull(b) and len(b) > min_article_length)
    df = df[good_idx]

    # Reduce by source.
    df = reduce_by_source(df, thresh=source_thresh)

    # Reduce by # events
    if n_events and n_events < np.inf:
        keep_event_uris = event_uris[:n_events]
        new_df = df[df['eventUri'].apply(lambda uri: uri in keep_event_uris)]

        assert n_events == len(new_df['eventUri'].unique())
        print("\tReduced from %d to %d events, %d to %d articles." % (
            len(event_uris), len(keep_event_uris), len(df), len(new_df)))
        df = new_df
        event_uris = keep_event_uris

    df.to_csv(clean_csv_file, encoding='utf-8')

    return df, event_uris


def vectorize_articles(df, event_uris, pkl_file, min_vocab_length=100,
                       lda_min_appearances=2, lda_vectorization_type='count', force=False):
    print("Vectorizing data...")
    if not force and op.exists(pkl_file):
        with open(pkl_file, 'rb') as fp:
            article_counts, vocab, vectorizer = pkl.load(fp)
    else:
        docs = df['body'].values
        l_docs = do_lemmatize(docs)
        article_counts, vocab, vectorizer = do_vectorize(
            l_docs, min_df=lda_min_appearances, type=lda_vectorization_type)

        with open(pkl_file, 'wb') as fp:
            pkl.dump((article_counts, vocab, vectorizer), fp)

    # Limit by vocab
    good_article_idx = np.squeeze(np.asarray((article_counts > 0).sum(axis=1) >= min_vocab_length))
    article_counts = article_counts[good_article_idx]
    good_vocab_idx = np.squeeze(np.asarray(article_counts.sum(axis=0) > 0))
    article_counts = article_counts[:, good_vocab_idx]
    vocab = vocab[good_vocab_idx]
    df = df.iloc[good_article_idx]

    return article_counts, vocab, vectorizer, df


def group_article_counts_by_source(df, article_counts):
    # Inputs:
    # df - dataframe; one row per article. 'source' column defines the news source.
    # article_counts - matrix; one row per article, one column per vocabulary item, value is the # of times
    #  that word appears in the article.
    #
    # Outputs: source_counts
    # matrix; one row per news source, one column per vocabulary item, value is the # of times
    #  that word appears across all articles from that news source.

    # TODO: group by news source.
    return article_counts


def model_articles(df, article_counts, vectorizer, vocab, event_uris, n_events=2,
                   frequency_thresh=0.5, force=False):
    print("Training model ...")
    # Now model
    model_pkl = 'lda-model-ev%d.pkl' % n_events
    if not force and op.exists(model_pkl):
        # This error catch all isn't working correctly.
        # the vocabulary from the articles are not assigned.

        with open(model_pkl, 'rb') as fp:
            lda_labels, lda_output_mat, lda_cats, lda_mat, model = pkl.load(fp)
    else:
        article_events = df['eventUri'].values
        for uri in event_uris:
            print uri
            event_article_counts = article_counts[article_events == uri]
            n_articles = event_article_counts.shape[0]
            word_freq_over_articles = (event_article_counts > 0).sum(axis=0) / float(n_articles)
            word_freq_over_articles = np.squeeze(np.asarray(word_freq_over_articles))

            # Store common words for this event, then blank them out in the word counts
            common_vocab_idx = word_freq_over_articles >= frequency_thresh
            article_count_idx = np.asmatrix(article_events == uri).T * np.asmatrix(common_vocab_idx)
            article_counts[article_count_idx] = 0

            print '\tevent vocab:', vocab[common_vocab_idx]

        source_counts = group_article_counts_by_source(df=df, article_counts=article_counts)

        lda_labels, lda_output_mat, lda_cats, lda_mat, model = do_lda(
            lda_mat=source_counts, vectorizer=vectorizer, vocab=vocab,
            n_topics=10, n_top_words=10, n_iter=1500, return_model=True)
        with open(model_pkl, 'wb') as fp:
            pkl.dump((lda_labels, lda_output_mat, lda_cats, lda_mat, model), fp)

    return article_counts, lda_labels, lda_output_mat, lda_cats, lda_mat, model


def main(csv_file='raw_dataframe.csv', n_events=2, min_article_length=250,
         force=False, min_vocab_length=100, min_articles=500, source_thresh=0.75,
         lda_min_appearances=2, lda_vectorization_type='count',
         plotly_username=None, plotly_api_key=None, eventregistry_api_key=None):
    """
    Do it all!
    """
    # Note: to force re-download of event article info, you'll have to delete files manually.
    # Too risky to pass a flag...
    df, event_uris = load_and_clean(
        csv_file=csv_file, min_articles=min_articles, eventregistry_api_key=eventregistry_api_key,
        n_events=n_events, min_article_length=min_article_length, source_thresh=source_thresh)

    n_events = n_events if n_events < np.inf else len(df['eventUri'].unique())
    pkl_file = '%s-ev%s.pkl' % (csv_file.replace('.csv', ''), n_events)

    article_counts, vocab, vectorizer, df = vectorize_articles(
        df=df, event_uris=event_uris, pkl_file=pkl_file, force=force,
        lda_min_appearances=lda_min_appearances, lda_vectorization_type=lda_vectorization_type,
        min_vocab_length=min_vocab_length)
    _, lda_labels, lda_output_mat, lda_cats, lda_mat, model = model_articles(
        df=df, event_uris=event_uris, vectorizer=vectorizer, vocab=vocab,
        article_counts=article_counts, force=force, n_events=n_events)

    tsne_plotly(lda_output_mat, lda_cats, lda_labels, df['source'], username=plotly_username, api_key=plotly_api_key)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train and visualize an LDA model on news article bias.')

    # Training data
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--csv-file', default=None)
    parser.add_argument('--n-events', type=int, default=np.inf)

    # Model parameters
    parser.add_argument('--min-articles', type=int, default=500,
                        help='Min # articles per event, to keep the event')
    parser.add_argument('--source-thresh', type=float, default=0.75,
                        help='Min %% of events a news source must cover, to be included.')
    parser.add_argument('--min-article-length', type=int, default=250,
                        help='Min # words in an article (pre-parsing)')
    parser.add_argument('--min-vocab-length', type=int, default=100,
                        help='Min # words in an article (post-lemmatizing, vectorizing)')
    parser.add_argument('--lda-min-appearances', type=int, default=2,
                        help='Min # appearances of a word, to be included in the vocabulary')
    parser.add_argument('--lda-vectorization-type', default='count', choices=('count', 'tfidf'),
                        help='Type of vectorization of article to word counts, to do.')

    # API info
    parser.add_argument('--plotly-username', default='bakeralex664')
    parser.add_argument('--plotly-api-key', default='hWwBstLnNCX5CsDZpOSU')
    parser.add_argument('--eventregistry-api-key', default='8b86c30c-cb8f-4d3f-aa84-077f3090e5ba')

    args = vars(parser.parse_args())
    args['csv_file'] = args['csv_file'] or 'raw_dataframe-min%d.csv' % args['min_articles']
    print main(**args)
