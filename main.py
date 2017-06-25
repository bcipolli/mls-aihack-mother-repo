import os.path as op
import pickle as pkl
import re

import numpy as np
import pandas as pd

from nlp_demo import do_lda, do_lemmatize, do_vectorize
from plotting import tsne_plotly


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


def load_and_clean(csv_file, n_events=2, min_article_length=250):
    print("Loading and cleaning text...")
    df = pd.read_csv(csv_file)
    df['body'] = df['body'].apply(clean_text)
    event_uris = df['eventUri'].unique()

    # Reduce by article length
    good_idx = df['body'].apply(
        lambda b: not pd.isnull(b) and len(b) > min_article_length)
    df = df[good_idx]

    # Reduce by source.
    df = reduce_by_source(df)

    # Reduce by # events
    if n_events and n_events < np.inf:
        keep_event_uris = event_uris[:n_events]
        new_df = df[df['eventUri'].apply(lambda uri: uri in keep_event_uris)]

        assert n_events == len(new_df['eventUri'].unique())
        print("\tReduced from %d to %d events, %d to %d articles." % (
            len(event_uris), len(keep_event_uris), len(df), len(new_df)))
        df = new_df
        event_uris = keep_event_uris

    return df, event_uris


def vectorize_articles(df, event_uris, pkl_file, min_vocab_length=100, force=False):
    print("Vectorizing data...")
    if not force and op.exists(pkl_file):
        with open(pkl_file, 'rb') as fp:
            article_counts, vocab, vectorizer = pkl.load(fp)
    else:
        docs = df['body'].values
        l_docs = do_lemmatize(docs)
        article_counts, vocab, vectorizer = do_vectorize(l_docs, min_df=2, type="count")

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
    print("Model each event separately...")
    # Now model
    model_pkl = 'lda-model-%d.pkl' % n_events
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
         force=False, min_vocab_length=100):
    """
    Do it all!
    """
    pkl_file = '%s-%d.pkl' % (csv_file.replace('.csv', ''), n_events)

    df, event_uris = load_and_clean(
        csv_file=csv_file, n_events=n_events, min_article_length=min_article_length)
    article_counts, vocab, vectorizer, df = vectorize_articles(
        df=df, event_uris=event_uris, pkl_file=pkl_file, force=force,
        min_vocab_length=min_vocab_length)
    _, lda_labels, lda_output_mat, lda_cats, lda_mat, model = model_articles(
        df=df, event_uris=event_uris, vectorizer=vectorizer, vocab=vocab,
        article_counts=article_counts, force=force, n_events=n_events)
    tsne_plotly(lda_output_mat, lda_cats, lda_labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the bot')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--n-events', type=int, default=21)
    parser.add_argument('--min-article-length', type=int, default=250)
    parser.add_argument('--min-vocab-length', type=int, default=100)
    parser.add_argument('--csv-file', default='raw_dataframe.csv')
    args = vars(parser.parse_args())

    print main(**args)
