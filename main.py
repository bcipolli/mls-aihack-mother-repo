import os.path as op
import pickle as pkl
import re

import numpy as np
import pandas as pd

from nlp_demo import do_lemmatize, do_vectorize


def clean_text(txt):
    txt = re.sub(r'[^\sa-zA-Z]', '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt


def load_and_clean(csv_file, n_events=2, min_article_length=250):
    print("Loading and cleaning text...")
    df = pd.read_csv(csv_file)
    df['body'] = df['body'].apply(clean_text)
    event_uris = df['eventUri'].unique()

    # Reduce by article length
    good_idx = df['body'].apply(
        lambda b: not pd.isnull(b) and len(b) > min_article_length)
    df = df[good_idx]

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
            doc_counts, vocab = pkl.load(fp)
    else:
        docs = df['body'].values
        l_docs = do_lemmatize(docs)
        doc_counts, vocab, vectorizer = do_vectorize(l_docs, min_df=2, type="count")

        # Limit by vocab
        good_article_idx = np.squeeze(np.asarray((doc_counts > 0).sum(axis=1) >= min_vocab_length))
        doc_counts = doc_counts[good_article_idx]
        good_vocab_idx = np.squeeze(np.asarray(doc_counts.sum(axis=0) > 0))
        doc_counts = doc_counts[:, good_vocab_idx]
        vocab = vocab[good_vocab_idx]
        df = df.iloc[good_article_idx]

        with open(pkl_file, 'wb') as fp:
            pkl.dump((doc_counts, vocab), fp)

    return doc_counts, vocab, vectorizer, df


def model_articles(df, doc_counts, vectorizer, vocab, event_uris, n_events=2, frequency_thresh=0.5, force=False):

    print("Model each event separately...")
    doc_events = df['eventUri'].values
    residual_vocabs = []
    doc_residual_counts = []
    for uri in event_uris:
        print uri
        event_doc_counts = doc_counts[doc_events == uri]
        n_articles = event_doc_counts.shape[0]
        word_freq_over_articles = (event_doc_counts > 0).sum(axis=0) / float(n_articles)
        word_freq_over_articles = np.squeeze(np.asarray(word_freq_over_articles))

        common_vocab_idx = word_freq_over_articles >= frequency_thresh
        # event_vocab = vocab[common_vocab_idx]
        residual_vocabs.append(vocab[~common_vocab_idx])
        doc_residual_counts.append(event_doc_counts[:, ~common_vocab_idx])

    return residual_vocabs, doc_residual_counts


def main(csv_file='raw_dataframe.csv', n_events=2, min_article_length=250,
         force=False, min_vocab_length=100):
    """
    Do it all!
    """
    pkl_file = '%s-%d.pkl' % (csv_file.replace('.csv', ''), n_events)

    df, event_uris = load_and_clean(
        csv_file=csv_file, n_events=n_events, min_article_length=min_article_length)
    doc_counts, vocab, vectorizer, df = vectorize_articles(
        df=df, event_uris=event_uris, pkl_file=pkl_file, force=force,
        min_vocab_length=min_vocab_length)
    model_articles(
        df=df, event_uris=event_uris, vectorizer=vectorizer, vocab=vocab,
        doc_counts=doc_counts, force=force, n_events=n_events)


if __name__ == '__main__':
    print main(force=True)
