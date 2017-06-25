import numpy as np

import lda
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def do_lemmatize(docs):
    # Stem words (normalize words within the same part of speech)
    # lemmatize words (normalize words across parts of speech)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed_docs = []
    for di, doc in enumerate(docs):
        if di % 100 == 99:
            print("\tProcessed doc %d of %d" % (di + 1, len(docs)))
        words = [w.strip() for w in doc.split(' ') if w.strip()]
        stem_words = [stemmer.stem(w) for w in words]
        new_words = [lemmatizer.lemmatize(w) for w in stem_words]
        stemmed_docs.append(' '.join(new_words))
    return stemmed_docs


def do_vectorize(docs, type="count", min_word_length=3, min_df=1, sentiment_weight=0.0):
    if type == "count":
        # Count words
        cls = CountVectorizer
    elif type == "tfidf":
        # Count words, normalize by frequency
        cls = TfidfVectorizer
    else:
        raise NotImplementedError()

    vectorizer = cls(
        min_df=min_df, token_pattern=u'(?u)\\b[a-zA-Z]{%d,}\\b' % min_word_length,
        max_df=np.inf, analyzer='word', stop_words="english",
        encoding='utf-8', decode_error='ignore')

    vectorizer.fit(docs)
    X = vectorizer.transform(docs)
    vocab = np.asarray(vectorizer.vocabulary_.keys())
    vocab = vocab[np.argsort(vectorizer.vocabulary_.values())]  # ordered by counts (

    if sentiment_weight > 0:
        # from nltk.sentiment.vader import SentimentIntensityAnalyzer
        # sid = SentimentIntensityAnalyzer()
        # sentiments_df = body_df.apply(sid.polarity_scores)
        raise NotImplementedError()

    return X, vocab, vectorizer


def do_lda(lda_mat, vectorizer, vocab, n_topics=10, n_top_words=10, n_iter=1500,
           model=None, verbose=1, random_state=1, return_model=False):
    """
    Uses LDA to algorithmically find topics in a bag of words.

    Parameters
    ----------
    docs : An array of documents.
    This is generated by text_functions.read_from_CSV
    n_topics: the number of topics for LDA to find
    n_top_words: when creating topic labels, how many words to keep?
    verbose: int (optional), logging level

    Returns
    ----------

    lda_labels : the word that most represents each category

    lda_output_mat : program x topic weight matrix

    lda_cats : the argmax for lda topics of each program
    """
    # Make sure lda_mat has valid values.
    lda_mat = (100 * lda_mat).astype(int)
    good_idx = lda_mat.sum(axis=0) > 0
    good_idx = np.reshape(np.asarray(good_idx), (good_idx.size,))
    lda_mat = lda_mat[:, good_idx]
    vocab = vocab[good_idx]

    # need to find all words that were used to build a program x word
    #  count matrix for LDA
    n_docs = lda_mat.shape[0]
    if n_docs < n_topics and model is None:
        raise ValueError("Must have more docs than topics! ({n_docs} < {n_topics})".format(
            n_docs=n_docs, n_topics=n_topics))

    if model is None:
        if verbose > 0:
            print("Running LDA for {n_topics} topics for {n_iter} iterations.".format(
                n_topics=n_topics, n_iter=n_iter))
        model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_state)
        model.vectorizer = vectorizer
        model.vocab = vocab
        model.fit(lda_mat)  # .astype(np.int64))

    # top word for each topic
    lda_labels = []
    t_word = model.topic_word_
    topic_order_idx = np.argsort(np.linalg.norm(t_word, axis=1))[::-1]
    topic_word = t_word[topic_order_idx]  # order by max length
    for ti, topic_dist in enumerate(topic_word):
        topic_words = vocab[np.argsort(topic_dist)][::-1]
        topic_words = topic_words[:n_top_words]
        lda_labels.append(' '.join(topic_words))
        if verbose > 0:
            print('Topic {}: {}'.format(ti + 1, ' '.join(topic_words)))

    if verbose > 0:
        print("\tBuilding [document x topic weight] output matrix")
    lda_cats = np.zeros(n_docs, dtype=int)
    lda_output_mat = np.zeros((n_docs, n_topics))
    for x in xrange(n_docs):
        lda_output_mat[x, :] = model.doc_topic_[x][topic_order_idx]
        lda_cats[x] = np.argmax(lda_output_mat[x, :])

    if return_model:
        return lda_labels, lda_output_mat, lda_cats, lda_mat, model
    else:
        return lda_labels, lda_output_mat, lda_cats, lda_mat


if __name__ == '__main__':
    article_text = """
    Police have said they are considering manslaughter charges in relation to the deadly Grenfell Tower blaze as they revealed that both the insulation and tiles at the building failed safety tests.
    Det Supt Fiona McCormack, who is overseeing the investigation, said on Friday that officers had established that the initial cause of the fire was a fridge-freezer and that it was not started deliberately.
    She said they were trying to get to the bottom of why the fire started so quickly. Insulation
    """
    docs = [article_text] * 20
    l_docs = do_lemmatize(docs)
    vc_docs = do_vectorize(l_docs, type="count")
    vt_docs = do_vectorize(l_docs, type="tfidf")
    # print lda_categories(docs)
    print "Original text: ", article_text
    print "Stemmed & counted text: ", vc_docs[0].todense(), vc_docs[1]
    print "Stemmed & counted (tfidf) text: ", vt_docs[0].todense(), vt_docs[1]
