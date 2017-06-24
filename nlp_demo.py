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


def do_vectorize(docs, type="count", min_word_length=3, min_df=1):
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

    return X, vocab

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
