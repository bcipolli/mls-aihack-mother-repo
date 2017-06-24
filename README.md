** Goal:

* Match news articles
* Find common text in the articles, and remove
* Cluster "residual words" via topic modeling
* Apply "topics" to all residuals from a news source, to get their "bias"


** Setup & Installation:

```
pip install -r requirements.txt
python -c "import nltk; nltk.download()"
```

Then choose to install the "popular" collection.


** NLP Demo

This demo shows stemming, lemmatizing, and word counting (including tf-idf)

```
python nlp_demo.py
```
