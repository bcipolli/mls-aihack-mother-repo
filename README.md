**Goal:**

* Match news articles
* Find common text in the articles, and remove
* Cluster "residual words" via topic modeling
* Apply "topics" to all residuals from a news source, to get their "bias"


**Setup & Installation:**

```
pip install -r requirements.txt
python -c "import nltk; nltk.download()"
```

Then choose to install the "popular" collection.


**NLP Demo**

This demo shows stemming, lemmatizing, and word counting (including tf-idf)

```
python nlp_demo.py
```


**Downloading data**

Run

```
python registry_data.py
```

You can tweak parameters, such as the min # articles per event or api key, within the script.



**Modeling**


```
python main.py
```

**Viewing Results**  

At the end of the modeling process a 3D graph will be generated for visualization purposes.

**Results**
- Found common words across news articles within an event.
- When clustering “residual” words via LDA, a lot of emotion words appear
- Sources did not separate by topic
  - MAYBE: sources use emotional words to describe the news; not consistent by event.

**Future Directions**
- Model new source bias within a particular topic
- Boost / attenuate emotion words via sentiment analysis
- See if there’s bias by author
- Include & apply fake news dataset.
