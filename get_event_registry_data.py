import pandas as pd
from eventregistry import *

DATA_PATH = ""
API_KEY = ""
er = EventRegistry(apiKey = API_KEY)
q = QueryEventsIter(lang='eng',
                    minArticlesInEvent=500,
                    maxArticlesInEvent=501)

# collects event ids
event_data = []
for event in q.execQuery(er, #sortBy = "date",
                        maxItems=1001):
    event_data.append(event)
df_event = pd.DataFrame(event_data)
event_uris = df_event.uri.tolist()


event_uris = [ev for ev in event_uris if ev[:3] == 'eng']

all_articles = []
for uri in event_uris:
    print "current uri: ", uri
    current_event_data = []
    
    iter = QueryEventArticlesIter(uri)
    for article in iter.execQuery(er, 
                                  lang = "eng",
                                 returnInfo=ReturnInfo(articleInfo = ArticleInfoFlags(bodyLen=-1, 
                                                                           concepts=True, 
                                                                           categories=True, 
                                                                           originalArticle=True))):
        
        current_event_data.append(article)
        
    tmp_df = pd.DataFrame(current_event_data)
    print "shape of df: {}".format(tmp_df.shape)
    print "unique url: {}".format(len(set(tmp_df['url'])))
    all_articles.append(tmp_df)
    

final_df = pd.concat(all_articles)
final_df = final_df.reset_index()
final_df.to_csv(DATA_PATH)
