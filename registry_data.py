import pandas as pd
import eventregistry as er


def fetch_events(api_key, min_articles=500, csv_file=None):
    event_registry = er.EventRegistry(apiKey=api_key)
    qei = er.QueryEventsIter(
        lang='eng', minArticlesInEvent=min_articles)

    # collects event ids
    event_data = []
    for event in qei.execQuery(event_registry, maxItems=1001):
        event_data.append(event)
    df_event = pd.DataFrame(event_data)
    event_uris = df_event.uri.tolist()

    event_uris = [ev for ev in event_uris if ev[:3] == 'eng']

    all_articles = []
    for uri in event_uris:
        print "current uri: ", uri
        current_event_data = []

        query_iter = er.QueryEventArticlesIter(uri)
        for article in query_iter.execQuery(er, lang="eng"):
            current_event_data.append(article)

        tmp_df = pd.DataFrame(current_event_data)
        print "shape of df: {}".format(tmp_df.shape)
        print "unique url: {}".format(len(set(tmp_df['url'])))
        all_articles.append(tmp_df)

    final_df = pd.concat(all_articles)

    if csv_file:
        final_df.to_csv(csv_file, encoding='utf-8')
    return final_df


if __name__ == '__main__':
    fetch_events(csv_file='raw_dataframe.csv', api_key="21c52a6d-4ce5-48b7-98df-7c00c27f866a")
