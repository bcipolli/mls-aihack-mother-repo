import os.path as op

import eventregistry as er
import pandas as pd


def fetch_events(api_key, min_articles=500, force=False):
    event_registry = er.EventRegistry(apiKey=api_key)
    qei = er.QueryEventsIter(
        lang='eng', minArticlesInEvent=min_articles)

    # collects event ids
    all_events_csv_file = op.join('csv', 'events_min%d.csv' % min_articles)
    if not force and op.exists(all_events_csv_file):
        df_event = pd.read_csv(all_events_csv_file)
    else:
        event_data = []
        for event in qei.execQuery(event_registry, maxItems=1001):
            event_data.append(event)
        df_event = pd.DataFrame(event_data)
        df_event.to_csv(all_events_csv_file, encoding='utf-8')
        del event_data

    event_uris = df_event.uri.tolist()
    event_uris = [ev for ev in event_uris if ev[:3] == 'eng']

    all_articles = []
    for uri in event_uris:
        print "current uri: ", uri
        current_event_data = []

        event_csv_file = op.join('csv', 'event-%s.csv' % uri)
        if not force and op.exists(event_csv_file):
            tmp_df = pd.read_csv(event_csv_file)
        else:
            query_iter = er.QueryEventArticlesIter(uri)
            for article in query_iter.execQuery(event_registry, lang="eng"):
                current_event_data.append(article)
            tmp_df = pd.DataFrame(current_event_data)
            tmp_df.to_csv(event_csv_file, encoding='utf-8')

        print "shape of df: {}".format(tmp_df.shape)
        print "unique url: {}".format(len(set(tmp_df['url'])))
        all_articles.append(tmp_df)

    final_df = pd.concat(all_articles)
    final_csv_file = 'articles-min%d.csv' % min_articles
    if final_csv_file:
        final_df.to_csv(final_csv_file, encoding='utf-8')
    return final_df


if __name__ == '__main__':
    fetch_events(api_key="599feb03-0270-47d7-9910-61f3ad7dc77c")
