import os.path as op

import eventregistry as er
import pandas as pd


def fetch_event_articles(api_key, min_articles=500, force=False, save_on_api_fail=True, csv_file=None):

    # Uncache csv file.
    if not force and op.exists(csv_file):
        print("Loading articles from disk...")
        return pd.read_csv(csv_file)

    event_registry = er.EventRegistry(apiKey=api_key, repeatFailedRequestCount=2)

    # Single query to collect event ids
    all_events_csv_file = op.join('csv', 'events_min%d.csv' % min_articles)
    all_events_gzip_file = all_events_csv_file + '.gz'
    if not force and op.exists(all_events_gzip_file):
        df_event = pd.read_csv(all_events_gzip_file, compression='gzip')
    else:
        event_data = []
        qei = er.QueryEventsIter(
            lang='eng', minArticlesInEvent=min_articles, maxArticlesInEvent=min_articles * 10)
        for event in qei.execQuery(event_registry, maxItems=1001):
            event_data.append(event)
        df_event = pd.DataFrame(event_data)
        df_event.to_csv(all_events_gzip_file, encoding='utf-8', compression='gzip')
        del event_data

    event_uris = df_event.uri.tolist()
    event_uris = [ev for ev in event_uris if ev[:3] == 'eng']
    print("Downloading articles for %d events..." % len(event_uris))

    # Loop to retrieve all articles for an event.
    return_info = er.ReturnInfo(
        articleInfo=er.ArticleInfoFlags(
            bodyLen=-1,
            concepts=True,
            categories=True,
            originalArticle=True))

    all_articles = []
    api_failed = False
    for uri in event_uris:
        print "current uri: ", uri
        current_event_data = []

        event_gzip_file = op.join('csv', 'event-%s.csv.gz' % uri)
        if not force and op.exists(event_gzip_file):
            tmp_df = pd.read_csv(event_gzip_file, compression='gzip')
        elif api_failed:
            print("\tSkipping; API failed.")
            try:
                query_iter = er.QueryEventArticlesIter(uri)
                for article in query_iter.execQuery(event_registry, lang="eng", returnInfo=return_info):
                    current_event_data.append(article)
            except TypeError:
                # This is how API errors come through.
                if save_on_api_fail:
                    print("\tWARNING: API failed. Skipping.")
                    api_failed = True  # end loop; we can't continue.
                    continue
                else:
                    raise

            # Specify columns, so that we skip any empty events.
            tmp_df = pd.DataFrame(current_event_data, columns=[
                'body', 'categories', 'concepts', 'date', 'dateTime', 'eventUri',
                'id', 'isDuplicate', 'lang', 'originalArticle', 'sim', 'source',
                'time', 'title', 'uri', 'url'])
            tmp_df.to_csv(event_gzip_file, encoding='utf-8', compression='gzip')

        if len(tmp_df) == 0:
            print("WARNING: event contains no articles.")
        # print "shape of df: {}".format(tmp_df.shape)
        # print "unique url: {}".format(len(set(tmp_df['url'])))
        all_articles.append(tmp_df)

    # Combine all news articles into a single dataframe.
    final_df = pd.concat(all_articles)
    csv_file = csv_file or 'articles-min%d.csv' % min_articles
    final_df.to_csv(csv_file, encoding='utf-8')

    return final_df


if __name__ == '__main__':
    fetch_event_articles(api_key="8b86c30c-cb8f-4d3f-aa84-077f3090e5ba")
