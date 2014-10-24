import time
import argparse

from reddit_collection import RedditCollection


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scrape", help="scrape data fresh",
                    action="store_true")
parser.add_argument("-r", "--subreddit", help="specify subreddits delimited by +",
                    action="store")
parser.add_argument("-p", "--pickle", help="path to pickled df",
                    action="store")
args = parser.parse_args()


if __name__ == "__main__":

    r = RedditCollection()

    if args.subreddit is not None:
        subreddits = args.subreddit.split('+')
    else:
        subreddits = r.subreddits

    if args.scrape:
        r.scrape(subreddits)
        r.pickle()

    elif args.pickle is not None:
        r.from_pickle(args.pickle)

    else:
        print("Please specify a path to a pickle or an -s flag.")
        exit()

    r.preprocess()
    
    corpus       = r.lsi_transform(r.tfidf_transform(r.build_corpus()))
    similarities = r.compute_subreddit_similarities(subreddits[0], corpus) # TODO: store the 'main subreddit'?
    json.dump(similarities, 'data/processed/similarities_10-24.json')









