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
        r.read_pickle(args.pickle)

    else:
        print("Please specify a path to a pickle or an -s flag.")
        exit()

    r.preprocess()
    r.process_similarities()
    r.process_top_tfidf()









