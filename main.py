import time
import argparse
import logging
import json

from reddit_collection import RedditCollection
from settings import intro_text


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scrape", help="scrape data fresh",
                    action="store_true")
parser.add_argument("-r", "--subreddit", help="specify subreddits delimited by +",
                    action="store")
parser.add_argument("-p", "--pickle", help="path to pickled df",
                    action="store")
parser.add_argument("-t" , "--parse-tree", help="path to json output of corenlp batch_parse",
                    action="store")
args = parser.parse_args()


if __name__ == "__main__":

    print intro_text

    r = RedditCollection()

    if args.subreddit is not None:
        subreddits = args.subreddit.split('+')
    else:
        subreddits = r.subreddits

    if args.scrape:
        r.scrape(subreddits)
        r.pickle()
        parse_tree = None

    elif args.parse_tree is not None and args.pickle is not None:
        r.read_pickle(args.pickle)
        parse_tree = json.load(open(args.parse_tree, 'rb'))

    elif args.pickle is not None:
        r.read_pickle(args.pickle)
        parse_tree = None

    else:
        print("Please specify a path to a pickle or an -s flag.")
        exit()

    r.preprocess()
    r.process_similarities()
    r.process_top_tfidf(25)
    r.process_sentiments(parse_tree)











