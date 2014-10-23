import argparse

import settings
from scraper import RedditScraper

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scrape", help="scrape data fresh",
                    action="store_true")
parser.add_argument("-r", "--subreddit", help="specify subreddits delimited by +",
                    action="store")
parser.add_argument("-p", "--preprocessed", help="path to preprocessed json file",
                    action="store")
args = parser.parse_args()


if __name__ == "__main__":

    if args.subreddit is not None:
        subreddits = args.subreddit.split('+')
    else:
        subreddits = settings.config['subreddits']

    if args.scrape:
        df = RedditScraper().scrape(subreddits)
