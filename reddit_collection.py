import time
import praw
import pandas as pd

from settings import config
from settings import text_columns
from utils import tokenize


class RedditClient(object):
    """
    Base class for interacting with the Reddit API.
    """
    def __init__(self):
        self.user_agent = config['user_agent']
        self.limit      = config['limit']
        self.subreddits = config['subreddits']
        self._connect()

    def _connect(self):
        self.reddit = praw.Reddit(user_agent=self.user_agent)


class RedditCollection(RedditClient):
    """
    Scrapes Reddit and extracts structured post and comment text, 
    which can be normalized, parsed, annotated, and otherwise modeled.
    """
    def __init__(self):
        super(RedditCollection, self).__init__()
        self.subreddits = None
        self.posts      = None


    def scrape(self, subreddits=None):
        """
        pulls down the limit of posts for each subreddit
        gets columns in config for each post or comment 
        returns pandas DataFrame

        WARNING: this can take a long ass time 
        """
        if subreddits:
            self.subreddits = subreddits
    
        posts = []

        for subred in self.subreddits:
            print "\n\nfetching /r/%s ...\n" % (subred)

            for p in self.reddit.get_subreddit(subred, fetch=True).get_hot(limit=self.limit):
                sys.stdout.write('.')
                sys.stdout.flush()

                posts.append(Submission(p))
                p.replace_more_comments(limit=20)
                posts.extend([Submission(c) for c in praw.helpers.flatten_tree(p.comments)])

            print "\ngot %s posts.\n\n" % (str(len(posts_dict[subred])))

        df       = pd.DataFrame(posts)
        df.index = df.name
        self.df  = df

        return self.df


    def pickle(self):
        """
        pickle the posts dataframe 
        """
        assert isinstance(self.posts, pd.DataFrame)

        fpath = 'data/raw/%s_%i.pkl' % ('_'.join(self.subreddits), int(time.time()))
        print "\n\nsaving to %s..."  % (fpath)

        self.posts.to_pickle(fpath)


    def preprocess(self):
        """
        tokenize text_columns and sum, dump to new tokens column 
        """
        assert isinstance(self.posts, pd.DataFrame)

        # TODO:
        # - sum text columns



class Submission(object):
    """
    Class for a Reddit post or comment.
    Dictionary of values 
    """
    def __init__(self, praw_submission):
        self.columns = config['columns']
        self._populate(praw_submission)
        return self.dict 

    def _populate(self, praw_submission):
        """
        populate the dict containing post data
        """
        self.dict              = {col: getattr(praw_submission, col, None) for col in self.columns}
        self.dict['subreddit'] = praw_submission.subreddit.title









