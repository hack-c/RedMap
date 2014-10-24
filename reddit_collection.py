import sys
import os 
import time
import praw
import gensim
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
        self.corpus     = None
        self.df         = None


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

                posts.append(Submission(p).dict)
                p.replace_more_comments(limit=20)
                posts.extend([Submission(c).dict for c in praw.helpers.flatten_tree(p.comments)])

            print "\ngot %i posts.\n\n" % (len(posts))

        df       = pd.DataFrame(posts)
        df.index = df.name
        self.df  = df

        return self.df


    def build_fpath(self):
        """
        return a string with the current subreddits and time
        """
        '%s_%i' % ('_'.join(self.subreddits), int(time.time()))


    def pickle(self):
        """
        pickle the posts dataframe 
        """
        assert isinstance(self.df, pd.DataFrame)

        fpath = 'data/raw/' + self.build_fpath() + '.pkl'
        print "\n\nsaving to %s..."  % (fpath)

        self.df.to_pickle(fpath)


    def read_pickle(fpath):
        """
        read from pickle
        """
        assert os.path.exists(fpath)

        self.df = pd.io.pickle.read_pickle(fpath)


    def preprocess(self):
        """
        tokenize text_columns and sum, dump to new tokens column 
        """
        assert isinstance(self.df, pd.DataFrame)

        self.df['body'].fillna(self.df['selftext'], inplace=True)
        self.df['title'].apply(lambda x: u'' if x is None else x)
        del self.df['selftext']

        for col in ('body', 'title'):
            self.df[col + '_tokens'] = self.df[col].apply(tokenize)

        self.df['tokens'] = [t + b for t,b in zip(df.body_tokens, df.title_tokens)]


    def get_subreddit_docs(self):
        """
        return a dict mapping subreddit names to long lists of tokens
        """
        docs_dict = {subred: subr['tokens'].sum() for subr in [self.df[self.df.subreddit == subred] for subred in self.subreddits]}
        doc_map   = dict(list(enumerate(flattened.keys())))

        return docs_dict, doc_map


    def build_corpus(self):
        """
        serialize and return gensim corpus of subreddit-documents
        """
        fpath = self.build_fpath()

        docs_dict, doc_map = self.get_subreddit_docs()
        docs               = docs_dict.values()

        dictionary = gensim.corpora.Dictionary(docs)
        once_ids   = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(once_ids)   # remove tokens that only occur once 
        dictionary.compactify()
        dictionary.save('data/processed/' + fpath + '.dict')

        corpus = [dictionary.doc2bow(doc) for doc in docs]

        gensim.corpora.MmCorpus.serialize ('data/processed/' + fpath + '.mm', corpus)
        corpus = gensim.corpora.MmCorpus  ('data/processed/' + fpath + '.mm')

        self.corpus = corpus

        return corpus


    def tfidf_transform(self, corpus):
        """
        transform gensim corpus to normalized tf-idf
        """
        tfidf        = gensim.models.TfidfModel(corpus, normalize=True)
        corpus_tfidf = tfidf[corpus]

        return corpus_tfidf





class Submission(object):
    """
    Class for a Reddit post or comment.
    Dictionary of values 
    """
    def __init__(self, praw_submission):
        self.columns = config['columns']
        self._populate(praw_submission)


    def _populate(self, praw_submission):
        """
        populate the dict containing post data
        """
        self.dict = {col: getattr(praw_submission, col, None) for col in self.columns}
        self.dict['subreddit'] = praw_submission.subreddit.title









