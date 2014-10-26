import sys
import os 
import time
import praw
import gensim
import pandas as pd

from settings import config
from settings import text_columns
from utils import tokenize


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


class RedditClient(object):
    """
    Base class for interacting with the Reddit API.
    """
    def __init__(self):
        self.user_agent  = config['user_agent']
        self.limit       = config['limit']
        self.subreddits  = config['subreddits']
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
        self.fpath       = None
        self.df          = None
        self.corpus      = None
        self.dictionary  = None
        self.tfidf       = None
        self.lsi         = None



    def scrape(self, subreddits=None):
        """
        pulls down the limit of posts for each subreddit
        gets columns in config for each post or comment 
        returns pandas DataFrame

        WARNING: this can take a long ass time 
        """
        if subreddits:
            self.subreddits = subreddits
        self.main_subreddit = self.subreddits[0]
    
        posts = []

        for subred in self.subreddits:
            print "\n\nfetching /r/%s ...\n" % (subred)

            for p in self.reddit.get_subreddit(subred, fetch=True).get_hot(limit=self.limit):

                try:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                    posts.append(Submission(p).dict)
                    p.replace_more_comments(limit=20)
                    posts.extend([Submission(c).dict for c in praw.helpers.flatten_tree(p.comments)])
                except Exception:
                    time.sleep(10)

            print "\ngot %i posts.\n\n" % (len(posts))

        df        = pd.DataFrame(posts)
        df.index  = df.name
        self.df   = df

        return self.df


    def build_fpath(self):
        """
        return a string with the current subreddits and time
        """
        return '%s_%i' % ('+'.join(self.subreddits), int(time.time()))


    def pickle(self):
        """
        pickle the posts dataframe 
        """
        assert isinstance(self.df, pd.DataFrame)

        self.fpath = self.build_fpath()

        fpath = 'data/raw/' + self.fpath + '.pkl'
        
        print "\n\nsaving to %s..."  % (fpath)

        self.df.to_pickle(fpath)


    def read_pickle(self, fpath):
        """
        read from pickle
        """
        assert os.path.exists(fpath)
        self.fpath = fpath.split('/')[-1][:-4]  # hack... but whatever

        print "\n\nreading pickle from %s..." % (fpath)

        self.df = pd.io.pickle.read_pickle(fpath)


    def preprocess(self):
        """
        tokenize text_columns and sum, dump to new tokens column 
        """
        assert isinstance(self.df, pd.DataFrame)

        print "\n\npreprocessing text..."

        self.df['body'].fillna(self.df['selftext'], inplace=True)
        self.df['body']   = self.df['body'].apply (lambda x:  u'' if x is None else x)
        self.df['title']  = self.df['title'].apply(lambda x:  u'' if x is None else x)
        del self.df['selftext']

        self.df['subreddit'] = self.df['subreddit'].apply(unicode.lower)

        for col in ('body', 'title'):
            self.df[col + '_tokens'] = self.df[col].apply(tokenize)

        self.df['tokens'] = [t + b for t,b in zip(self.df.body_tokens, self.df.title_tokens)]


    def get_subreddit_docs(self):
        """
        return a dict mapping subreddit names to long lists of tokens
        """
        docs_dict  = {subred: subr['tokens'].sum() for subr in [self.df[self.df.subreddit == subred] for subred in self.subreddits]}
        doc_map    = dict(list(enumerate(docs_dict.keys()))) # later we'll need to know which numbered doc corresponds to which subreddit

        return docs_dict, doc_map


    def build_corpus(self):
        """
        serialize and return gensim corpus of subreddit-documents
        """
        self.docs_dict, self.doc_map  = self.get_subreddit_docs()
        self.docs                     = self.docs_dict.values()

        dictionary  = gensim.corpora.Dictionary(self.docs)
        once_ids    = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(once_ids)   # remove tokens that only occur once 
        dictionary.compactify()
        dictionary.save('data/processed/' + self.fpath + '.dict')
        self.dictionary = dictionary

        corpus = [dictionary.doc2bow(doc) for doc in self.docs]

        gensim.corpora.MmCorpus.serialize ('data/processed/' + self.fpath + '.mm', corpus)
        corpus = gensim.corpora.MmCorpus  ('data/processed/' + self.fpath + '.mm')

        self.corpus = corpus

        return corpus


    def tfidf_transform(self, corpus):
        """
        transform gensim corpus to normalized tf-idf
        """
        tfidf              = gensim.models.TfidfModel(corpus, normalize=True)
        self.tfidf         = tfidf 
        self.corpus_tfidf  = tfidf[corpus]

        return self.corpus_tfidf


    def lsi_transform(self, corpus):
        """
        fit lsi 
        """
        lsi              = gensim.models.LsiModel(corpus, id2word=self.dictionary, num_topics=300)
        self.lsi         = lsi
        self.corpus_lsi  = lsi[corpus]

        return self.corpus_lsi


    def compute_subreddit_similarities(self, main_subreddit, corpus_lsi):
        """
        compute document similarities between main_subreddit and the rest
        """
        vec_bow  = self.dictionary.doc2bow(self.docs_dict[main_subreddit])
        vec_lsi  = self.lsi[vec_bow]
        index    = gensim.similarities.MatrixSimilarity(corpus_lsi)    

        index.save('data/processed/' + self.fpath + '_lsi.index')

        sims  = index[vec_lsi]
        sims  = sorted(enumerate(sims), key=lambda item: -item[1])

        similarity_map = {}

        for (i, sim) in sims:
            similarity_map[main_subreddit + '_to_' + self.doc_map[i]] = str(sim)

        return similarity_map

    def process_similarities(self):
        """
        process document similarity vectors with LSI on tfidf transformed corpus,
        dump to 
        """
        print "\n\nprocessing document similarities..."

        corpus        = self.lsi_transform(self.tfidf_transform(self.build_corpus()))
        similarities  = self.compute_subreddit_similarities(self.main_subreddit, corpus) 
        
        json.dump(similarities, 'data/processed/' + self.fpath + '_similarities.json')


    def get_total_score(self, term):
        """
        sum the scores for posts and comments which mention term
        """
        return self.df[self.df['tokens'].map(lambda x: term in x)]['score'].sum()


    def process_top_tfidf(self, n):
        """
        extract the top n highest-ranked terms from the tfidf model and their measures
        """
        print "\n\nsaving most relevant tf-idf terms..."
        
        self.top_tfidf = {}

        id2token = dict((v,k) for k,v in self.dictionary.token2id.iteritems())

        for i, doc in enumerate(self.corpus_tfidf):
            top_n_terms = sorted(doc, key=lambda item: item[1], reverse=True)[:n]
            self.top_tfidf[self.doc_map[i]] = {
                id2token[x[0]]: {
                    'tfidf': str(x[1]), 
                    'total_points': self.get_total_score(x),
                    'sentiment': []
                } for x in top_n_terms
            }

        json.dump(self.top_tfidf, 'data/processed/' + self.fpath + '_top_100_tfidf.json')









