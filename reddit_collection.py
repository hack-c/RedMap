import sys
import os 
import time
import json
import itertools
import numpy as np 
import pandas as pd
import gensim
import praw
from corenlp import batch_parse

from settings import config
from settings import text_columns
from settings import corenlpdir
from settings import rawtextdir
from utils import dump_lines_to_text
from utils import remove_nonascii
from utils import build_line
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
        self.dict = {col: getattr(praw_submission, col, np.nan) for col in self.columns}
        self.dict['subreddit'] = praw_submission.subreddit.display_name


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


class RedMap(RedditClient):
    """
    Scrapes Reddit and extracts structured post and comment text, 
    which can be normalized, parsed, annotated, and otherwise modeled.
    """
    def __init__(self, posts_list=None, df=None):
        super(RedMap, self).__init__()
        self._populate(posts_list=posts_list, df=df)


    def _populate(self, posts_list=None, df=None, subreddits=None, fpath=None):
        if posts_list is not None and df is not None:
            raise AttributeError("Can't supply both raw posts_list and df.")
        elif posts_list is not None:
            self.df = posts_list
        elif df is not None:
            self.df = df
        self.subreddits      = subreddits or config['subreddits']
        self.main_subreddit  = self.subreddits[0]
        self.fpath           = fpath
        self.corpus          = None
        self.dictionary      = None
        self.tfidf           = None
        self.lsi             = None
        self.parse_tree      = None


    def scrape(self, subreddits):
        """
        pulls down the limit of posts for each subreddit
        gets columns in config for each post or comment 
        returns pandas DataFrame

        WARNING: this can take a long ass time 
        """
        assert subreddits is not None, "Please specify subreddits."
    
        posts = []

        for subred in subreddits:
            subred_posts = []
            print "\n\nfetching /r/%s...\n" % (subred)

            for p in self.reddit.get_subreddit(subred).get_hot(limit=self.limit):
                try:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                    subred_posts.append(Submission(p).dict)
                    p.replace_more_comments(limit=20)
                    subred_posts.extend([Submission(c).dict for c in praw.helpers.flatten_tree(p.comments)])

                except Exception:
                    time.sleep(10)

            posts.extend(subred_posts)            
            print "\ngot %i posts.\n\n" % (len(subred_posts))

        self._populate(posts_list=posts, subreddits=subreddits)


    def build_fpath(self):
        """
        return a string with the current subreddits and time
        """
        return '%s_%i' % ('+'.join(self.subreddits), int(time.time()))


    def pickle(self, suffix=''):
        """
        pickle the posts dataframe 
        """
        assert isinstance(self.df, pd.DataFrame)

        self.fpath = self.build_fpath()

        fpath = 'data/raw/' + self.fpath + suffix + '.pkl'
        
        print "\n\nsaving to %s..."  % (fpath)

        self.df.to_pickle(fpath)


    def read_pickle(self, inpath):
        """
        read from pickle
        """
        assert os.path.exists(inpath)
        fpath = inpath.split('/')[-1][:-4]  # hack... but whatever

        print "\n\nreading pickle from %s..." % (inpath)

        subreddits  = fpath[:-11].split('+')  # also a hack
        df          = pd.io.pickle.read_pickle(inpath)
        self._populate(df=df, subreddits=subreddits, fpath=fpath)
        print "\n\ndone."


    def read_parse_tree(self):
        """
        read json output from preprocessed corenlp batch_parse
        """
        treepath = "data/processed/" + self.fpath + "_parse_tree.json"
        assert os.path.exists(treepath), "no parse tree here: %s" % treepath
        print "\n\nreading json from %s..." % (treepath)
        with open(treepath, 'rb') as jsonfile:
            parse_tree = json.load(jsonfile)

        return parse_tree


    def preprocess(self):
        """
        tokenize text_columns and sum, dump to new tokens column 
        """
        assert isinstance(self.df, pd.DataFrame)

        print "\n\npreprocessing text..."

        print "\n\nnormalizing..."
        self.df['body']        = self.df['body'].fillna(self.df['selftext']).fillna(u'').apply(remove_nonascii)
        self.df['title']       = self.df['title'].fillna(u'').apply(remove_nonascii)
        self.df['subreddit']   = self.df['subreddit'].apply(unicode.lower)
        self.df['norm_score']  = self.normalize_score(self.df['score'])
        self.df['exp_score']   = self.df['norm_score'].apply(np.exp)
        self.df.index          = self.df['name']
        del self.df['selftext']

        print "\n\ntokenizing..."
        for col in ('body', 'title'):
            self.df[col + '_tokens'] = self.df[col].apply(tokenize)
        self.df['tokens'] = [t + b for t,b in zip(self.df.body_tokens, self.df.title_tokens)]

        self._patch_subreddit_title_display_name()  # TODO: remove!!

        print "\n\ndone."


    def get_subreddit_docs(self):
        """
        return a dict mapping subreddit names to long lists of tokens
        """
        docs_list  = [list(itertools.chain.from_iterable(subr['tokens'])) for subr in [self.df[self.df.subreddit == subred] for subred in self.subreddits]]
        doc_map    = dict(list(enumerate(self.subreddits)))  # later we'll need to know which numbered doc corresponds to which subreddit

        return docs_list, doc_map


    def build_corpus(self):
        """
        serialize and return gensim corpus of subreddit-documents
        """
        print "\n\nbuilding corpus..."
        self.docs_list, self.doc_map  = self.get_subreddit_docs()

        print "\n\nbuilding dictionary..."
        self.dictionary  = gensim.corpora.Dictionary(self.docs_list)
        once_ids         = [tokenid for tokenid, docfreq in self.dictionary.dfs.iteritems() if docfreq == 1]
        self.dictionary.filter_tokens(once_ids)  # remove tokens that only occur once 
        self.dictionary.compactify()
        self.dictionary.save('data/processed/' + self.fpath + '.dict')

        print "\n\nserializing corpus..."
        corpus       = [self.dictionary.doc2bow(doc) for doc in self.docs_list]
        gensim.corpora.MmCorpus.serialize('data/processed/' + self.fpath + '.mm', corpus)
        self.corpus  = gensim.corpora.MmCorpus('data/processed/' + self.fpath + '.mm')

        print "\n\ndone."

        return self.corpus


    def tfidf_transform(self, corpus):
        """
        transform gensim corpus to normalized tf-idf
        """
        print "\n\ntransforming corpus to tfidf..."
        self.tfidf         = gensim.models.TfidfModel(corpus, normalize=True)
        self.corpus_tfidf  = self.tfidf[corpus]

        return self.corpus_tfidf


    def lsi_transform(self, corpus):
        """
        fit lsi 
        """
        print "\n\nfitting lsi..."
        self.lsi         = gensim.models.LsiModel(corpus, id2word=self.dictionary, num_topics=300)
        self.corpus_lsi  = self.lsi[corpus]

        return self.corpus_lsi


    def compute_subreddit_similarities(self, main_subreddit, corpus_lsi):
        """
        compute document similarities between main_subreddit and the rest
        """
        print "\n\ncomputing /r/%s <--> /r/* similarities..." % main_subreddit
        vec_bow  = self.dictionary.doc2bow(self.docs_list[0])
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
        
        fpath = 'data/processed/' + self.fpath + '_similarities.json'
        print "\n\nsaving document similarities to %s." % fpath 
        with open(fpath, 'wb') as similaritiesfile:
            json.dump(similarities, similaritiesfile)


    def get_total_score(self, term):
        """
        sum the scores for posts and comments which mention term
        """
        return self.find_occurrences([term])['score'].sum()


    def get_total_exp_score(self, term):
        """
        sum the exp_scores for posts and comments which mention term
        """
        return self.find_occurrences([term])['exp_score'].sum()


    def process_top_tfidf(self, n):
        """
        extract the top n highest-ranked terms from the tfidf model and their measures
        """
        print "\n\nprocessing tf-idf terms..."
        
        id2token = dict((v,k) for k,v in self.dictionary.token2id.iteritems())

        rows = []

        for i, doc in enumerate(self.corpus_tfidf):
            for term in sorted(doc, key=lambda item: item[1], reverse=True)[:n]:
                sys.stdout.write('.')
                sys.stdout.flush()
                rows.append({'term': id2token[term[0]], 
                             'tfidf': str(term[1]), 
                             'total_points': self.get_total_score(term),
                             'subreddit': self.doc_map[i]})

        self.top_tfidf    = pd.DataFrame(rows)
        self.tfidf_fpath  = 'data/processed/' + self.fpath + '_top_%i_tfidf.pkl' % n

        print "\n\ndone."


    def find_occurrences(self, terms, df=self.df, field='tokens', nonzero_score=False):
        """
        take list-like of terms 
        return df containing only rows where a term in terms is mentioned
        """
        if nonzero_score:
            return df[(df[field].apply(lambda t: bool(set(t) & set(terms)))) & (df['score'] != 0)]
        else:
            return df[df[field].apply(lambda t: bool(set(t) & set(terms)))]


    def dump_occurrences(self, subreddit=None):
        """
        find mentions of top ranked tfidf terms
        exclude mentions with zero score
        format and dump batchwise to text
        """
        print "\n\nwriting file batch for CoreNLP pipeline..."
        if subreddit is None:
            subreddit = self.main_subreddit

        terms       = self.top_tfidf[self.top_tfidf['subreddit'] == subreddit]['term']
        occurrences = self.find_occurrences(terms, nonzero_score=True)
        lines       = occurrences.apply(build_line, axis=1)

        dump_lines_to_text(lines, rawtextdir)


    def corenlp_batch_parse(self, rawtextdir=rawtextdir):
        """
        perform the batch parse on a directory full of text files, containing one "body" per line.
        return a dict mapping unique ids to mean sentiments.
        """    
        
        print "\n\ninitiating batch parse..."
        parsed      = batch_parse(rawtextdir, corenlpdir)
        parse_tree  = [x for x in parsed]
        fpath       = "data/processed/" + self.fpath + "_parse_tree.json"
        
        print "\n\nsaving parse tree to %s..." % fpath
        with open(fpath, 'wb') as parsetreefile:
            json.dump(parse_tree, parsetreefile)
        print "\n\ndone."

        return parse_tree


    def get_row_ids(self, df):
        """
        lable the rows with their corresponding reddit post id.
        """
        return df.apply(lambda s: s['text'][1] if s['text'][0] == "UNIQQQID" else np.nan, axis=1).fillna(method='ffill')


    def get_sentence_length(self, df):
        """
        lable the rows with the length of the sentence in tokens.
        """
        return df['text'].apply(len)
        

    def get_main_sentiments(self, df, terms):
        """
        group df rows by id
        return a df mapping post id to 'main sentiment',
        i.e. sentimentValue for longest sentence in the post.
        """
        df['name']    = self.get_row_ids(df)
        df['length']  = self.get_sentence_length(df)
        sentiments    = df.groupby('name').apply(lambda subf: subf['sentimentValue'][subf['length'].idxmax()])

        return  dict(zip(sentiments.index, sentiments))


    def label_post_sentiments(self, parse_tree):
        """
        take in the output of the batch_parse
        return a df mapping post ids to the 'main' sentiment value
        """
        sentences       = list(itertools.chain.from_iterable([block['sentences'] for block in parse_tree]))
        sentences_df    = pd.DataFrame(sentences)
        sentiments_map  = self.get_main_sentiments(sentences_df)

        return self.df['name'].apply(lambda name: sentiments_map.get(name, np.nan))


    def label_term_sentiments(self):
        """
        take in json of top_tfidf 
        """
        print "\n\nextracting sentiments and annotating terms..."
        self.top_tfidf['sentiment']  = self.top_tfidf[self.top_tfidf['subreddit'] == self.main_subreddit]['term'].apply(self.get_mean_termwise_sentiment)
        print "\n\npickling top_tfidf..."
        pd.io.pickle.to_pickle(self.top_tfidf, self.tfidf_fpath)
        print "\n\ndone."


    def normalize_score(self, score_series):
        """
        normalize the scores to [-1,1]
        """
        neg_scores      = score_series[score_series < 0]
        pos_scores      = score_series[score_series >= 0]
        assert len(neg_scores) + len(pos_scores) == len(score_series)

        neg_score_range = float(neg_scores.max() - neg_scores.min())
        pos_score_range = float(pos_scores.max() - pos_scores.min())

        return score_series.apply(lambda s: s / neg_score_range if s < 0 else s / pos_score_range)
        

    def get_mean_termwise_sentiment(self, term):
        """
        compute the mean sentiment over submissions in which term occurs
        """
        term_subframe     = self.find_occurrences([term], nonzero_score=True)
        weighted_sents    = term_subframe.apply(lambda s: s['sentiment']*s['exp_score'], axis=1)
        weights           = term_subframe['exp_score']
        return weighted_sents.sum() / weights.sum()


    def process_sentiments(self, premade_tree=None):
        """
        sentiment pipeline
        """
        self.dump_occurrences()
        parse_tree            = premade_tree if premade_tree is not None else self.corenlp_batch_parse()
        self.df['sentiment']  = self.label_post_sentiments(parse_tree)
        self.label_term_sentiments()



    def _patch_subreddit_title_display_name(self):
        """
        this is a patch for data where the wrong subreddit name was scraped.
        TODO: REMOVE!
        """
        subrmap = {u'financial news and views': u'finance', u'advanced fitness': u'advancedfitness'}
        self.df['subreddit'] = self.df['subreddit'].apply(lambda s: subrmap.get(s, s))
























