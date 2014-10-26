import sys
import argparse
import itertools
import string
import json
import praw
import nltk
import gensim
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from settings import useless_words

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scrape", help="scrape data fresh",
                    action="store_true")
parser.add_argument("-r", "--subreddit", help="specify subreddits delimited by +",
                    action="store")
parser.add_argument("-p", "--preprocessed", help="path to preprocessed json file",
                    action="store")
args = parser.parse_args()




def scrape_and_extract(subreddits):
    """
    pulls down hot 200 posts for each subreddit
    gets title, body, and comments
    returns a dict with these indexed by subreddit

    WARNING: this takes a long ass time 
    """

    r = praw.Reddit(user_agent="comment grabber for viz by /u/charliehack")

    posts_dict = {}

    for subred in subreddits:
        posts_dict[subred] = {}

        print "\n\n====================[ fetching /r/%s ...                       ]====================\n" % (subred)

        try:
            all_posts = r.get_subreddit(subred, fetch=True).get_hot(limit=None)
        except Exception:
            continue

        for p in all_posts:
            sys.stdout.write('.')
            sys.stdout.flush()
            p.replace_more_comments(limit=16)
            flat_comments_list = praw.helpers.flatten_tree(p.comments)
            posts_dict[subred]['posts'][p.id] = {
                'title':p.title,
                'body':p.selftext,
                'score':p.score if len(p.selftext) > 10 else None,
                'url':p.short_link,
                'comments':{c.id: {'body':c.body, 'score':c.score} for c in flat_comments_list}}

        print "\n====================[ got %s posts.                            ]====================\n\n" % (str(len(posts_dict[subred])))

    return posts_dict


def dump_to_json(posts_dict, fpath="hot_posts.json"):
    """
    dumps crawled posts to a .json file
    """
    print "\n\n====================[ saving data to %s ...    ]=====================\n\n" % (fpath)
    
    with open(fpath, "wb") as f:
        json.dump(posts_dict, f)

    print "====================[ done.                                    ]=====================\n\n"

def load_from_json(fpath="hot_posts.json"):
    """
    loads crawled posts from .json file
    returns dict 
    """
    print "\n\n====================[ loading data from %s ... ]====================\n" % (fpath)
    
    with open(fpath, "rb") as f:
        return_dict = json.load(f)
        print "====================[ done.                                        ]=====================\n\n"
        return return_dict


def dump_text_to_json(posts_dict, fpath="data/raw/raw_text.json"):
    """
    extracts just the text from each post and its comments
    prepends a 'sentence' with just the id, for doing lookups later
    writes to a file.
    one 'body' per line (perhaps with multiple sentences.)

    """





def normalize_tokenize_string(raw_string):
    """
    take in a string, return a tokenized and normalized list of words
    """
    table = {ord(c): None for c in string.punctuation}
    assert isinstance(raw_string, unicode)
    return filter(
        lambda x: x not in useless_words, 
        nltk.word_tokenize(raw_string.lower().translate(table))
    )


def load_and_preprocess_dict(posts_dict, subreddits):
    """
    normalize and tokenize text using nltk
    return a dict
    """

    for subred in subreddits:
        
        print "\n\n\nnormalizing /r/%s ...\n" % (subred)
        
        for post_id, post in posts_dict[subred]['posts'].iteritems():
            sys.stdout.write('.')
            sys.stdout.flush()
            posts_dict[subred]['posts'][post_id]['tokenized'] = {
                    'title': normalize_tokenize_string(post['title']),
                    'body': normalize_tokenize_string(post['body']),
                    'score': post['score'],
                    'url': post['url'],
                    'comments': {c_id: {'body': normalize_tokenize_string(c['body']), 'score': c['score']} for c_id, c in post['comments'].iteritems()}
                }
            

    print "\n\ndone."
    return posts_dict


def flatten_post_to_tokens(tokenized_post_dict):
    """
    post_dict -> list of tokens
    """
    return tokenized_post_dict['title'] + tokenized_post_dict['body'] + list(itertools.chain(*[tokenized_post_dict['comments'][k]['body'] for k in tokenized_post_dict['comments'].keys()]))


def flatten_dict_to_tokens(tokens_dict):
    """
    organized tokens_dict -> long list of tokens
    """
    flattened = {}
    for subred in tokens_dict.keys():
        flattened[subred] = list(itertools.chain(*[flatten_post_to_tokens(post['tokenized']) for post in tokens_dict[subred]['posts'].values()]))
    return flattened

def get_freqdist(tokenized_doc):
    """
    takes in tokenized dict, concatenates all lists of terms
    returns freqdist
    """
    return nltk.FreqDist(tokenized_doc)


def build_tree(processed_dict):
    """
    taken organized tokens_dict, get freqdist, build tree with tf and sentiment values
    dump to json
    """

    fdist = get_freqdist(flatten_dict_to_tokens(processed_dict))
    nodes = fdist.most_common(200)
    tree = [['term', 'parent', 'frequency (size)', 'sentiment (color)']]
    for n in nodes:
        tree.append([n[0], '/r/nootropics', n[1], 0])

    # TODO: get sentiment

    json.dump(tree, open('data/processed/tree.json', 'wb'))


def build_gensim_corpus(processed_dict):
    """
    iterate over the large dict with ['tokenized'] posts
    append lists of tokens for post titles + bodies and 
    
    """

    texts = []


def get_total_points(term, processed_dict):
    """
    sum the scores for posts and comments which mention term
    """
    scores = []

    for post in processed_dict['nootropics']['posts'].itervalues():
        if term in set(post['tokenized']['body'] + post['tokenized']['title']):
            scores.append(post['score'] or 1)
        for comment in post['tokenized']['comments'].itervalues():
            if term in set(comment['body']):
                scores.append(comment['score'] or 1)

    return sum(scores)


# def flatten_posts_to_list(posts_dict):
    # """
    # we saved the posts indexed by subreddit, but we might want to train a model on the _whole damn thang_
    # this flattens the dict into a single list
    # """
    # return [item for sublist in [posts_dict[item] for item in subreddits] for item in sublist]


# if __name__ == "__main__":
#     if args.subreddit is not None:
#         subreddits = args.subreddit.split('+')
#     if args.scrape:
#         raw_posts = scrape_and_extract (subreddits=subreddits)
#         dump_to_json (raw_posts, fpath='data/raw/hot_10-22-2014.json')
#     if args.preprocessed is not None:
#         processed = load_from_json(fpath=args.preprocessed)
#     else:
#         raw_posts = load_from_json (fpath='data/raw/hot_10-22-2014.json')
#         subreddits = raw_posts.keys() # this is a hack, should fix this somehow...
#         processed  = load_and_preprocess_dict (raw_posts, subreddits=subreddits)
#         dump_to_json (processed, fpath='data/processed/hot_tokenized_10-21-2014.json')
    
#     print "flattening..."
#     flattened  = flatten_dict_to_tokens (processed)
#     doc_map    = dict(list(enumerate(flattened.keys())))
#     texts      = flattened.values()

#     print "\n\nbuilding corpus..."
#     dictionary = gensim.corpora.Dictionary(texts)
#     once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
#     dictionary.filter_tokens(once_ids)
#     dictionary.compactify()
#     dictionary.save('data/processed/flattened_subreddits.dict')
#     corpus = [dictionary.doc2bow(text) for text in texts]
#     gensim.corpora.MmCorpus.serialize('data/processed/flattened_subreddits.mm', corpus)
#     corpus = gensim.corpora.MmCorpus('data/processed/flattened_subreddits.mm')

    # print "\n\ntransforming to tf-idf..."
    # tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    # corpus_tfidf = tfidf[corpus]

    # print "\n\ntraining lsi..."
    # lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    # corpus_lsi = lsi[corpus_tfidf]

    # print "\n\ncomputing /r/nootropics <--> /r/* similarities..."
    # vec_bow = dictionary.doc2bow(flattened['nootropics'])
    # vec_lsi = lsi[vec_bow]
    # index = gensim.similarities.MatrixSimilarity(lsi[corpus])    
    # index.save('data/processed/flattened_subreddits_lsi.index')
    # sims = index[vec_lsi]
    # sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # for (i, sim) in sims:
    #     processed[doc_map[i]]['similarity_to_nootropics'] = str(sim)

    
    print "\n\nsaving most relevant tf-idf terms..."
    id2token = dict((v,k) for k,v in dictionary.token2id.iteritems())

    for i, doc in enumerate(corpus_tfidf):
        top_100_terms = sorted(doc, key=lambda item: item[1], reverse=True)[:100]
        processed[doc_map[i]]['top_100_tfidf_terms'] = {
            id2token[x[0]]: {
                'tfidf':str(x[1]), 
                'total_points': get_total_points(x, processed),
                'sentiment':[]
            } for x in top_100_terms
        }

    dump_to_json(processed, fpath='data/processed/full_10-22-2014.json')















    
