import argparse
import string
import json
import praw
import nltk
import gensim
import pandas as pd

from metapython.op.text_preprocessing import preprocess_raw_text
from metapython.op.text_preprocessing import remove_stopwords
from metapython.op.text_preprocessing import remove_words_in_set
from metapython.op.text_preprocessing import join_text_columns


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scrape", help="scrape data fresh",
                    action="store_true")
args = parser.parse_args()


subreddits = ['nootropics', 
              'nutrition', 
              'Health', 
              'FixMyDiet', 
              'Dietetics', 
              'supplements', 
              'stackadvice', 
              'afinil', 
              'drugnerds', 
              'foodnerds', 
              'caffeine', 
              'braintraining', 
              'enhance', 
              'tdcs', 
              'selfimprovement', 
              'gainit', 
              'advancedfitness', 
              'steroids', 
              'longevity', 
              'SENS', 
              'Futurism', 
              'Futurology', 
              'Posthumanism', 
              'Singularitarianism', 
              'Singularity', 
              'Transhuman', 
              'Transhumanism', 
              'Neurophilosophy', 
              'SFStories']

useless_words = {'actually', 'also', 'always', 'another', 'anyone', 'anything', 'around', 'back', 'bad', 'become', 'best', 'better', 'book', 'cant', 'could', 'cut', 'david', 'day', 'days', 'diet', 'different', 'doesnt', 'dont', 'downvoting', '10', 'eating', 'enough', 'even', 'every', 'far', 'feel', 'find', 'first', 'future', 'get', 'getting', 'go', 'going', 'good', 'got', 'great', 'guys', 'gym', 'hard', 'help', 'high', 'human', 'id', 'ill', 'im', 'isnt', 'ive', 'keep', 'know', 'less', 'life', 'like', 'little', 'long', 'look', 'looking', 'lot', 'low', 'made', 'make', 'many', 'may', 'maybe', 'mg', 'might', 'months', 'much', 'need', 'never', 'new', 'one', 'people', 'pretty', 'probably', 'put', 'raises', 'really', 'right', 'say', 'see', 'someone', 'something', 'start', 'started', 'still', 'sub', 'sure', 'take', 'taking', 'test', 'thats', 'thing', 'things', 'think', 'though', 'time', 'try', 'two', 'us', 'use', 'using', 'want', 'way', 'week', 'weeks', 'well', 'without', 'wont', 'work', 'world', 'would', 'years', 'youre'}
useless_words = set(nltk.corpus.stopwords.words('english') + list(useless_words))

def scrape_and_extract(subreddits=subreddits):
    """
    pulls down hot 200 posts for each subreddit
    gets title, body, and comments
    returns a dict with these indexed by subreddit

    WARNING: this takes a long ass time 
    """

    r = praw.Reddit(user_agent="comment grabber for viz by /u/charliehack")

    posts_dict = {}

    for subred in subreddits:
        posts_dict[subred] = []

        print "==========[ fetching /r/%s ... ]==========\n\n" % (subred)

        try:
            all_posts = r.get_subreddit(subred, fetch=True).get_hot(limit=200)
        except Exception:
            continue

        for p in all_posts:
            p.replace_more_comments(limit=16)
            flat_comments_list = praw.helpers.flatten_tree(p.comments)
            posts_dict[subred].append(
                {'title':p.title, 
                'body':p.selftext, 
                'comments':[c.body for c in flat_comments]})
            print('.'),

        print "==========[ got %s posts.      ]==========\n\n" % (str(len(posts_dict[subred])))


    return posts_dict


def dump_to_json(posts_dict, fpath="hot_posts.json"):
    """
    dumps crawled posts to a .json file
    """
    print "==========[ saving data to %s ...    ]===========\n\n" % (fpath)
    
    with open(fpath, "wb") as f:
        json.dump(text, f)


def load_from_json(fpath="hot_posts.json"):
    """
    loads crawled posts from .json file
    returns dict 
    """
    print "\n\n==========[ loading data from %s ... ]==========\n\n" % (fpath)
    
    with open(fpath, "rb") as f:
        return json.load(f)


# def load_and_preprocess_dict(posts_dict, subreddits=['nootropics']):
#     """
#     normalize and tokenize text using nltk
#     return a dict
#     """
#     table = string.maketrans("","")
#     processed = {}

#     for subred in subreddits:
#         tokenized = filter(
#                         lambda x: x not in useless_words, 
#                         nltk.word_tokenize(
#                             ' '.join(
#                                 [' '.join(
#                                     [post['title'], post['body'], post['comments']]).lower() for post in posts_dict[subred]
#                                 ])
#                             )
#                         )


#     for subred in subreddits:
#         data = posts_dict['subred']
        



def flatten_posts_to_list(posts_dict):
    """
    we saved the posts indexed by subreddit, but we might want to train a model on the _whole damn thang_
    this flattens the dict into a single list
    """
    return [item for sublist in [posts_dict[item] for item in subreddits] for item in sublist]


def load_and_preprocess_df(posts_list):
    """
    load the posts into a dataframe, preprocess the text
    returns dataframe object
    """
    print "==========[ preprocessing text ...               ]==========\n\n"

    df     = pd.DataFrame(posts)
    df     = preprocess_raw_text(cols=['title', 'body', 'comments']) (df)
    df     = join_text_columns(incols=['title', 'body', 'comments'], outcol=['post_document']) (df)
    df     = remove_stopwords(cols=['post_document'], langs=['english']) (df)
    df['post_document'] = df['post_document'].apply(lambda l: [x for x in l if x not in useless_words])

    return df


def train_lda(df, num_topics=200):
    """
    takes in preprocessed df, trains LDA on the post_document column, prints out the topics
    """
    print "==========[ training model ...                   ]==========\n\n"

    # format the post documents as a gensim dictionary corpus
    gensim_dict = gensim.corpora.Dictionary(df['post_document'])

    # transform the corpus as a sparse bag-of-words representation and store it in another column
    df['post_document_bow'] = df['post_document'].apply(gensim_dict.doc2bow)

    # train lda on the bag-of-words column
    model = gensim.models.ldamodel.LdaModel(df['post_document_bow'], num_topics=num_topics, id2word=gensim_dict)
    topics = model.show_topics(num_topics=num_topics)

    print "==========[ done.                                ]==========\n\n"


    for topic, index in zip(topics, range(num_topics)):
        print "==========[              topic #%s:               ]==========\n" % (str(index+1))
        print topic
        print
        print


if __name__ == "__main__":
    if args.scrape:
        posts = scrape_and_extract()
        dump_to_json (posts)
        posts = flatten_posts_to_list (posts)
    else:
        posts = flatten_posts_to_list( load_from_json() )    
    df = load_and_preprocess_df (posts)
    train_lda (df)







    
