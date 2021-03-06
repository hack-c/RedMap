import nltk
import string

version = "0.0.1"

config = {
    'user_agent': "comment grabber for a data visualization by /u/charliehack",
    'limit': None,
    'subreddits': ['nootropics'],
    'columns': ['name','title','selftext','body','score','short_link','parent_id'],
}

corenlpdir = '../corenlp-python/stanford-corenlp-full-2014-08-27/'
rawtextdir = 'data/raw/bodytext'

text_columns = ['title','selftext','body']

nonascii_table = {i: None for i in range(128,65375)}
punctuation_table = {ord(c): None for c in string.punctuation}

intro_text = """
__________           .___ _____                 
\______   \ ____   __| _//     \ _____  ______  
 |       _// __ \ / __ |/  \ /  \\\\__  \ \____ \ 
 |    |   \  ___// /_/ /    Y    \/ __ \|  |_> >
 |____|_  /\___  >____ \____|__  (____  /   __/ 
        \/     \/     \/       \/     \/|__|    

by Charlie Hack
version %s
""" % version


useless_words = {
    '',
    '1', 
    '2', 
    '3', 
    '4', 
    '5',
    '10', 
    'actually', 
    'also', 
    'always', 
    'another', 
    'anyone', 
    'anything', 
    'around', 
    'back', 
    'bad', 
    'book', 
    'cant', 
    'could', 
    'cut', 
    'david', 
    'day', 
    'days', 
    'different', 
    'doesnt', 
    'dont', 
    'downvoting', 
    'eating',
    'endbody', # TODO: find a better way to do this... this is functioning as a special "STOP" character.
    'enough', 
    'even', 
    'every', 
    'far', 
    'feel', 
    'find', 
    'first', 
    'future', 
    'get', 
    'getting', 
    'go', 
    'going', 
    'good', 
    'got', 
    'great', 
    'gt', 
    'guys', 
    'gym', 
    'hard', 
    'help', 
    'high', 
    'human', 
    'id', 
    'ill', 
    'im', 
    'isnt', 
    'ive', 
    'keep', 
    'know', 
    'less', 
    'life', 
    'like', 
    'little', 
    'long', 
    'look', 
    'looking', 
    'lot', 
    'low', 
    'made', 
    'make', 
    'many', 
    'may', 
    'maybe', 
    'mg', 
    'might', 
    'months', 
    'much', 
    'need', 
    'never', 
    'new', 
    'one', 
    'people', 
    'pretty', 
    'probably', 
    'put', 
    'raises', 
    'really', 
    'right', 
    'say', 
    'see', 
    'someone', 
    'something', 
    'start', 
    'started', 
    'still', 
    'sub', 
    'sure', 
    'take', 
    'taking', 
    'test', 
    'thats', 
    'thing', 
    'things', 
    'think', 
    'though', 
    'time', 
    'try', 
    'two', 
    'us', 
    'use', 
    'using', 
    'want', 
    'way', 
    'week', 
    'weeks', 
    'well', 
    'without', 
    'wont', 
    'work', 
    'world', 
    'would', 
    'years', 
    'youre'
}

useless_words = set(nltk.corpus.stopwords.words('english') + list(useless_words))