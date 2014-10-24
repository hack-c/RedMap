import string
import nltk

import settings


def tokenize(raw_string):
    """
    take in a string, return a tokenized and normalized list of words
    """
    table = {ord(c): None for c in string.punctuation}
    assert isinstance(raw_string, unicode)
    return filter(
        lambda x: x not in settings.useless_words, 
        nltk.word_tokenize(raw_string.lower().translate(table))
    )

def remove_nonascii(s):
    """
    strip out nonascii chars
    """
    return "".join(i for i in s if ord(i)<128)