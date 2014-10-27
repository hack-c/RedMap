import unittest

from reddit_collection import RedditCollection


class TestDumpToText(unittest.TestCase):

    def __init__(self):
        super(TestDumpToText, self).__init__()
        self.r = RedditCollection()
        self.r.read_pickle('data/raw/nootropics_1414300097.pkl')
        self.r.preprocess()


    def test_find_occurrences(self):
        terms           = ['caffeine', 'modafinil']
        occurrences_df  = self.r.find_occurrences(terms)
        self.assertGreater(len(self.r.df), len(occurrences_df))
