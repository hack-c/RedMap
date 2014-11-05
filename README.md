```
__________           .___ _____                 
\______   \ ____   __| _//     \ _____  ______  
 |       _// __ \ / __ |/  \ /  \\__  \ \____ \ 
 |    |   \  ___// /_/ /    Y    \/ __ \|  |_> >
 |____|_  /\___  >____ \____|__  (____  /   __/ 
        \/     \/     \/       \/     \/|__|    

by Charlie Hack
version 0.0.1 
```


Introduction
------------
RedMap is a tool for pulling down arbitrary collections of subreddits, and
applying various techniques from from NLP to the text.

Right now RedMap supports Latent Semantic Indexing for subreddit semantic similarity
and tf-idf for isolating key terms (using the wonderful `gensim` library), and sentiment 
analysis (using the Stanford CoreNLP Recursive Neural Network model for sentiment analysis, 
the current state of the art.)

Scraping is accomplished with with `praw`, a wrapper for Reddit's json API.

Once processed, the data dumps to json. Next step for the project is to visualize this
information in a beautiful and intuitive way in the browser.


Usage
-----
Run `python main.py` with:

*  an `-s` flag to scrape new data from Reddit (this can take a while)
*  combined with an `-r` flag to specify the desired subreddits, which takes arguments in the form  

`python main.py -s -r funny+askscience+business`

*  you can also specify paths to pre-scraped data or pre-processed CoreNLP batch-parse output json.


Installation
------------
For most users, running `python setup.py install` or `python setup.py develop` will
suffice to process dependencies.

You will also need to download the Stanford CoreNLP distribution to process sentiments
(this can also take a while by the way, depending on how many key terms you're interested
in.) I'm using version 3.4.1; here's a download [link](http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip). Put the path to where this directory is stored in `settings.py`.

There is also a slight modification that must be made to the corenlp wrapper library.

I'll include the diff here for convenience. I've submitted an issue to their [Bitbucket](https://bitbucket.org/torotoki/corenlp-python/issue/15/keyerror-on-namedentitytag-and-lemma-when).

```
diff --git a/corenlp/corenlp.py b/corenlp/corenlp.py
diff --git a/corenlp/corenlp.py b/corenlp/corenlp.py
diff --git a/corenlp/corenlp.py b/corenlp/corenlp.py
index ecc6129..78afab1 100755
--- a/corenlp/corenlp.py
+++ b/corenlp/corenlp.py
@@ -315,19 +315,20 @@ def parse_parser_xml_results(xml, file_name="", raw_output=False):
             token = raw_sent_list[id]['tokens']['token']
             sent['words'] = [
                 [unicode(token['word']), OrderedDict([
-                    ('NamedEntityTag', str(token['NER'])),
+                    #('NamedEntityTag', str(token['NER'])),
                     ('CharacterOffsetEnd', str(token['CharacterOffsetEnd'])),
                     ('CharacterOffsetBegin', str(token['CharacterOffsetBegin'])),
                     ('PartOfSpeech', str(token['POS'])),
-                    ('Lemma', unicode(token['lemma']))])]
+                    #('Lemma', unicode(token['lemma']))
+                                       ])]
             ]
         else:
             sent['words'] = [[unicode(token['word']), OrderedDict([
-                ('NamedEntityTag', str(token['NER'])),
+                #('NamedEntityTag', str(token['NER'])),
                 ('CharacterOffsetEnd', str(token['CharacterOffsetEnd'])),
                 ('CharacterOffsetBegin', str(token['CharacterOffsetBegin'])),
-                ('PartOfSpeech', str(token['POS'])),
-                ('Lemma', unicode(token['lemma']))])]
+                ('PartOfSpeech', str(token['POS'])),])]
+                #('Lemma', unicode(token['lemma']))
                              for token in raw_sent_list[id]['tokens']['token']]
 
         sent['dependencies'] = [[enforceList(dep['dep'])[i]['@type'],
diff --git a/corenlp/default.properties b/corenlp/default.properties
index 01e3cba..6fbabe6 100644
--- a/corenlp/default.properties
+++ b/corenlp/default.properties
@@ -1,4 +1,4 @@
-annotators = tokenize, ssplit, pos, lemma, ner, parse, dcoref
+annotators = tokenize, ssplit, parse, sentiment
 # annotators = tokenize, ssplit, pos, lemma, ner, parse, dcoref, sentiment
 
 # A true-casing annotator is also available (see below)
 ```




