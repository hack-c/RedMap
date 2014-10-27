# import sys
# import argparse
# import json
# from corenlp import batch_parse

# from settings import corenlpdir
# from settings import rawtextdir


# def corenlp_batch_parse(rawtextdir=rawtextdir):
#     """
#     perform the batch parse on a directory full of text files, containing one "body" per line.
#     return a dict mapping unique ids to mean sentiments.
#     """    
    
#     print "\n\ninitiating batch parse..."
#     parsed      = batch_parse(rawtextdir, corenlpdir)
#     parse_tree  = [x for x in parsed]
#     fpath       = "data/processed/" + self.fpath + "_parse_tree.json"
    
#     print "\n\nsaving parse tree to %s..." % fpath
#     json.dump(parse_tree, fpath)

#     return parse_tree 





# def get_sentiments(parse_tree):
#     """ 
#     take in the output of the batch_parse function
#     extract the sentiments for each group of sentences constituting a "body"
#     return a dict mapping ids to the sentiment values for each sentence in the body
#     """
    

#     sentiments = {}
#     current_doc = None

#     print "extracting sentiments from parse tree...\n\n"
    
#     for parsed_sentences in parse_tree:
#         for s in parsed_sentences['sentences']:
#             sys.stdout.write('.')
#             sys.stdout.flush()

#             if s['text'][0] == 'UNIQQQID':
#                 sentiments[current_doc] = []
#                 continue

#             sentiments[current_doc].append(float(s['sentimentValue']))
            
#     print "done.\n\n"
#     return sentiments

# def compute_main_sentiments(sentiments_dict):
#     """
#     {u'cladv2i': [0.0, 3.0], u'clahz3i': [3.0, 4.0]} --> {u'cladv2i': 0.0, u'clahz3i': 4.0}
#     """
#     # TODO!
  
# def compute_mean_sentiments(sentiments_dict):
#     """
#     {u'cladv2i': [0.0, 3.0], u'clahz3i': [3.0, 4.0]} --> {u'cladv2i': 1.5, u'clahz3i': 3.5}
#     """
#     return {k: sum(v)/len(v) for k,v in sentiments_dict.iteritems()}


# def annotate(sentiments, processed_dict):
#     """
#     take in a mapping of unique ids to sentiments and a json data object containing post data
#     annotate each post in sentiments with its sentiment
#     return the json data object
#     """
#     print "annotating posts with sentiments..."

#     for pid, sentiment in sentiments.iteritems():
#         sys.stdout.write('.')
#         sys.stdout.flush()

#         address = pid.split(":")

#         if len(address)   == 1:
#             processed_dict['nootropics']['posts'][pid]['sentiment'] = sentiment
#         elif len(address) == 2:
#             processed_dict['nootropics']['posts'][address[0]]['comments'][address[1]]['sentiment'] = sentiment

#     return processed_dict


# def annotate_tfidf_terms(processed_dict):
#     """
#     take processed dict, 
#     make a pass over each post and comments,
#     take set intersection between terms and the post tokens,
#     for term in intersection append sentiment to term dict
#     """

#     print "\n\nannotating tfidf terms with mean sentiment...\n\n"

#     terms = set(processed_dict['nootropics']['top_100_tfidf_terms'].keys())

#     for post in processed_dict['nootropics']['posts'].itervalues():
#         post_body_intersection = terms.intersection(set(post['tokenized']['body'] + post['tokenized']['title']))
#         if post_body_intersection and post.has_key('sentiment'):
#             for term in post_body_intersection:
#                 processed_dict['nootropics']['top_100_tfidf_terms'][term]['sentiment'].append((post['sentiment'], post['score'] or 1))
#         for comment in post['tokenized']['comments'].itervalues():
#             comment_intersection = terms.intersection(set(comment['body']))
#             if comment_intersection and comment.has_key('sentiment'):
#                 for term in comment_intersection:
#                     processed_dict['nootropics']['top_100_tfidf_terms'][term]['sentiment'].append((comment['sentiment'], comment['score'] or 1))

#     return processed_dict


# def compute_termwise_weighted_sentiment(processed_dict):
#     """
#     compute the average sentiment for each term, weighted by the score of each mention
#     """
#     print "computing termwise weighted sentiment...\n\n"

#     for term_dict in processed_dict['nootropics']['top_100_tfidf_terms'].itervalues():
#         term_dict['weighted_average_sentiment'] = sum(x[0]*x[1] for x in term_dict['sentiment']) / float(sum([x[1] for x in term_dict['sentiment']]))

#     print "done.\n\n"

#     return processed_dict



# if __name__ == "__main__":

#     if args.preprocessed is not None:
#         processed = load_from_json(args.preprocessed)
#         terms = set(processed['nootropics']['top_100_tfidf_terms'].keys()[:20])
#         dump_mentions_to_raw_text(terms, processed, 'data/raw/bodytext')
#         parse_raw_text('data/raw/bodytext')

#     elif args.tree is not None:
#         parse_tree = load_from_json(args.tree)
#         sentiments = get_sentiments(parse_tree)
#         sentiments = compute_mean_sentiments(sentiments)
#         json.dump(sentiments, open("data/processed/sentiments_dict.json", 'wb'))
#         processed = annotate_tfidf_terms(annotate(sentiments, load_from_json("data/processed/full_10-22-2014.json")))
#         processed = compute_termwise_weighted_sentiment(processed)
#         print "\n\nsaving annotated json object...\n\n"
#         json.dump(processed, open("data/processed/full_10-23-2014.json", 'wb'))
#         print "done.\n\n"







