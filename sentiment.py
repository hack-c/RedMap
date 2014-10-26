import sys
import argparse
import json

from corenlp import batch_parse


corenlpdir = '../corenlp-python/stanford-corenlp-full-2014-08-27/'
raw_text_dir = 'data/raw/bodytext'

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--preprocessed", help="path to preprocessed json file",
                    action="store")
parser.add_argument("-t", "--tree", help="path to preprocessed parse tree",
                    action="store")
args = parser.parse_args()


# def load_from_json(fpath):
#     """
#     loads crawled posts from .json file
#     returns dict 
#     """
#     print "\n\n====================[ loading data from %s ... ]====================\n" % (fpath)
    
#     with open(fpath, "rb") as f:
#         return_dict = json.load(f)
#         print "====================[ done.                                        ]=====================\n\n"
#         return return_dict


# def remove_nonascii(s):
#     """
#     strip out nonascii chars
#     """
#     return "".join(i for i in s if ord(i)<128)


# def build_lines_body(post_id, post):
#     """
#     format string to write for post body
#     """
#     return ["UNIQQQID " + post_id + ". " + remove_nonascii(post['body']) + "\n\n"] if len(post['body']) > 10 else []


# def build_lines_comment(post_id, comment_id, post):
#     """
#     format string for a single comment
#     """
#     return ["UNIQQQID " + post_id + ":" + comment_id + ". " + remove_nonascii(post['comments'][comment_id]['body']) + "\n\n"]


# def build_lines_whole_post(post_id, post):
#     """
#     take in unique post id
#     return list of strings in proper format
#     """

#     body_line     = build_lines_body(post_id, post)
#     comment_lines = [build_lines_comment(post_id, comment_id, post) for comment_id in post['comments'].keys()]

#     return body_line + comment_lines


# def find_mentions(terms, processed_dict):
#     """
#     iterates over processed data object,
#     if one of terms occurs in post or comment body,
#     build line and append to list
#     return list of lines to be written to text file
#     """
#     lines = []

#     print "\n\nfinding mentions..."

#     for post_id, post in processed_dict['nootropics']['posts'].iteritems():
#         sys.stdout.write('.')
#         sys.stdout.flush()

#         if terms.intersection(set(post['tokenized']['title'] + post['tokenized']['body'])):
#            lines.extend(build_lines_body(post_id, post))

#         for comment_id, comment in post['tokenized']['comments'].iteritems():
#             if terms.intersection(set(comment['body'])):
#                 lines.extend(build_lines_comment(post_id, comment_id, post))

#     return lines 



# def dump_mentions_to_raw_text(terms, processed_dict, outdir):
#     """
#     load json from inpath
#     for each post body, write line:
#         UNIQQQID asdf3. <body>
#     for each comment body, write line:
#         UNIQQQID asdf3:sdfg4. <body>
#     do batches of 100 bodies per file.
#     """

#     outfile              = open(outdir + '/raw_bodies_1.txt', 'wb')
#     lines_written        = 0
#     files_written        = 1
#     total_lines_written  = 0


#     print "\n\nwriting file %i...\n" % (files_written)

#     for line in find_mentions(terms, processed_dict):
#         sys.stdout.write('.')
#         sys.stdout.flush()

#         if lines_written > 100:
#             outfile.close()
#             print "\n\nfile %i done.\n\n" % (files_written)
#             files_written += 1
#             lines_written  = 0
#             outfile = open(outdir + '/raw_bodies_%i.txt' % (files_written), 'wb')
#             print "\n\nwriting file %i... \n\n" % (files_written)

#         outfile.write(line)
#         lines_written       += 1
#         total_lines_written += 1

#     print "\n\nwrote %i lines.\n\n" % (total_lines_written)

#     outfile.close()



def parse_raw_text(raw_text_dir):
    """
    perform the batch parse on a directory full of text files, containing one "body" per line.
    return a dict mapping unique ids to mean sentiments.
    """    
    
    print "initiating batch parse ...\n\n"
    parsed          = batch_parse(raw_text_dir, corenlpdir)
    parse_tree      = [x for x in parsed]
    sentiments      = get_sentiments(parse_tree)
    mean_sentiments = compute_mean_sentiments(sentiments)

    print "\n\nsaving ...\n\n"
    json.dump(parse_tree, open("data/processed/parse_tree.json", 'wb'))
    json.dump(mean_sentiments, open("data/processed/sentiments_dict.json", 'wb'))
    print "done."

    return mean_sentiments 



def get_sentiments(parse_tree):
    """ 
    take in the output of the batch_parse function
    extract the sentiments for each group of sentences constituting a "body"
    return a dict mapping ids to the sentiment values for each sentence in the body
    """
    

    sentiments = {}
    current_doc = None

    print "extracting sentiments from parse tree...\n\n"
    
    for parsed_sentences in parse_tree:
        for s in parsed_sentences['sentences']:
            sys.stdout.write('.')
            sys.stdout.flush()

            if s['text'][0] == 'UNIQQQID':

                if len(s['text'])   == 3:
                    current_doc = s['text'][1]

                elif len(s['text']) == 5:
                    current_doc = s['text'][1] + s['text'][2] + s['text'][3]

                sentiments[current_doc] = []
                continue

            sentiments[current_doc].append(float(s['sentimentValue']))
            
    print "\n\ndone.\n\n"
    return sentiments

def compute_main_sentiments(sentiments_dict):
    """
    {u'cladv2i': [0.0, 3.0], u'clahz3i': [3.0, 4.0]} --> {u'cladv2i': 0.0, u'clahz3i': 4.0}
    """
    # TODO!
  
def compute_mean_sentiments(sentiments_dict):
    """
    {u'cladv2i': [0.0, 3.0], u'clahz3i': [3.0, 4.0]} --> {u'cladv2i': 1.5, u'clahz3i': 3.5}
    """
    return {k: sum(v)/len(v) for k,v in sentiments_dict.iteritems()}


def annotate(sentiments, processed_dict):
    """
    take in a mapping of unique ids to sentiments and a json data object containing post data
    annotate each post in sentiments with its sentiment
    return the json data object
    """
    print "annotating posts with sentiments..."

    for pid, sentiment in sentiments.iteritems():
        sys.stdout.write('.')
        sys.stdout.flush()

        address = pid.split(":")

        if len(address)   == 1:
            processed_dict['nootropics']['posts'][pid]['sentiment'] = sentiment
        elif len(address) == 2:
            processed_dict['nootropics']['posts'][address[0]]['comments'][address[1]]['sentiment'] = sentiment

    return processed_dict


def annotate_tfidf_terms(processed_dict):
    """
    take processed dict, 
    make a pass over each post and comments,
    take set intersection between terms and the post tokens,
    for term in intersection append sentiment to term dict
    """

    print "\n\nannotating tfidf terms with mean sentiment...\n\n"

    terms = set(processed_dict['nootropics']['top_100_tfidf_terms'].keys())

    for post in processed_dict['nootropics']['posts'].itervalues():
        post_body_intersection = terms.intersection(set(post['tokenized']['body'] + post['tokenized']['title']))
        if post_body_intersection and post.has_key('sentiment'):
            for term in post_body_intersection:
                processed_dict['nootropics']['top_100_tfidf_terms'][term]['sentiment'].append((post['sentiment'], post['score'] or 1))
        for comment in post['tokenized']['comments'].itervalues():
            comment_intersection = terms.intersection(set(comment['body']))
            if comment_intersection and comment.has_key('sentiment'):
                for term in comment_intersection:
                    processed_dict['nootropics']['top_100_tfidf_terms'][term]['sentiment'].append((comment['sentiment'], comment['score'] or 1))

    return processed_dict


def compute_termwise_weighted_sentiment(processed_dict):
    """
    compute the average sentiment for each term, weighted by the score of each mention
    """
    print "computing termwise weighted sentiment...\n\n"

    for term_dict in processed_dict['nootropics']['top_100_tfidf_terms'].itervalues():
        term_dict['weighted_average_sentiment'] = sum(x[0]*x[1] for x in term_dict['sentiment']) / float(sum([x[1] for x in term_dict['sentiment']]))

    print "done.\n\n"

    return processed_dict



if __name__ == "__main__":

    if args.preprocessed is not None:
        processed = load_from_json(args.preprocessed)
        terms = set(processed['nootropics']['top_100_tfidf_terms'].keys()[:20])
        dump_mentions_to_raw_text(terms, processed, 'data/raw/bodytext')
        parse_raw_text('data/raw/bodytext')

    elif args.tree is not None:
        parse_tree = load_from_json(args.tree)
        sentiments = get_sentiments(parse_tree)
        sentiments = compute_mean_sentiments(sentiments)
        json.dump(sentiments, open("data/processed/sentiments_dict.json", 'wb'))
        processed = annotate_tfidf_terms(annotate(sentiments, load_from_json("data/processed/full_10-22-2014.json")))
        processed = compute_termwise_weighted_sentiment(processed)
        print "\n\nsaving annotated json object...\n\n"
        json.dump(processed, open("data/processed/full_10-23-2014.json", 'wb'))
        print "done.\n\n"







