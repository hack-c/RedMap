import sys
import argparse
import json

from corenlp import batch_parse
from preprocess import load_from_json

corenlpdir = '../corenlp-python/stanford-corenlp-full-2014-08-27/'
raw_text_dir = 'data/raw/bodytext'

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--preprocessed", help="path to preprocessed json file",
                    action="store")
args = parser.parse_args()


def remove_nonascii(s):
    """
    strip out nonascii chars
    """
    return "".join(i for i in s if ord(i)<128)


def build_lines_body(post_id, post):
    """
    format string to write for post body
    """
    return ["UNIQQQID " + post_id + ". " + remove_nonascii(post['body']) + "\n\n"] if len(post['body']) > 10 else []


def build_lines_comment(post_id, comment_id, post):
    """
    format string for a single comment
    """
    return ["UNIQQQID " + post_id + ":" + comment_id + ". " + remove_nonascii(post['comments'][comment_id]['body']) + "\n\n"]


def build_lines_whole_post(post_id, post):
    """
    take in unique post id
    return list of strings in proper format
    """

    body_line     = build_line_body(post_id, post)
    comment_lines = [build_line_comment(post_id, comment_id, post) for comment_id in post['comments'].keys()]

    return body_line + comment_lines


def find_mentions(terms, processed_dict):
    """
    iterates over processed data object,
    if one of terms occurs in post or comment body,
    build line and append to list
    return list of lines to be written to text file
    """
    lines = []

    print "\n\nfinding mentions..."

    for post_id, post in processed_dict['nootropics']['posts'].iteritems():
        sys.stdout.write('.')
        sys.stdout.flush()

        if terms.intersection(set(post['tokenized']['title'] + post['tokenized']['body'])):
           lines.extend(build_lines_body(post_id, post))

        for comment_id, comment in post['tokenized']['comments'].iteritems():
            if terms.intersection(set(comment)):
                lines.extend(build_lines_comment(post_id, comment_id, post))

    return lines 



def dump_mentions_to_raw_text(terms, processed_dict, outdir):
    """
    load json from inpath
    for each post body, write line:
        UNIQQQID asdf3. <body>
    for each comment body, write line:
        UNIQQQID asdf3:sdfg4. <body>
    do batches of 100 bodies per file.
    """

    outfile              = open(outdir + '/raw_bodies_1.txt', 'wb')
    lines_written        = 0
    files_written        = 1
    total_lines_written  = 0


    print "\n\nwriting file %i...\n" % (files_written)

    for line in find_mentions(terms, processed_dict):
        sys.stdout.write('.')
        sys.stdout.flush()

        if lines_written > 100:
            outfile.close()
            print "\n\nfile %i done.\n\n" % (files_written)
            files_written += 1
            lines_written  = 0
            outfile = open(outdir + '/raw_bodies_%i.txt' % (files_written), 'wb')
            print "\n\nwriting to file %i ... \n\n" % (files_written)

        outfile.write(line)
        lines_written       += 1
        total_lines_written += 1

    print "\n\nwrote %i lines.\n\n" % (total_lines_written)

    outfile.close()



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
    json.dump(mean_sentiments, open("data/processed/sentiments_dict.json", 'wb'))
    print "done."

    return mean_sentiments 



def get_sentiments(parse_tree):
    """ 
    take in the output of the batch_parse function
    extract the sentiments for each group of sentences constituting a "body"
    return a dict mapping ids to the sentiment values for each sentence in the body
    """
    parsed_sentences = parse_tree[0]['sentences']

    sentiments = {}
    current_doc = None

    for s in parsed_sentences:
        sys.stdout.write('.')
        sys.stdout.flush()

        if s['text'][0] == 'UNIQQQID':
            current_doc = s['text'][1]
            sentiments[current_doc] = []
            continue

        sentiments[current_doc].append(float(s['sentimentValue']))
        
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


if __name__ == "__main__":

    if args.preprocessed is not None:
        processed = load_from_json(args.preprocessed)

    terms = set(processed['nootropics']['top_100_tfidf_terms'].keys()[:20])
    dump_mentions_to_raw_text(terms, processed, 'data/raw/bodytext')
    parse_raw_text('data/raw/bodytext')



    # dump_json_to_raw_text('data/raw/hot_posts_raw_with_id.json', 'data/raw/bodytext')
    # parse_raw_text('data/raw/bodytext')






