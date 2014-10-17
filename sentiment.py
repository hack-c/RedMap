import json
from corenlp import batch_parse

corenlpdir = '../corenlp-python/stanford-corenlp-full-2014-08-27/'
raw_text_dir = 'data/raw/bodytext.txt'


def dump_json_to_raw_text(inpath, outdir):
    """
    load json from inpath
    for each post body, write line:
        UNIQQQID asdf3. <body>
    for each comment body, write line:
        UNIQQQID asdf3:sdfg4. <body>
    do batches of 100 bodies per file.
    """
    raw_posts = json.load(open(inpath, 'rb'))
    outfile   = open(outdir + '/raw_bodies_1.txt', 'wb')

    lines_written = 0
    files_written = 1
    
    for subred in raw_posts.keys():
        for post_id, post in raw_posts[subred].iteritems():

            if lines_written > 100:
                outfile.close()
                files_written += 1
                lines_written  = 0
                outfile = open(outdir + '/raw_bodies_%i.txt' % (files_written), 'wb')

            lines = build_lines(post_id, post)
            outfile.writelines(lines)
            lines_written += len(lines)

    outfile.close()


    def build_lines(post_id, post):
        """
        take in unique post id
        return list of strings in proper format
        """
        body_line     = ["UNIQQQID " + post_id + ". " + post['body'] + "\n"] if len(post['body']) > 10 else []
        comment_lines = ["UNIQQQID " + post_id + ":" + comment_id + ". " + comment['body'] + "\n" for comment_id, comment in post['comments'].iteritems()]

        return body_line + comment_lines



def parse_raw_text(raw_text_dir=raw_text_dir):
    """
    perform the batch parse on a directory full of text files, containing one "body" per line.
    return a dict mapping unique ids to mean sentiments.
    """    
    parsed          = batch_parse(raw_text_dir, corenlpdir)
    parse_tree      = [x for x in parsed]
    sentiments      = get_sentiments(parse_tree)
    mean_sentiments = compute_mean_sentiments(sentiments)

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
    return {k: lambda v: }

def compute_mean_sentiments(sentiments_dict):
    """
    {u'cladv2i': [0.0, 3.0], u'clahz3i': [3.0, 4.0]} --> {u'cladv2i': 1.5, u'clahz3i': 3.5}
    """
    return {k: sum(v)/len(v) for k,v in sentiments_dict.iteritems()}


if __name__ == "__main__":

    dump_json_to_raw_text('data/raw/hot_posts_raw_with_id.json', 'data/raw/bodytext')






