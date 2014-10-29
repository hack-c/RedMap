import sys
import string
import nltk

import settings


def tokenize(raw_string):
    """
    take in a string, return a tokenized and normalized list of words
    """
    if raw_string == '':
        return [u'']
    if not isinstance(raw_string, unicode):
        raise TypeError("%s is not unicode." % raw_string)
    table = {ord(c): None for c in string.punctuation}
    return filter(
        lambda x: x not in settings.useless_words, 
        nltk.word_tokenize(raw_string.lower().translate(table))
    )


def remove_nonascii(s):
    """
    strip out nonascii chars
    """
    table = {i: None for i in range(128,256)}
    return s.translate(table)


def build_line(s):
    """
    take in a series containing a single submission
    format string to write for post body
    """
    return "UNIQQQID " + s['name'] + ". " + remove_nonascii(s['body']) + " ENDBODY.\n\n" if len(s['body']) > 10 else ""


def dump_lines_to_text(lines, outdir):
    """
    take in an iterable of lines
    dump to text files in outdir
    """
    outfile              = open(outdir + '/raw_bodies_1.txt', 'wb')
    lines_written        = 0
    files_written        = 1
    total_lines_written  = 0

    print "\n\nwriting file %i...\n" % (files_written)

    for line in lines:
        sys.stdout.write('.')
        sys.stdout.flush()

        if lines_written > 100:
            outfile.close()
            print "\n\nfile %i done." % (files_written)
            files_written += 1
            lines_written  = 0
            outfile = open(outdir + '/raw_bodies_%i.txt' % (files_written), 'wb')
            print "\n\nwriting file %i...\n" % (files_written)

        outfile.write(line)
        lines_written       += 1
        total_lines_written += 1

    print "\n\nwrote %i lines." % (total_lines_written)

    outfile.close()