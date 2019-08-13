import sys
import argparse
from gensim.corpora import WikiCorpus

def next_fname(output_dir, num=0):
    """Get the next filename to use for writing new articles."""
    count = 0
    fname = output_dir + '/' + '{:>07d}'.format(num) + '.txt'
    return count, (num+1), fname

def make_corpus(input_file, output_dir, size=10000):
    """Convert Wikipedia xml dump file to text corpus"""

    wiki = WikiCorpus(input_file)
    count, num, fname = next_fname(output_dir)
    output = open(fname, 'w')

    # iterate over texts and store them
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        count += 1
        if (count == size):
            print('%s Done.' % fname)
            output.close()
            count, num, fname = next_fname(output_dir, num)
            output = open(fname, 'w')

    # clean up resources
    output.close()
    print('Completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-o', '--output', default='./')
    args = parser.parse_args()
    dump_file = args.file
    output_dir = args.output
    make_corpus(dump_file, output_dir)