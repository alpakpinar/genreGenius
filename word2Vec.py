#!/usr/bin/env python

import argparse
from utils.data_processor import DataProcessor

def main():
    '''Execute the job.
    Load and pre-process the lyrics,
    get the song arrays and labels and
    store them in npy files.

    If num_common_words option is specified,
    num_common_words number of most common words 
    will be removed from the lyrics.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_common_words', help='Number of most common words to be removed.', type=int)
    args = parser.parse_args()

    processor = DataProcessor()
    processor.dump_to_npy(num_common_words = args.num_common_words)

if __name__ == '__main__':
    main()
     
