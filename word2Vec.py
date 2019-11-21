#!/usr/bin/env python

from utils.data_processor import DataProcessor

def main():
    '''Execute the job.
    Load and pre-process the lyrics,
    get the song arrays and labels and
    store them in npy files.'''
    processor = DataProcessor()
    processor.dump_to_npy()

if __name__ == '__main__':
    main()
     
