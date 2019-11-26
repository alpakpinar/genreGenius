import numpy as np
import pandas as pd
import os
import re
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from collections import Counter

class DataProcessor:
    '''Class for loading and preprocessing lyrics from an input csv file.
       Feed the words into Google Word2Vec algorithm to vectorize.
       Save the vectors to output npy file.'''
    def __init__(self, input_file='songdata.csv'):
        self.df = pd.read_csv(input_file)
        # Drop cover songs
        self.df = self.df.drop_duplicates('song')
        self.num_songs = self.df.shape[0]
        print('{} songs loaded!'.format(self.num_songs))
        out_dir = './npy_dir'
        if not os.path.exists(out_dir): os.mkdir(out_dir)

    def _id(self):
        '''Create unique ID for each song using the link in the data.'''
        self.links = list(self.df['link'].str.split('/'))
        # Create the ID from the last entry, strip .html part
        create_id = lambda l: l[-1].strip('.html')
        self.ids = list(map(create_id, self.links))
        return self.ids

    def _preprocess(self):
        '''Pre-process the lyrics. Create a unique ID for each song using the link in the data.
        Remove NLTK's stopwords from the lyrics.
        Count the number of occurrence for each word after preprocessing.

        Returns two objects:
        -- A dictionary which maps (artist, song_name, ID)
        to the lyrics (list of words).
        -- A Counter object which hold the number of occurrences
        of each word in the pre-processed lyrics.'''
        self.dict = {}
        self.artists = list(self.df['artist'])
        self.song_names = list(self.df['song'])
        self.ids = self._id()
        self.lyrics = self.df['text']

        self.song_names = list(map(lambda word: word.replace("'", ""), self.song_names))

        word_counter = Counter()

        for idx, lyric in enumerate(self.lyrics):
            print('Processing lyrics: {}/{}'.format(idx, self.num_songs), end='\r')
            
            # Feed each string into the cool gensim preprocessor:
            # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess

            processed_lyrics = simple_preprocess(lyric)

            # Remove the stopwords
            sw = stopwords.words('english')
            processed_lyrics = list(filter(lambda word: word not in sw, processed_lyrics))
    
            # Count the number of words
            word_counter.update(processed_lyrics)

            # Fill the dictionary
            self.dict[(self.artists[idx], self.song_names[idx], self.ids[idx])] = processed_lyrics

        print('Processing lyrics: {}/{}'.format(self.num_songs, self.num_songs))
        print('Pre-processing complete!')
        return self.dict, word_counter

    def _save_labels_to_npy(self, data_dict):
        '''Save artist, song name pairs to an output .npy file.'''
        labels = list(data_dict.keys())
        drop_id = lambda key : key[:-1] # Don't save the ID 

        labels_clean = list(map(drop_id, labels))

        # Turn it into 2D numpy array
        labels_clean = np.array(labels_clean)

        # Save the labels to npy file
        out_file = './npy_dir/labels.npy'
        print('*'*20)
        print('Saving labels into {}'.format(out_file))
        with open(out_file, 'wb+') as f:
            np.save(f, labels_clean)

        print('Saved labels into {}'.format(out_file))

    def _vectorize_song(self, song, model):
        '''Using the model provided, get the resultant vector for the given song.'''
        # Initialize array of zeros
        vector = np.zeros(300) # Update this do not hardcode

        # Get the vectprs for words and sum them up
        for word in song:
            try:
                vector += model[f'{word}']
            except:
                continue

        # Normalize
        vector = vector / np.linalg.norm(vector)
        return vector

    def _vectorize(self, data):
        '''Given the data dictionary, get the vectors corresponding to each song
           and save them into a npy file.'''
        # Load the model 
        print('*'*20)
        print('Downloading Word2Vec model')
        model_file = './google_word2vec/GoogleNews-vectors-negative300.bin'
        model = KeyedVectors.load_word2vec_format(model_file, binary=True)
        print('Word2Vec model loaded!')
        print('*'*20)

        # Vectorize the songs
        self.vectors = []
        song_list = list(data.values())
        for idx, song in enumerate(song_list):
            print('Vectorizing: {}/{}'.format(idx, self.num_songs), end='\r')
            song_vector = self._vectorize_song(song, model)
            self.vectors.append(song_vector)

        # Convert into 2D numpy array
        self.vectors = np.array(self.vectors)
        print('Vectorizing: {}/{}'.format(self.num_songs, self.num_songs))
        print('Vectorizing complete!')

        # Save to npy file
        out_file = './npy_dir/vectors.npy'
        print('Saving vectors into {}'.format(out_file))

        with open(out_file, 'wb+') as f:
            np.save(f, self.vectors)

        print('Saved vectors into {}'.format(out_file))
        print('Done')
        print('*'*20)

    def get_common_words(self, word_counter, num_words=100):
        '''Get the most common words in the lyrics. Number of common words
           is specified in num_words option, and defaults to 100.  
        '''
        take_word = lambda tup : tup[0] # Take the words out of tuples
        take_count = lambda tup : tup[1] # Take the counts out of tuples

        most_common_words_counts = word_counter.most_common(num_words)

        most_common_words = list(map(take_word, most_common_words_counts) )
        most_common_counts = list(map(take_count, most_common_words_counts) )
 
        return most_common_words, most_common_counts

    def clean_common_words(self, data_dict, word_counter, num_words=100):
        '''Delete the most common (num_words) words from the lyrics.'''
        common_words, counts = self.get_common_words(word_counter, num_words)

        # Update the lyrics
        for dict_key in data_dict.keys():
            dict_key = tuple(dict_key) 
            data_dict[dict_key] = list(filter(lambda word: word not in common_words, data_dict[dict_key]))

        return data_dict

    def dump_to_npy(self):
        '''Vectorize the pre-processed lyrics and dump the data for each song
           to output .npy files. 
        '''
        self.dict, word_counter = self._preprocess()
        self._save_labels_to_npy(self.dict)
        self._vectorize(self.dict)
                                      
