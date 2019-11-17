from gensim.utils import simple_preprocess
import pandas as pd

class DataLoader:
    '''Class for loading and preprocessing lyrics from an input csv file.'''
    def __init__(self, input_file='songdata.csv'):
        self.df = pd.read_csv(input_file)
        self.num_songs = self.df.shape[0]
        print('{} songs loaded!'.format(self.num_songs))

    def _id(self):
        '''Create unique ID for each song using the link in the data.'''
        self.links = list(self.df['link'].str.split('/'))
        # Create the ID from the last entry, strip .html part
        create_id = lambda l: l[-1].strip('.html')
        self.ids = list(map(create_id, self.links))
        return self.ids

    def _preprocess(self):
        '''Pre-process the lyrics. Create a unique ID for each song using the link in the data.

        Returns a dictionary which maps (artist, song_name, ID)
        to the lyrics (list of words).'''
        self.dict = {}
        self.artists = list(self.df['artist'])
        self.song_names = list(self.df['song'])
        self.ids = self._id()
        self.lyrics = self.df['text']

        for idx, lyric in enumerate(self.lyrics):
            if idx % 5000 == 0: print('Processing lyrics: {}'.format(idx))
            # Feed each string into the cool gensim preprocessor:
            # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess

            processed_lyrics = simple_preprocess(lyric)

            # Fill the dictionary
            self.dict[(self.artists[idx], self.song_names[idx], self.ids[idx])] = processed_lyrics

        print('Pre-processing complete!')
        return self.dict

    def _aggregate(self, data):
        '''Aggregate the lyrics into a 2D array.'''
        # Initialize the 2D list
        word_list_2d = []
        for word_list in list(data.values()):
            word_list_2d.append(word_list)
        
        return word_list_2d
    
    def get_data(self):
        '''Get a tuple containing: 
        Dictionary that maps (artist, songname, ID) --> lyrics
        2D set of lyrics as list of words.
        '''
        self.dict = self._preprocess()
        self.word_list_2d = self._aggregate(self.dict)
        return (self.dict, self.word_list_2d)
