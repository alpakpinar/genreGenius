# genreGenius

Lyrics data was downloaded from [Kaggle](https://www.kaggle.com/mousehead/songlyrics/data). The data is scraped from LyricsFreak. It is in csv format and has columns for the artist name, song title, the link to the song on LyricsFreak, and the song lyrics. All lyrics are in English and non-ASCII characters have been removed from the data.

## Loading and Processing the Data

Data is loaded and processed using the DataProcessor class, defined in `utils/data_processor.py`

### Pre-processing

The song data is downloaded from `songdata.csv` file. Lyrics are pre-processed using Gensim's [preprocessor](https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess). In the pre-processing step, all special characters are removed, all letters are converted into lowercase and very short words like 'a' are dropped out. 

### Vectorizing

Using Google's Word2Vec algorithm, each song is mapped to a 300-dimensional vector. Each song vector corresponds to the sum of the vectors of words in the lyrics of that song. All such vectors are stored in a 2D-numpy array format in the `npy_dir/output.npy` file. All the song labels, in the form of (artist\_name, song\_name) are stored in another npy file, `npy_dir/labels.npy`.

**NOTE:** Google's Word2Vec algorithm is **not** stored in this Github repository. For DataProcessor to work, make sure that the relevant .bin file is installed and located in the directory `./google_word2vec`. The algorithm can be installed from this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). 

Also make sure that the package Gensim is installed. If not, the package can be installed with pip:

```
pip install gensim
```

Once these are installed, from a Python 3 environment, execute `word2vec.py` to get the output .npy files:

```
./word2vec.py
```

