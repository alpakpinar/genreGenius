# genreGenius

Lyrics data was downloaded from [Kaggle](https://www.kaggle.com/mousehead/songlyrics/data). The data is scraped from LyricsFreak. It is in csv format and has columns for the artist name, song title, the link to the song on LyricsFreak, and the song lyrics. All lyrics are in English and non-ASCII characters have been removed from the data.

## Setting Up The Environment

For this project, it is recommended to use a new conda environment.  

### Creating The New Environment

A conda environment can be created using `conda_env.yml` file located in the top directory: 

```
conda env create -f ./conda_env.yml
``` 

Executing this command will create a new environment called genreGenius, containing all needed packages like gensim, umap-learn and NLTK. This environment can be activated at any time by executing:

```
conda activate genreGenius
``` 

### Setup NLTK 

In this project, [NLTK](https://www.nltk.org/) library is used to get a list of stopwords to be removed from the song lyrics. To setup NLTK, source `setup_nltk.sh` file:

```
source setup_nltk.sh
```

This script will load the stopwords data into the `./nltk_data` directory (if it is not already installed) and set the neccessary environment variable `NLTK_DATA=./nltk_data` so that nltk can access the stopwords stored in this directory.

## Loading and Processing the Data

Data is loaded and processed using the DataProcessor class, defined in `utils/data_processor.py`

### Pre-processing

The song data is downloaded from `songdata.csv` file. Lyrics are pre-processed using Gensim's [preprocessor](https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess). In the pre-processing step, all special characters are removed, all letters are converted into lowercase and very short words like 'a' are dropped out. 

### Vectorizing

Using Google's Word2Vec algorithm, each song is mapped to a 300-dimensional vector. Each song vector corresponds to the sum of the vectors of words in the lyrics of that song. All such vectors are stored in a 2D-numpy array format in the `npy_dir/output.npy` file. All the song labels, in the form of (artist\_name, song\_name) are stored in another npy file, `npy_dir/labels.npy`.

**NOTE:** Google's Word2Vec algorithm is **not** stored in this Github repository. For DataProcessor to work, make sure that the relevant .bin file is installed and located in the directory `./google_word2vec`. The algorithm can be installed from this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). 

Once the algorithm is installed, from the genreGenius environment, execute `word2vec.py` to get the output .npy files:

```
./word2vec.py
```
### Removing Most Common Words

`word2vec.py` can be executed with specifying a number of most common words to be removed from the lyrics, **in addition to** the stop words. To make use of this feature, use `-n` command line argument as follows:

```
./word2vec.py -n 100
```

In the example above, the processor will remove the 100 most common words from the lyrics.

