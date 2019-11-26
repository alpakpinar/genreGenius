#!/bin/bash

# Install the stopwords dataset if not installed already
if [ ! -d "./nltk_data" ]
then
    python -m nltk.downloader -d ./nltk_data stopwords
else
    echo "./nltk_data directory already exists, moving on!"
fi

# Set the env variable so that NLTK can find the data
export NLTK_DATA="./nltk_data"
