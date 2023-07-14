#!/bin/bash

mkdir resources
cd resources

# Prepare IMDB dataset
wget http://mng.bz/0tIo
unzip 0tIo
rm 0tIo

# Prepare GloVe word-embeddings
mkdir glove.6B
cd glove.6B
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
