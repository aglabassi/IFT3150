#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:41:36 2019

@author: Abdel

Produce a classifier (conv neural net) for polarity classification
"""


#------------------------------------------------------------------------------
#If in google colab

IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    import nltk
    drive.mount('/content/gdrive/')
    nltk.download('punkt')
    nltk.download('stopwords')
    PATH = 'gdrive/My Drive/IFT3150/'

else:
    PATH = '../'

#------------------------------------------------------------------------------
#Load files

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname('__file__'), fpath))

load_src("data_utils", PATH+"data/utils.py" )


#------------------------------------------------------------------------------
#Load data

import pandas as pd

data = pd.read_csv(PATH+"data/twitter160polarity_mixed.csv", 
                   encoding="ISO-8859-1") #0 for negative polarity, 1 for positive
data = data[:160000]

sentences, labels = data["text"].values, data["target"].values

#------------------------------------------------------------------------------
#Getting pretrained word embeddings and vocab

import gensim

WORD_EMBEDDINGS_DIM = 200 #25,50,100,200 values

word_emb_path = 'word_embeddings/w2v.twitter.27B.' + str(WORD_EMBEDDINGS_DIM)+ 'd.txt'
model_we = gensim.models.KeyedVectors.load_word2vec_format(PATH+word_emb_path)
vocab = model_we.vocab

#------------------------------------------------------------------------------
#Feature extraction
#Get idx and BoW representation of inputs (idxs and bagofword) 

from data_utils import TwitterPreprocessorTokenizer, text_to_sequence

tokenizer = TwitterPreprocessorTokenizer(stem=False,stopwords=True)
word_idxs = text_to_sequence(sentences, tokenizer, vocab)

#train/valid split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(word_idxs, labels)

#------------------------------------------------------------------------------
#Build the model

from CNN_classifier import CNN_clf

TRAIN = True
BATCH_SIZE = 32
EPOCHS = 3

cnn = CNN_clf(model_we.vectors, X_train.shape[1])
if TRAIN:
    cnn.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_valid, y_valid))
    cnn.save_weights('polarity_clf.h5')
else:
    cnn.load_weights('polarity_clf.h5')

preds = cnn.predict(X_valid)
    
    





