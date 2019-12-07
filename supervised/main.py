#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  25 11:31:45 2019

@author: Abdel
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

load_src("classifier", PATH+"supervised/polarity/classifier.py")
load_src("data_utils", PATH+"data/utils.py" )

#------------------------------------------------------------------------------
#Load data
import pandas as pd
import numpy as np

#Waiting for the official data to be approved by ethic comittee.

data = pd.read_csv(PATH+"data/twitter160polarity_mixed.csv", encoding="ISO-8859-1") 
data = data[:4000]

sentences, labels = data["text"], data["target"]


#------------------------------------------------------------------------------
#Getting pretrained word embeddings and vocab

import gensim

WORD_EMBEDDINGS_DIM = 200 #DONT MODIFY

word_emb_path = 'word_embeddings/w2v.twitter.27B.' + str(WORD_EMBEDDINGS_DIM)+ 'd.txt'
model_we = gensim.models.KeyedVectors.load_word2vec_format(PATH+word_emb_path)
vocab = model_we.vocab

#------------------------------------------------------------------------------
#Feature extraction
#Get idx and BoW representation of inputs (idxs and bagofword) 

from data_utils import TwitterPreprocessorTokenizer, text_to_sequence

BIG_SENTENCE_LENGTH = 111 #DONT MODIFY

tokenizer = TwitterPreprocessorTokenizer(stem=False, remove_stopwords=False)
word_idxs = text_to_sequence(sentences, tokenizer, vocab, maxlen_absolute=BIG_SENTENCE_LENGTH)

##------------------------------------------------------------------------------
#Get emotional polarities

from classifier import CNN_classifier

polarity_clf = CNN_classifier(model_we.vectors, BIG_SENTENCE_LENGTH)
polarity_clf.load_weights(PATH+'supervised/polarity/polarity_clf.h5')
emotional_polarities = polarity_clf.predict(word_idxs) #0:negative. 1:positive
np.savetxt(PATH+"supervised/results.out", emotional_polarities)