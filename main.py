#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:00:48 2019

@author: Abdel
"""

#The modelisor is a neural net that learns a dense representation of sentences. 
#It is trained in a unsepervised way, using the binary code (provided by the embeddor)
#in the output layer.

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_utils", "data/utils.py" )
load_src("embeddor", "embeddor/BinaryEmbeddor.py" )

import gensim
import pandas as pd
from data_utils import TwitterPreprocessorTokenizer, get_twitter_dataset, text_to_sequence, text_to_bow
from embeddor import BinaryEmbeddor

#HYPERPARAMETERS
PRETRAIN_LOWDIM_EMBS = True
WORD_EMBEDDING_DIM = 100 #25,50,100,200 values


#Load data
data = get_twitter_dataset("data/tweet.tsv")
sentences, labels = data["input"], data["target"]

#Getting pretrained word embeddings and vocab
emb_path = 'word_embeddings/w2v.twitter.27B.' + str(WORD_EMBEDDING_DIM)+ 'd.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(emb_path)
vocab = model.vocab


#Get idx and BoW representation of inputs
tokenizer = TwitterPreprocessorTokenizer(stem=False,stopwords=True)
word_idxs = text_to_sequence(sentences, tokenizer, vocab)
bows = text_to_bow(sentences, tokenizer)


#Get the binary low dim sentence representation (embedded sentences)
emb_path = 'embeddor/embeddings.csv'
if PRETRAIN_LOWDIM_EMBS:  
    lowdim_sentences_embs = pd.read_csv(emb_path, sep=',').values
else:
    lowdim_sentences_embs = BinaryEmbeddor()(bows.todense())
    pd.DataFrame(lowdim_sentences_embs).to_csv(emb_path, index=False)






