#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:00:48 2019

@author: Abdel
"""

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_utils", "../data/utils.py" )


import gensim
from utils import raw_texts_to_sequences, BoW
from data_utils import TwitterPreprocessorTokenizer

#texts = ...

#Getting pretrained embeddings and vocab
model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/w2v.twitter.27B.100d.txt')
vocab = model.vocab


#Get idx and BoW representation of inputs
tokenizer = TwitterPreprocessorTokenizer(stem=False)
idxs = raw_texts_to_sequences(texts, tokenizer,vocab)
bows = BoW(texts, tokenizer)




