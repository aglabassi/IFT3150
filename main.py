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
load_src("modelisor_", "modelisor/modelisor.py" )
load_src("modelisor_utils", "modelisor/utils.py" )

import pandas as pd


#HYPERPARAMETERS
USE_PRETRAIN_LOWDIM_EMBS = True  #False to retrain binary embeddings
WORD_EMBEDDING_DIM = 100 #25,50,100,200 values


#Load data
from data_utils import get_twitter_dataset

data = get_twitter_dataset("data/tweet.tsv")
sentences, labels = data["input"], data["target"]

#Getting pretrained word embeddings and vocab
import gensim

emb_path = 'word_embeddings/w2v.twitter.27B.' + str(WORD_EMBEDDING_DIM)+ 'd.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(emb_path)
vocab = model.vocab


#Get idx and BoW representation of inputs
from data_utils import TwitterPreprocessorTokenizer,  text_to_sequence, text_to_bow

tokenizer = TwitterPreprocessorTokenizer(stem=False,stopwords=True)
word_idxs = text_to_sequence(sentences, tokenizer, vocab)
bows = text_to_bow(sentences, tokenizer)


#Get the binary low dim sentence representation (embedded sentences)
from embeddor import BinaryEmbeddor

emb_path = 'embeddor/embeddings.csv'
if USE_PRETRAIN_LOWDIM_EMBS:  
    lowdim_sentences_embs = pd.read_csv(emb_path, sep=',').values
else:
    lowdim_sentences_embs = BinaryEmbeddor()(bows.todense())
    pd.DataFrame(lowdim_sentences_embs).to_csv(emb_path, index=False)

X_train, y_train, X_test, y_test = word_idxs[:6000], lowdim_sentences_embs[:6000], word_idxs[6000:], lowdim_sentences_embs[6000:]


#Training the CNN to get lowdim sentence modelisation
import torch
import torch.optim as optim
from torch.nn import BCELoss

from modelisor_ import CNN
from modelisor_utils import train_model

cnn_modelisor = CNN(vocab_size=len(vocab), sentence_length=word_idxs.shape[1], 
          emb_weights=torch.FloatTensor(model.vectors))
criterion = BCELoss()
optimizer = optim.Adam(cnn_modelisor.parameters(), lr=0.00001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_modelisor.to(device=device)

for epoch in range(100):
    a = train_model(cnn_modelisor, X_train, y_train, X_test, y_test, 
                    criterion, optimizer, device, batch_size=32)
    print(a)



