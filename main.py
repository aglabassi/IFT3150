#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:00:48 2019

The modelisor is a neural net that learns a dense representation of sentences. 
It is trained in a unsepervised way, using the binary code (provided by the embeddor)
in the output layer.

@author: Abdel
"""

import time

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
    PATH = ''



#------------------------------------------------------------------------------
#Load files

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname('__file__'), fpath))

load_src("data_utils", PATH+"data/utils.py" )
load_src("embeddor", PATH+"embeddor/BinaryEmbeddor.py" )
load_src("modelisor_", PATH+"modelisor/modelisor.py" )
load_src("modelisor_utils", PATH+"modelisor/utils.py" )
load_src("CNN_classifier", PATH+"polarity/CNN_classifier.py")

#------------------------------------------------------------------------------
#Load data

import pandas as pd

data = pd.read_csv(PATH+"data/twitter160polarity_mixed.csv", 
                   encoding="ISO-8859-1") 
data = data[:4000]

sentences, labels = data["text"], data["target"]

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

from data_utils import TwitterPreprocessorTokenizer, text_to_sequence, text_to_bow

BIG_SENTENCE_LENGTH = 111 #DONT MODIFY

tokenizer = TwitterPreprocessorTokenizer(stem=False,remove_stopwords=False)
word_idxs = text_to_sequence(sentences, tokenizer, vocab, 
                             maxlen_absolute=BIG_SENTENCE_LENGTH)
bows = text_to_bow(sentences, tokenizer)

#------------------------------------------------------------------------------
#Polarity classification

from CNN_classifier import CNN_classifier
    
polarity_clf = CNN_classifier(model_we.vectors, BIG_SENTENCE_LENGTH)
polarity_clf.load_weights('polarity/polarity_clf.h5')
polarities = polarity_clf.predict(word_idxs)
 

#------------------------------------------------------------------------------
#Get the binary low dim sentence representation (embedded sentences)

USE_PRETRAIN_BIN_EMBS = False
BIN_DIM = 20

bin_emb_path = 'embeddor/bin_embs_'+str(BIN_DIM)+'d.csv'

if USE_PRETRAIN_BIN_EMBS:  
    lowdim_sentences_embs = pd.read_csv(PATH+bin_emb_path, sep=',').values
else:
    t = time.time()
    from embeddor import BinaryEmbeddor
    lowdim_sentences_embs = BinaryEmbeddor(bin_dim=BIN_DIM)(bows.todense())
    pd.DataFrame(lowdim_sentences_embs).to_csv(PATH+bin_emb_path, index=False)
    t = time.time() - t
    print("binary embeddings trained get after ", t, "seconds for n_inputs= ", 
          len(sentences))



#------------------------------------------------------------------------------
#Simple train / valid split
    
TRAIN_SIZE=0.8
DEL = int(len(word_idxs)*TRAIN_SIZE)

X_train, bin_train = word_idxs[:DEL], lowdim_sentences_embs[:DEL]
X_test, bin_test = word_idxs[DEL:], lowdim_sentences_embs[DEL:]

#------------------------------------------------------------------------------
#Training the CNN to get lowdim sentence modelisation


import torch
import torch.optim as optim
from torch.nn import BCELoss

from modelisor_ import CNN
from modelisor_utils import train_model

#Hyperparameters
EPOCHS = 20
BATCH_SIZE = 16
K_TOP = 3
LEARNING_RATE = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_modelisor = CNN(vocab_size=len(vocab), sentence_length=BIG_SENTENCE_LENGTH, \
                    emb_weights=torch.FloatTensor(model_we.vectors),  k_top=K_TOP)
cnn_modelisor.to(device)

criterion = BCELoss()
optimizer = optim.Adam(cnn_modelisor.parameters(), lr=LEARNING_RATE)

losses, losses_valid = [],[]
for epoch in range(EPOCHS):
    loss, loss_valid = train_model(cnn_modelisor, X_train, bin_train, X_test, bin_test, 
                    criterion, optimizer, device, batch_size=BATCH_SIZE)
    
    losses.append(loss)
    losses_valid.append(losses)
    plot(losses, losses_valid)
    
    
    
X_train_reduced = cnn_modelisor.get_hidden_representation(torch.tensor(X_train, dtype=torch.long))


#------------------------------------------------------------------------------
#Utils


def plot(losses, losses_valid):
    import matplotlib.pyplot as plt
    
    plt.plot(losses, label="train loss")
    plt.plot(losses_valid, label="valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    