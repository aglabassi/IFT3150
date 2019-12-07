#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:00:48 2019

The modelisor is a neural net that learns a dense representation of sentences. 
It is trained in a unsepervised way, using the binary code (provided by the embeddor)
in the output layer.

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

load_src("data_utils", PATH+"data/utils.py" )
load_src("embeddor", PATH+"unsupervised/embeddor/BinaryEmbeddor.py" )
load_src("modelisor", PATH+"unsupervised/modelisor/modelisor.py" )
load_src("modelisor_utils", PATH+"unsupervised/modelisor/utils.py" )


#------------------------------------------------------------------------------
#Load data
import numpy as np
import pandas as pd

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

from data_utils import TwitterPreprocessorTokenizer, text_to_sequence, text_to_bow

BIG_SENTENCE_LENGTH = 111 #DONT MODIFY

tokenizer = TwitterPreprocessorTokenizer(stem=False,remove_stopwords=False)
word_idxs = text_to_sequence(sentences, tokenizer, vocab, maxlen_absolute=BIG_SENTENCE_LENGTH)
bows = text_to_bow(sentences, tokenizer)


#------------------------------------------------------------------------------
#Get the binary low dim sentence representation (embedded sentences)

USE_PRETRAIN_BIN_EMBS = True
BIN_DIM = 20

bin_emb_path = PATH+'unsupervised/embeddor/bin_embs_'+str(BIN_DIM)+'d.csv'

if USE_PRETRAIN_BIN_EMBS:  
    bin_embs = pd.read_csv(PATH+bin_emb_path, sep=',').values
else:
    import time
    from embeddor import BinaryEmbeddor
    t = time.time()
    bin_embs = BinaryEmbeddor(bin_dim=BIN_DIM)(bows.todense())
    pd.DataFrame(bin_embs).to_csv(PATH+bin_emb_path, index=False)
    t = time.time() - t
    print("binary embeddings trained get after ", t, "seconds for n_inputs= ", len(sentences))



#------------------------------------------------------------------------------
#Simple train / valid split
    
TRAIN_SIZE = 0.8
DEL = int(len(word_idxs)*TRAIN_SIZE)

X_train, bin_train = word_idxs[:DEL], bin_embs[:DEL]
X_test, bin_test = word_idxs[DEL:], bin_embs[DEL:]


#------------------------------------------------------------------------------
#Training the CNN modelisor to get lowdim sentence modelisation


import torch
import torch.optim as optim
from torch.nn import BCELoss

from modelisor import Automodelisor
from modelisor_utils import train_model


#Hyperparameters
EPOCHS = 10
BATCH_SIZE = 128
K_TOP = 3
LEARNING_RATE = 0.01


#initalise the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

automodelisor = Automodelisor(vocab_size=len(vocab), 
                    sentence_length=BIG_SENTENCE_LENGTH,
                    emb_weights=torch.FloatTensor(model_we.vectors),  
                    k_top=K_TOP,
                    bin_dim=BIN_DIM)

automodelisor.to(device)

criterion = BCELoss()
optimizer = optim.Adam(automodelisor.parameters(), lr=LEARNING_RATE)

#training the model
losses, losses_valid = [],[]
for epoch in range(EPOCHS):
    loss, loss_valid = train_model(automodelisor, X_train, bin_train, X_test, bin_test, 
                    criterion, optimizer, device, batch_size=BATCH_SIZE)
    print(loss)
    losses.append(loss)
    losses_valid.append(losses)
    
    
#------------------------------------------------------------------------------
#get the sentence dense representation

inputs = torch.tensor(word_idxs, dtype=torch.long)
sentences_reduced = torch.zeros(0)
sentences_reduced = automodelisor.modelisor(inputs).detach().numpy()


#------------------------------------------------------------------------------
#Clustering

from sklearn.cluster import KMeans
n_classes = np.unique(labels).shape[0]
clustering = KMeans(n_clusters=n_classes, 
                                assign_labels="discretize", random_state=0).fit(sentences_reduced)

results = clustering.labels_
np.savetxt(PATH+"unsupervised/results.out", results)

