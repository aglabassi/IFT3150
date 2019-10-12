#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:30:20 2019

@author: Abdel Ghani
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#A convolutional neural network for text modelisation
class CNN(nn.Module):
    
    def __init__(self, vocab_size, sentence_length, emb_weights=None, n_kernel1=4, n_kernel2=4, size1=1, size2=2, k_top=3):
        
        super(CNN, self).__init__()          
        
        #If emb weigths are provided, load it, else init a new embeding layer
        embedding_dim = emb_weights.shape[1] if emb_weights is not None else 50
        try:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.embedding.weight.data.copy_(emb_weights)
        except:
            ''
         
        #inputs of format (N,1, embedding_dim, sentence_len),
        #output of format (N,n_kernel,embedding_dim, seentence_len+size-1)
        self.conv1 = nn.Conv2d(1, n_kernel1, (1,size1), padding=(0, size1-1))
        
        self.conv2 = nn.Conv2d(n_kernel1, n_kernel2, (1,size2), padding=(0,size2-1))
    
        self.linear = nn.Linear(k_top*embedding_dim,10)
        
        self.dynamic_pooler = DynamicKMaxPool(k_top, sentence_length, n_conv=2)
      
    
    def forward(self,x):
        hidden_rep = self.get_hidden_representation(x)
        
        #reshape to vector format
        hidden_rep = hidden_rep.view(hidden_rep.shape[0],hidden_rep.shape[1]*hidden_rep.shape[2])
        
        return F.softmax(self.linear(hidden_rep))
    
    
    #return a emb/2 * k_top shaped matrix giving a reduced-dimension representation of a sentence (emb*len(s))
    def get_hidden_representation(self,x):        
        embedded = torch.unsqueeze(self.embedding(x), 1)
        embedded = torch.transpose(embedded, 2, 3)
                
        conved1 = F.relu(self.conv1(embedded).squeeze(3))
        
        pooled1 = self.dynamic_pooler(conved1, 1, embedded.shape[2])
        
        conved2 = F.relu(self.conv2(pooled1))

        #fold(conved2)
        
        pooled2 = self.dynamic_pooler(conved2,2,embedded.shape[2])
        
        #We get rid of the first dimension : kernel dimension
        pooled2 = F.adaptive_max_pool2d(pooled2.transpose(1,2), (1,pooled2.shape[3])).squeeze()
        
        return pooled2
        
        


#inspired from https://arxiv.org/pdf/1404.2188.pdf
class DynamicKMaxPool(nn.Module):
    
    def __init__(self, k_top, sentence_length, n_conv):
        super().__init__()
        self.k_top = k_top
        self.sentence_length = sentence_length
        self.n_conv = n_conv
        
    
    def forward(self, X, conv_idx, heigth):
        temp = ((self.n_conv - conv_idx) / self.n_conv) * self.sentence_length
        k_l = int(np.round( max(self.k_top, np.ceil(temp) )))
        
        return F.adaptive_max_pool2d(X, (heigth,k_l))
    


batch = torch.randint(0,433,(32,70))
cnn = CNN(433,70)
cnn(batch)      
