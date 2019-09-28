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


#A convolutional neural network for text classification 
class CNN(nn.Module):
    
    def __init__(self, vocab_size, sentence_length, emb_weights=None, n_kernel1=4, n_kernel2=4, window_sizes1=[3], size2=2, k_top=3):
        
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
        self.convolutionners1 = nn.ModuleList([ nn.Conv2d(1, n_kernel1, (1,size), padding=(0, size-1) ) for size in window_sizes1  ])
        
        self.pooler = DynamicKMaxPooling(k_top,sentence_length, 2)
        
        self.convonlutionner2 = nn.Conv2d(n_kernel1, n_kernel2, (1,size2), padding=(0,size2-1))
        
        self.linear = nn.Linear(len(window_sizes1)*n_kernel2*k_top*embedding_dim,10)
      
        
    
    def forward(self,x):
        return F.softmax(self.linear(self.get_hidden_representation(x)))
    
    
    def get_hidden_representation(self,x):        
        embedded = torch.unsqueeze(self.embedding(x), 1)
        embedded = torch.transpose(embedded, 2,3)
                
        conveds1 = [F.relu(convolutionner(embedded).squeeze(3)) for convolutionner in self.convolutionners1 ]
        
        pooleds1 = [ self.pooler(conved, 1, embedded.shape[2])  for conved in conveds1 ]
        
        conveds2 = [ F.relu(self.convonlutionner2(pooled)) for pooled in pooleds1 ]
        
        #fold(conveds2)
        
        pooleds2 = [ self.pooler(conved,2,embedded.shape[2]) for conved in conveds2 ]
        
        cated = torch.cat(pooleds2, dim=1)
        
        cated = torch.transpose(cated, 2,3).contiguous()
        
        dimensions = cated.shape[0], cated.shape[1]*cated.shape[2]*cated.shape[3]
        
        cated = cated.view(dimensions[0], dimensions[1])
        
        print(cated.shape)
        
        return cated
        
        


# Based on https://arxiv.org/pdf/1404.2188.pdf
# Based on Anton Melnikov implementation
class DynamicKMaxPooling(nn.Module):
    
    def __init__(self, k_top, sentence_length, n_conv_layers):
        super().__init__()
        # "L is the total  number  of  convolutional  layers
        # in  the  network;
        # ktop is the fixed pooling parameter for the
        # topmost  convolutional  layer" 
        self.k_top = k_top
        self.sentence_length = sentence_length
        self.n_conv_layers = n_conv_layers
        
    
    def forward(self, X, layer_idx, heigth):
        # l is the current convolutional layer
        # X is the input sequence
        # s is the length of the sequence
        # (for conv layers, the length dimension is last)
        k_ll = ((self.n_conv_layers - layer_idx) / self.n_conv_layers) * self.sentence_length
        k_l = round(max(self.k_top, np.ceil(k_ll)))
        out = F.adaptive_max_pool2d(X, (heigth,int(k_l)))
        return out
    
 
    
batch = torch.randint(0,433,(32,70))
cnn = CNN(500,70)
cnn(batch)
    
    

        
        
       
