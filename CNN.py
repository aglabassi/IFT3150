#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:30:20 2019

@author: Abdel Ghani
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


#A convolutional neural network for text classification (offensive language detection)
class CNN4old(nn.Module):
    
    def __init__(self, vocab_size, emb_weights=None):
        
        super(CNN4old, self).__init__()          
        
        #If emb weigths are provided, load it, else init a new embeding layer
        embedding_dim = emb_weights.shape[1] if emb_weights is not None else 50
        try:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.embedding.weight.data.copy_(emb_weights)
        except:
            ''
            
            
      
        
    
    def forward(self,x):
        
        embedded = torch.unsqueeze(self.embedding(x) , 1)
        
        conveds= [F.relu(self.drop1(conv(embedded).squeeze(3))) for conv in self.convs]
                
        pooleds = [F.max_pool1d(conved, conved.shape[2]).squeeze(2) for conved in conveds]
        
        cated = torch.cat(pooleds, dim=1)
        
        cated = self.drop2(cated)
            
        return torch.sigmoid(self.linear(cated))
    
    


        
        
       
