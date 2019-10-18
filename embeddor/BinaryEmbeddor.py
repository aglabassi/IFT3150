#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:28:19 2019

@author: Abdel

Contains utilities for getting the B embeddings, that is, Binary Neighbor Embeddings
of low dimensionnality


"""

from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
import numpy as np

class BinaryEmbeddor():
  
    def __init__(self, bin_dim=10, n_neighbors=300, rbf_cst=1):
        similarity_calculator = BinaryEmbeddor._get_similarity_calculator(n_neighbors, rbf_cst)
        self.embedding = SpectralEmbedding(n_components=bin_dim, affinity=similarity_calculator)
        
    def __call__(self,X):
        unormalized_binary_embs = self.embedding.fit_transform(X)
        return (np.sign(unormalized_binary_embs)+1)/2    
        
    def _get_similarity_calculator(n_neighbors, rbf_cst):
        def similarity_calculator(X):
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
            _, neighbors = neighbors.kneighbors(X)
            
            n_samples = X.shape[0]
            similarities = np.zeros((n_samples,n_samples))
            
            for i in range(n_samples):
                for j in range(i,n_samples):
                    if j in neighbors[i]: #Reduce running time of eigenmap
                            norm = np.linalg.norm((X[i]-X[j]))
                            similarities[i,j] = similarities[j,i] = np.exp(-norm/(2*(rbf_cst**2)))
                            
            return similarities
        
        return similarity_calculator
                    
            
            
            
            
      
      
      
  
  
     
     
    

