#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:29:17 2019

@author: Abdel
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding,Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from tensorflow.keras.initializers import Constant

 # inspired from "Convolutional Neural Networks for Sentence Classification" 
 #(Yoon Kim), https://arxiv.org/pdf/1408.5882.pdf
class CNN_classifier():
    def __init__(self, embedding_weights, sequence_length,
                 drop=0.5, n_filter=256, filter_sizes=[2,3,4]):
        
        vocab_size = embedding_weights.shape[0]
        embedding_dim = embedding_weights.shape[1]
        
        inputs = Input(shape=(sequence_length,), dtype='int32')
        
        embedding_layer = Embedding(vocab_size,
                                    embedding_dim,
                                    embeddings_initializer=Constant(embedding_weights),
                                    input_length=sequence_length,
                                    trainable=False)
        
        embedding = embedding_layer(inputs)
        
        reshape = Reshape((sequence_length,embedding_dim ,1))(embedding)
        
        convs = []
        for filter_size in filter_sizes: 
            convs.append( Conv2D(n_filter, 
                                 kernel_size=(filter_size,embedding_dim),
                                 padding='valid',
                                 kernel_initializer='normal', 
                                 activation='relu')(reshape))
        maxpools = []
        for i,filter_size in enumerate(filter_sizes):
            maxpools.append( MaxPool2D(pool_size=(sequence_length - filter_size + 1, 1),
                                       strides=(1,1), 
                                       padding='valid')(convs[i]) )
        
        
        concatenated_tensor = Concatenate(axis=1)(maxpools)
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=1, activation='sigmoid')(dropout)
        
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
     
    
    def __str__(self):
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)
    
    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def fit(self, X_train, y_train, batch_size=64, epochs=10, validation_data=None):
        self.model.fit(X_train, y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data)
    
    def predict(self, X_test, batch_size=None):
        return self.model.predict(X_test, batch_size=batch_size)