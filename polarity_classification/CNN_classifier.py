#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:29:17 2019

@author: Abdel
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import Constant


 # Deep CNN (diegoschapira)
class CNN_clf():
    def __init__(self, embedding_weigths, sequence_length):
        model = tf.keras.Sequential()
        
        model.add(layers.Embedding(embedding_weigths.shape[0],
                                    embedding_weigths.shape[1],
                                    embeddings_initializer=Constant(embedding_weigths),
                                    input_length=sequence_length,
                                    trainable=False))
        
        model.add(layers.Conv1D(128, 7, activation='relu',padding='same'))
        model.add(layers.MaxPooling1D())
        model.add(layers.Conv1D(256, 5, activation='relu',padding='same'))
        model.add(layers.MaxPooling1D())
        model.add(layers.Conv1D(512, 3, activation='relu',padding='same'))
        model.add(layers.MaxPooling1D())
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        self.model = model
    
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