#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:30:20 2019

@author: Abdel

Contains utils for representing text inputs and for training the CNN

"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


#Transform texts in sequence of tokens' indexs in vocab
def raw_texts_to_sequences(texts, tokenizer, vocab):
        
        tkzd_texts = [ tokenizer(text) for text in texts ]
        
        def get_idxs(tkzd_text):
            sequence = []
            for word in tkzd_text:
                try:
                    sequence.append(vocab[word].index)
                except:
                    ''                
            return sequence
        
        #Getting the sequences
        sequences = [ get_idxs(tkzd_text) for tkzd_text in tkzd_texts ]
           
        #Padding for getting a square matrix
        maxlen = max([len(x) for x in sequences])
        X = pad_sequences(sequences, padding='post', maxlen=maxlen)
        
        return X
      


#Given texts, returns it bag-of-words representation, using idf    
def BoW(texts, tokenizer):
  
  vectorizer = TfidfVectorizer(tokenizer, strip_accents='unicode', use_idf=True)
  
  return vectorizer.fit_transform(texts)
  
      
      
                    

