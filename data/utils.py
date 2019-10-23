#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:30:20 2019

@author: Abdel
"""

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
import re
from bs4 import BeautifulSoup


from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


#A custom tokenizer that prepocess and tokenize tweets
class TwitterPreprocessorTokenizer():
    
    def __init__(self, stem=True, remove_stopwords=True): 
      
        self.stopwords =  set(sw.words('english')) if remove_stopwords else set()
        self.stem = SnowballStemmer("english",True).stem if stem else lambda x:x
            
        
    def __call__(self, text):  
      
        cleaned_text =  TwitterPreprocessorTokenizer._clean(str(text))
        return [ self.stem(word) for word in word_tokenize(cleaned_text) if word not in self.stopwords ]
    
    
    def _clean(document):
        
        #HTML decoding
        cleaned_document = BeautifulSoup(document,'lxml').get_text()
        
        #Remove urls
        cleaned_document = re.sub('https?://[A-Za-z0-9./]+','', document)
        
        #Remove @Mention
        cleaned_document = re.sub(r'@[A-Za-z0-9]+','', cleaned_document)
        
        #UTF-8 BOM decoding
        try:
            cleaned_document = cleaned_document.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            ''
        
        #Keep just the letter and ! and ?
        cleaned_document = re.sub("[^!|?|a-zA-Z]", " ", cleaned_document)

        #To lower case        
        cleaned_document = cleaned_document.lower()
        
        return cleaned_document



#Transform texts in sequence of tokens' indexs in vocab
def text_to_sequence(texts, tokenizer, vocab, maxlen_absolute=None):
    
    def get_idxs(tkzd_text):
            sequence = []
            for word in tkzd_text:
                try:
                    sequence.append(vocab[word].index)
                except:
                    ''                
            return sequence
    
    maxlen_relative = 0
    sequences = ["0"]*texts.shape[0]
    for i,text in enumerate(texts):
        tkzd_text = tokenizer(text)
        sequence = get_idxs(tkzd_text)
        maxlen_relative = max(maxlen_relative, len(sequence))
        sequences[i] = sequence
        
    maxlen = maxlen_absolute if maxlen_absolute is not None else maxlen_relative
    X = pad_sequences(sequences, padding='post', maxlen=maxlen)
        
    return X
      


#Given texts, returns it bag-of-words representation, using idf    
def text_to_bow(texts, tokenizer):
  
  vectorizer = TfidfVectorizer(tokenizer, strip_accents='unicode', use_idf=True)
  
  return vectorizer.fit_transform(texts)
  

  