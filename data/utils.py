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
import pandas as pd


from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


#A custom tokenizer that prepocess and tokenize tweets
class TwitterPreprocessorTokenizer():
    
    def __init__(self, stem=True, stopwords=True): 
      
        self.stopwords =  set(sw.words('english')) if stopwords else set()
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

def get_twitter_dataset(path):
    
    #Getting data from first file, 30% offensive
    df1 = pd.read_csv(path, sep = "\t" )
    pos = df1[ df1["target"] == 1 ]
    neg = df1[ df1["target"] == 0 ]
    
    df1 = pd.concat([ pos, neg.iloc[:len(pos),] ])
    
    return df1

#Transform texts in sequence of tokens' indexs in vocab
def text_to_sequence(texts, tokenizer, vocab):
        
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
def text_to_bow(texts, tokenizer):
  
  vectorizer = TfidfVectorizer(tokenizer, strip_accents='unicode', use_idf=True)
  
  return vectorizer.fit_transform(texts)
  

  