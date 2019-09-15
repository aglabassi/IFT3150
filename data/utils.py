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
  