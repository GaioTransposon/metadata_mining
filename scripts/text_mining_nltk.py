#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:25:22 2022
@author: dgaio
"""

# Importing necessary library
import pandas as pd
import numpy as np
# installed ntkl in terminal: 
# conda install -n spyder_env -c anaconda nltk 
import nltk
import os
import nltk.corpus # sample text for performing tokenization

text = """In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern
side of South America"# importing word_tokenize from nltk"""

from nltk.tokenize import word_tokenize # Passing the string text into word tokenize for breaking the sentences
#nltk.download('punkt')
token = word_tokenize(text)
print(token)


# finding the frequency distinct in the tokens
# Importing FreqDist library from nltk and passing token into FreqDist
from nltk.probability import FreqDist
fdist = FreqDist(token)
fdist


# To find the frequency of top 10 words
fdist1 = fdist.most_common(10)
fdist1


# Stemming 
# =============================================================================
# # Importing Porterstemmer from nltk library
# # Checking for the word ‘giving’ 
# from nltk.stem import PorterStemmer
# pst = PorterStemmer()
# pst.stem("waiting")
# 
# # Checking for the list of words
# stm = ["waited", "waiting", "waits"]
# for word in stm :
#    print(word+ ":" +pst.stem(word))
# =============================================================================
# # Importing LancasterStemmer from nltk (Lancaster is more aggressive than Porter)
# from nltk.stem import LancasterStemmer
# lst = LancasterStemmer()
# stm = ["giving", "given", "given", "gave"]
# for word in stm :
#  print(word+ ":" +lst.stem(word))
# =============================================================================
# =============================================================================
# 
# # Lemmatization
# # This could be useful as ontologies will be in the singular form 
# # TO THINK ABOUT: possible issues with words that should stay in the plural form? 
#  # Importing Lemmatizer library from nltk
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer() 
# #nltk.download('wordnet')
# #nltk.download('omw-1.4')
# print("rocks :", lemmatizer.lemmatize("rocks")) 
# print("corpora :", lemmatizer.lemmatize("corpora"))
# print("women :", lemmatizer.lemmatize("women")) 
# 
# # Stop Words
# # importing stopwors from nltk library
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# #nltk.download('stopwords')
# a = set(stopwords.words('english'))
# text = """Cristiano Ronaldo was born 
# on February 5, 1985, in Funchal, Madeira, Portugal."""
# text1 = word_tokenize(text.lower())
# print("these are all my words: ", text1)
# my_not_stopwords = [x for x in text1 if x not in a]
# print("these are my words w/o stopwords: ", my_not_stopwords)
# my_stopwords = [x for x in text1 if x in a]
# print("these are my stopwords: ", my_stopwords)
# 
# 
# # Part of speech tagging (POS)
# text = """"vote to choose a particular man or a group (party) 
# to represent them in parliament"""
# # Tokenize the text
# tex = word_tokenize(text)
# #nltk.download('averaged_perceptron_tagger')
# for token in tex:
#     print(nltk.pos_tag([token]))
# 
# =============================================================================

# Named Entity Recognition (NER)
text = """Google’s CEO Sundar Pichai introduced the new Pixel 
at Minnesota Roi Centre Event”#importing chunk library from nltk"""
from nltk import ne_chunk # tokenize and POS Tagging before doing chunk
token = word_tokenize(text)
tags = nltk.pos_tag(token)
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
chunk = ne_chunk(tags)
chunk


# Could Chunking be useful instead of using N-grams? 
# Chunking: 
text = "We saw the yellow dog"
token = word_tokenize(text)
tags = nltk.pos_tag(token)
reg = "NP: {<DT>?<JJ>*<NN>}"
a = nltk.RegexpParser(reg)
result = a.parse(tags)
print(result)











