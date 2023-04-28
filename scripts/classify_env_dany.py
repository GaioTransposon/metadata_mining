#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:33:38 2023

@author: dgaio
"""


import time
import os
#import re
#import glob
#import csv 
#import matplotlib.pyplot as plt
#import nltk
import spacy
from fuzzywuzzy import fuzz
#import numpy as np
#import pandas as pd

#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.tokenize import WhitespaceTokenizer
#from nltk import ngrams
#from nltk.stem import WordNetLemmatizer


home=os.path.expanduser('~')
mypath=home+"/cloudstor/Gaio/MicrobeAtlasProject/"     


##########################################################################

# Read in glossaries and store into dictionary:

glossary_dic = {}   
# list files 
for file in os.listdir(mypath):
    if file.startswith("glossary"):

        # extract name of biome
        s=file
        s=s.replace('_', ' ').replace('.', ' ').split()[1:-1]
        biome='_'.join(s)
        #print(biome)
        
        path_to_file=mypath+file
        with open(path_to_file, 'r') as f:
            
            my_list=[]
            
            for line in f:
                # Remove linebreak which is the last character of the string
                curr_place = line[:-1]
                # Add item to the list
                my_list.append(curr_place)
            
        glossary_dic[biome]=my_list
        
        
##########################################################################

# Function which reads in a string and fuzzy matches it against glossaries. 
# It returns for each glossary, the keywords it found and its score. 

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')


# Define a function to classify the environment of a given text based on the presence of specific keywords
def classify_environment(text):
    
    # Process the text with spaCy
    doc = nlp(text)
    
    token_list=[]
    dicc={}
    scores=[]
    my_max=0
    
    # Compute a confidence score for each environment based on the fuzzy string matching between keywords and text tokens
    for k, v in glossary_dic.items():
        
        x_list=[]
        
        for term in v:
            
            for token in doc: 

                x=fuzz.ratio(term,token.text.lower())
                if x>= 80: 
                    #print(term, token, x)
                    x_list.append(x)
                    token_list.append(token)
              
        dicc['tokens']=token_list
        dicc[k]=len(x_list)
        
        scores.append(len(x_list))
        
        if my_max < len(x_list):
            my_max=len(x_list)
        
        
    if sorted(scores)[-2:][0]==sorted(scores)[-2:][1]:
        dicc['biome']="unknown"
        
            
    else: 
        dicc['biome']=k
        

    return dicc

##########################################################################


# Test the function with the example text
mytext = "These samples were retrieved from a lake in Colorado in June 2009. They were processed and sequenced using Liquid metagenomic extraction kit, and analysed. The results are relevant for applications in human."
print(classify_environment(mytext)) # Output: "water"


##########################################################################

# Run classification on metadata: 

home = os.path.expanduser( '~' )

file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/sample_info_clean.csv"
out_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/"
file1 = open(file_path, 'r')
lines = file1.readlines()
file1.close()



start = time.time()
my_dic={}
counter=0
for i in lines[0:1000]: 
    counter+=1
    sample_name = i.split(',')[0]
    my_list=[]
    
    text = i.split(',')[1]
    
    env = classify_environment(text)
    #print(counter,sample_name, env)



executionTime = (time.time() - start)
print('Execution time in seconds: ' + str(executionTime))
print('Execution time in minutes: ' + str(executionTime/60)) 

print('Execution time per sample (sec): ' + str(executionTime/counter))
print('Execution time for 2M samples (min): ' + str((executionTime/60)/counter*2000000))
print('Execution time for 2M samples (h): ' + str((executionTime/60/60)/counter*2000000))

    





