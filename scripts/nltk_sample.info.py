#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:44:28 2022
@author: dgaio
"""

#from nltk.tokenize import word_tokenize # Passing the string text into word tokenize for breaking the sentences
#nltk.download('punkt')
#from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#from nltk.stem import PorterStemmer
#from nltk import word_tokenize
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')

import time
import os
import re
import pandas as pd
#import csv 
#import matplotlib.pyplot as plt
import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.tokenize import WhitespaceTokenizer
#from nltk import ngrams
#from nltk.stem import WordNetLemmatizer


home = os.path.expanduser( '~' )

file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/sample.info_4000000"
out_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/"
file1 = open(file_path, 'r')
lines = file1.readlines()
file1.close()
  
###


###
start = time.time()

counter=0
bean_bag=[]
# add: if directories exist, delete them. 

# open blacklist, sort it, and save it as list
with open(out_path+"/black_list.txt", 'r') as file:
    black = file.readlines()
black = sorted(set([line.rstrip() for line in black]))

char_remov=['.', '%', '_', '[', ']', '/','=']
mydic={}
for line in lines: #2000:2054
    
    # if no line break, then it's the same sample. 
    if line!='\n':     
        
        
        
        # if line reports sample name, grab sample name: 
        if line.startswith('>'):
            sample_name=line.replace('>', '').strip()
            #print('\n##########', sample_name)
            # create dir based on 3 last characters of sample name: 
            here=sample_name
            
            mydic[sample_name]=[]
            keywords_list=[]

            
            last_3 = here[-3:]
            this_dir=out_path+'dir_'+last_3
            # if dir doesn't already exist make it 
            try:
                os.makedirs(this_dir)
            except FileExistsError:
                # directory already exists
                pass
            
        # any other line is the metadata content of the sample. Parse it: 
        else: 
            
            # get rid of end of line char
            line = line.strip()

            # get rid of all left string up to =
            line= line.split('=', 1)[-1]
            
            #subsitute defined characters with white space
            for char in char_remov:
                line = line.replace(char, " ")
            
            # remove all https: 
            line = re.sub(r'http\S+', '', line, flags = re.MULTILINE) #flags=re.MULT.iILINE)
   
            words = nltk.word_tokenize(line)
            #print(words)

                
            tagged = nltk.pos_tag(words)
            
            for k, pos in tagged: 
                #print(k)
                
                # percentage of numbers in word: 
                perc=float(len(''.join(re.findall('\d',k))))/float(len(k))
                
                #print(k, perc)
                if perc < 0.30 and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'JJ'):
                    k = nltk.WordNetLemmatizer().lemmatize(k) # nouns: plural to singular
                    k = k.lower()
                    
                    # add If not in black list:
                    if k not in black:
                        keywords_list.append(k)
                        bean_bag.append(k)
                        
        mydic[sample_name]=' '.join(keywords_list)
        
    else: # if there is a line break it's a new sample, add counter 

        counter+=1
        #print(sample_name)
        
        
        
# df = pd.DataFrame(mydic.items(), columns=['sample', 'metadata'])       
# write to file
pd.DataFrame(mydic.items(), columns=['sample', 'metadata']).to_csv(out_path+'sample_info_clean.csv', index=False)

        
executionTime = (time.time() - start)
print('Execution time in seconds: ' + str(executionTime))
print('Execution time in minutes: ' + str(executionTime/60)) 

print('Execution time per sample (sec): ' + str(executionTime/counter))
print('Execution time for 2M samples (min): ' + str((executionTime/60)/counter*2000000))
print('Execution time for 2M samples (h): ' + str((executionTime/60/60)/counter*2000000))


    
# # potentially useful pieces of code: 
# line=WhitespaceTokenizer().tokenize(line) # It does not tokenize on punctuation. 
# line=re.split('[(?!:=.,;"")]', line)     
# m=generate_ngrams(text=ii, WordsToCombine=1)
# b=generate_ngrams(text=ii, WordsToCombine=2)

# # to sort and eliminate duplicates from list: 
# MONO = sorted(set(flat_mono))

# # to unlist nested lists
# from itertools import chain
# list(chain.from_iterable(list(chain.from_iterable(keywords_list))))
                    
## get rid of punctuation except - 
#line = line.translate(str.maketrans("", "", my_punctuation_except_dash))
        
# # chuck if (alpha)numeric
# #if len(line)<=2:      # because usually sample names are lone-standing 
# line=remove_digit_strings(line)  

# # lemmatize
# for x in line: 
#     x = lemmatizer.lemmatize(x) # nouns: plural to singular
#     x = lemmatizer.lemmatize(x,'v') # verbs to infinitive form. Other options: 'a' adjectives, 'r' adverbs, 's' satellite adjectives 
#     x=x.lower()
    
# # Or you could just extract nouns!
# line = [word for (word, pos) in nltk.pos_tag(line) if is_noun(pos)] 
# nouns = [word for word, pos in nltk.pos_tag(word_tokenize(word)) if pos.startswith('N')]

# import spacy
# # using token-typing
# text = lines[2850:2900]
# nlp = spacy.load("en_core_web_sm")
# for l in text: 
#     doc = nlp(l)
#     keywords = [token.text for token in doc if token.is_alpha and token.ent_type_ == "PRODUCT"]




##### Frequency distribution 
from nltk.probability import FreqDist

fdist=FreqDist(bean_bag).most_common()

# write to file
pd.DataFrame(fdist, columns=['word', 'count']).to_csv(out_path+'bean_bag_freq.txt', index=False)

fdist=FreqDist(bean_bag)
##### Plot the Freq distribution
fdist.plot(30,cumulative=False)
#####





    












