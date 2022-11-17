#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:44:28 2022
@author: dgaio
"""


import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.tokenize import WhitespaceTokenizer

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
home = os.path.expanduser( '~' )

file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/sample.info_50000.txt"
file1 = open(file_path, 'r')
lines = file1.readlines()
file1.close()



def remove_digit_strings(foo):
  foo1 = [x for x in foo if not any(x1.isdigit() for x1 in x)]
  return(foo1)

my_punctuation_except_dash = punctuation.replace("-", "")

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


start = time.time()

bean_bag=[]
counter=0

for line in lines: 
    
    if line=='\n':
        #print('saving to file and deleting from memory')
        counter+=1
        
    else: 
        
        # grab sample name: 
        if line.startswith('>'):
            sample_name=line.replace('>', '').strip()
            #print(line)
            
        else: 
            # get rid of end of line char
            line = line.strip()
            
            # get rid of all left string up to =
            line= line.split('=', 1)[-1]
            
            # subsitute _ with white space  (FIRST!)
            line = line. replace('_', ' ') 
            
            # get rid of punctuation except - and _
            #re.sub(pattern, "", line_2) 
            line = line.translate(str.maketrans("", "", my_punctuation_except_dash))
        
            
            #TOKENIZATION: 
            line=WhitespaceTokenizer().tokenize(line) # It does not tokenize on punctuation.
                
            # remove url-addresses: 
            line = [x for x in line if not x.startswith('http')]
            
            # chuck if alpha-numeric
            if len(line)<=2:      # because usually sample names are lone-standing 
                line=remove_digit_strings(line)   

            # remove stopwords
            line = [word for word in line if word.lower() not in stopwords.words('english')]

            # Lemmatization
            lemmatizer = nltk.WordNetLemmatizer()
            line = [lemmatizer.lemmatize(t,'v') for t in line] # verbs: tense to infinitive
            line = [lemmatizer.lemmatize(t,'n') for t in line] # nouns: plural to singular
            line = [lemmatizer.lemmatize(t,'s') for t in line] # satellite adj
            line = [lemmatizer.lemmatize(t,'a') for t in line] # adjectives
            line = [lemmatizer.lemmatize(t,'r') for t in line] # adverbs
            
            
            # # Or you could just extract nouns!
            # is_noun = lambda pos: pos[:2] == 'NN'
            # line = [word for (word, pos) in nltk.pos_tag(line) if is_noun(pos)] 
            #nouns = [word for word, pos in nltk.pos_tag(word_tokenize(word)) if pos.startswith('N')]
            
        
            
            
            for item in line: 
                if len(line)>0:
                    bean_bag.append(item.lower())
            
            # remove words that are present in black list:
        
                
        
end = time.time()
print('number of samples:',counter) # 1325
print("Time elapsed: ", end-start)   # 9.735403060913086

print('hours to run on all samples would be: ', (2000000*(end-start)/counter)/60)
print('days to run on all samples would be: ', (2000000*(end-start)/counter)/60/24)


print(bean_bag)
    

#####
# create black list based on this : 

##### Frequency distribution 
from nltk.probability import FreqDist

fdist=FreqDist(bean_bag)
fdist

##### Plot the Freq distribution
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
#####

















    
# =============================================================================
#     
#     
# from collections import Counter
# out = Counter(bean_bag)
# print(out)
# 
#         
# 
# with open('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/words_occurrence.txt', 'w') as f:
#     for k,v in  out.most_common():
#         f.write( "{} {}\n".format(k,v) )
# =============================================================================


    
    
    
    
# =============================================================================
# my_list=[]
# one_mers_list=[]
# two_mers_list=[]
# three_mers_list=[]
#    
# for g in refined_list:
#     one_mers=one_mers_list.append(g)
# 
# for g,k in enumerate(refined_list): 
#         
#     if g+1 < len(refined_list):
#         
#         two_mers=refined_list[g]+' '+ refined_list[g+1]
#         two_mers_list.append(two_mers)
#         
#     if g+2 < len(refined_list):
#         
#         three_mers=refined_list[g]+' '+refined_list[g+1]+' '+refined_list[g+2]
#         three_mers_list.append(three_mers)
#         
#     #else: print(two_mers, three_mers) # not sure why there is something getting out of the loop here, but it doesn't seem so! 
# 
# 
# 
# # map each list against ontologies
#     
#     
# print(one_mers_list)
# print(two_mers_list)
# print(three_mers_list)
# =============================================================================

# =============================================================================
# # test code to create n-grams: 
# two_mers_list=[]
# three_mers_list=[]
# 
# my_list=[['dog','cat','lion','tiger','yyy','zzz'], ['dog2','cat2','lion2','tiger2','yyy2','zzz2']]
# 
# for s in my_list: 
#     for g,k in enumerate(s): 
#         
#         if g+1 < len(s):
#             
#             print('ok')
#             two_mers=s[g]+' '+ s[g+1]
#             two_mers_list.append(two_mers)
#             
#         if g+2 < len(s):
#             
#             print('okk')
#             three_mers=s[g]+' '+s[g+1]+' '+s[g+2]
#             three_mers_list.append(three_mers)
#             
#         else: print(two_mers, three_mers) # not sure why there is something getting out of the loop here, but it doesn't seem so! 
# 
# 
# print(two_mers_list)
# print(three_mers_list)
# =============================================================================












