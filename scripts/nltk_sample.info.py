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

import pandas 
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.tokenize import WhitespaceTokenizer
import time
import os
import csv 
import matplotlib.pyplot as plt
from nltk import ngrams

home = os.path.expanduser( '~' )

file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/sample.info_10000.txt"
out_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/"
file1 = open(file_path, 'r')
lines = file1.readlines()
file1.close()


###
my_punctuation_except_dash = punctuation.replace("-", "")

def remove_digit_strings(foo):
  foo1 = [x for x in foo if not any(x1.isdigit() for x1 in x)]
  return(foo1)

lemmatizer = nltk.WordNetLemmatizer()
is_noun = lambda pos: pos[:2] == 'NN'

stops = set(stopwords.words('english'))
###


###
start = time.time()

bean_bag=[]
counter=0


# add: if directories exist, delete them. 


for line in lines: 
    
    # if no line break, then it's the same sample. 
    if line!='\n':      
        
        # if line reports sample name, grab sample name: 
        if line.startswith('>'):
            sample_name=line.replace('>', '').strip()
            
            # create dir based on 3 last characters of sample name: 
            here=sample_name
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
            
            #subsitute _ with white space
            line = line.replace('_', ' ') 
            
            # get rid of punctuation except - 
            line = line.translate(str.maketrans("", "", my_punctuation_except_dash))

            # tokenization: 
            line=WhitespaceTokenizer().tokenize(line) # It does not tokenize on punctuation.

            # chuck if (alpha)numeric
            #if len(line)<=2:      # because usually sample names are lone-standing 
            line=remove_digit_strings(line)             
            
            new_line=[]
            for x in line: 
                #print(x)
                x = lemmatizer.lemmatize(x) # nouns: plural to singular
                x = lemmatizer.lemmatize(x,'v') # verbs to infinitive form. Other options: 'a' adjectives, 'r' adverbs, 's' satellite adjectives 
                x=x.lower()
                
                if x.startswith('http') or x in stops:
                    x=''

                elif len(x)>0:
                    
                    # add each word to bean bag
                    bean_bag.append(x)
                    
                    # concatenate back into a line
                    new_line.append(x)
                    
                    
                    
            sentence=' '.join(new_line)

            # make n-grams 
            bigrams = ngrams(sentence.split(), 2)
            trigrams = ngrams(sentence.split(), 3)
            quadrigrams = ngrams(sentence.split(), 4)
            
            
            

            my_bi=[]
            for grams in bigrams:
                my_bi.append(grams)
            name_of_file=this_dir+'/'+sample_name+'_bigrams.txt'
            with open(name_of_file,'a') as f:
                f.write("%r\n" %str(my_bi))
            
            my_tri=[]
            for grams in trigrams:
                my_tri.append(grams)
            name_of_file=this_dir+'/'+sample_name+'_trigrams.txt'
            with open(name_of_file,'a') as f:
                f.write("%r\n" %str(my_tri))
            
            my_quadri=[]
            for grams in quadrigrams:
                my_quadri.append(grams)
            name_of_file=this_dir+'/'+sample_name+'_quadrigrams.txt'
            with open(name_of_file,'a') as f:
                f.write("%r\n" %str(my_quadri))
                
                


    else: # if there is a line break it's a new sample, add counter 

        counter+=1
        #bean_bag=[]

end = time.time()
###
  
# save all words:    
name_of_file=out_path+'bean_bag.txt'
with open(name_of_file,'w') as f:
    f.write(str(bean_bag))



print('number of samples:',counter) # 1325
print("Time elapsed: ", end-start)   # 9.735403060913086

print('hours to run on all samples would be: ', (2000000*(end-start)/counter)/60)
print('days to run on all samples would be: ', (2000000*(end-start)/counter)/60/24)





# # Or you could just extract nouns!
# line = [word for (word, pos) in nltk.pos_tag(line) if is_noun(pos)] 
# nouns = [word for word, pos in nltk.pos_tag(word_tokenize(word)) if pos.startswith('N')]

# remove words that are present in black list:
                
    

#####
# create black list based on this : 

    



##### Frequency distribution 
from nltk.probability import FreqDist

fdist=FreqDist(bean_bag).most_common()
fdist



pandas.DataFrame(fdist, columns=['word', 'count']).to_csv(out_path+'bean_bag_freq.txt', index=False)



fdist=FreqDist(bean_bag)
##### Plot the Freq distribution
fdist.plot(30,cumulative=False)
#####





    


    
    
    
    
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












