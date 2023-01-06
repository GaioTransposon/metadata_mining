#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:20:31 2022

@author: danielagaio
"""





# conda create -n nltk_env
# conda install -c anaconda nltk
# conda install -c anaconda pandas
#conda install -c conda-forge matplotlib



import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.tokenize import WhitespaceTokenizer
import time
import os
import csv 

import pandas 
#import matplotlib.pyplot as plt


# from nltk.tokenize import word_tokenize # Passing the string text into word tokenize for breaking the sentences
# nltk.download('punkt')
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# from nltk.stem import PorterStemmer
# from nltk import word_tokenize
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')



# samples=[]
# hours=[]

home = os.path.expanduser( '~' )
file="sample.info_10000"





# atlas location
# file_path = home+"/projects/MicrobeAtlas-metadata/v_current/"

# UTS HPC location
# file_path = home+"/MicrobeAtlasProject/"

# (any) local 
file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/"     



file1 = open(file_path+file, 'r')
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
len(stops)
###


###
start = time.time()

bean_bag=[]
counter=0

for line in lines[3:30]: 
    
    if line!='\n':      
        
        # grab sample name: 
        if line.startswith('>'):
            sample_name=line.replace('>', '').strip()
            
            # create dir based on 3 last sample name: 
            here=sample_name
            last_3 = here[-3:]
            # if dir doesn't already exist make it 
            if not os.path.exists(file_path+last_3):
                os.makedirs(file_path+last_3)
            name_of_file=file_path+last_3+'/'+sample_name+'.txt'
            
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
            print(line)
            for x in line: 
                x = lemmatizer.lemmatize(x) # nouns: plural to singular
                x = lemmatizer.lemmatize(x,'v') # nouns are made singular by default, other options: 'a' adjectives, 'r' adverbs, 's' satellite adjectives 
                print(x)

                if x.lower().startswith('http') or x.lower() in stops:
                    x=''

                if len(x)>2:
                    bean_bag.append(x.lower())

    else: 

        counter+=1

end = time.time()
###
   
# save all words:    
name_of_file=file_path+'bean_bag_all.txt'
with open(name_of_file,'w') as f:
    f.write(str(bean_bag))



print('number of samples:',counter) # 1325
print("Time elapsed: ", end-start)   # 9.735403060913086

print('minutes to run on all samples would be: ', (2000000*(end-start)/counter)/60)
print('days to run on all samples would be: ', (2000000*(end-start)/counter)/60/24)




# #samples=[]
# #seconds=[]

# samples.append(counter)
# seconds.append((end-start))


# print(samples)
# print(seconds)



##### Frequency distribution 
from nltk.probability import FreqDist

fdist=FreqDist(bean_bag).most_common()
fdist


pandas.DataFrame(fdist, columns=['word', 'count']).to_csv(file_path+'bean_bag_freq.txt', index=False)



fdist=FreqDist(bean_bag)
##### Plot the Freq distribution
fdist.plot(60,cumulative=False)
#####




# =============================================================================
# ###
# 
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit
# 
# 
# def exponenial_func(x, a, b, c):
#     return a*np.exp(b*x)+c
# 
# samples2=[]
# for i in samples: 
#     print(i)
#     ii=float(i)
#     samples2.append(ii)
# 
# x = np.array(seconds)
# y = np.array(samples)
# 
# ###
# 
# # Based on the samples (max 100000) we tried on: 
# popt, pcov = curve_fit(exponenial_func, x, y, p0=(1, 1e-6, 1))
# 
# xx = np.linspace(0, 80, 100)
# yy = exponenial_func(xx, *popt)
# 
# r2 = 1. - sum((exponenial_func(x, *popt) - y) ** 2) / sum((y - np.mean(y)) ** 2)
# 
# plt.plot(x, y, 'o', xx, yy)
# plt.title('time to process 2M samples-metadata')
# plt.xlabel(r'seconds')
# plt.ylabel(r'samples')
# plt.text(30, 20000, "equation:\n{:.4f} exp({:.4f} x) + {:.4f}".format(*popt))
# plt.text(30, 0.1, "R^2:\n {}".format(r2))
# 
# plt.show()
# 
# ###
# 
# # For all samples 
# popt, pcov = curve_fit(exponenial_func, x, y, p0=(1, 1e-6, 1))
# 
# xx = np.linspace(20, 2000, 1000)
# yy = exponenial_func(xx, *popt)
# 
# r2 = 1. - sum((exponenial_func(x, *popt) - y) ** 2) / sum((y - np.mean(y)) ** 2)
# 
# plt.plot(x, y, 'o', xx, yy)
# plt.title('time to process 2M samples-metadata')
# plt.xlabel(r'seconds')
# plt.ylabel(r'samples')
# plt.xlim([0, 1830])
# plt.ylim([0, 2000000])
# 
# plt.show()
# 
# ###
# =============================================================================











