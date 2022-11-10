#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:44:28 2022
@author: dgaio
"""


import nltk
from nltk.corpus import stopwords
from string import punctuation

#from nltk.tokenize import word_tokenize # Passing the string text into word tokenize for breaking the sentences
#nltk.download('punkt')
#from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#from nltk.stem import PorterStemmer
#from nltk import word_tokenize
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')



file_path = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_50000.txt"
file1 = open(file_path, 'r')
lines = file1.readlines()
file1.close()


# grab sample name: 
sample_name=lines[0].replace('>', '') 


def remove_digit_strings(foo):
  foo1 = [x for x in foo if not any(x1.isdigit() for x1 in x)]
  return(foo1)

my_punctuation_except_dash = punctuation.replace("-", "")

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


my_list=[]
one_mers_list=[]
two_mers_list=[]
three_mers_list=[]
bean_bag=[]


# lines = ['study_STUDY_ABSTRACT=Currently encompasses whole genome sequencing of cultured pathogens as part of a surveillance project for 
the rapid detection of outbreaks of foodborne illnesses',
#           'experiment_DESIGN_DESCRIPTION=HiSeq deep shotgun sequencing of cultured isolate.',
#           'experiment_LIBRARY_CONSTRUCTION_PROTOCOL=150-bp library created for Salmonella enterica str. CFSAN030665.', 
#           'study_STUDY_DESCRIPTION=http://www.sanger.ac.uk/resources/downloads/bacteria/streptococcus-pneumoniae.html This data is part 
of a pre-publication release. For information on the proper use of pre-publication data shared by the Wellcome Trust Sanger Institute 
(including details of any publication moratoria), please see http://www.sanger.ac.uk/datasharing/']



new_stopwords = ["all", "due", "to", "on", "daily"]


from nltk.tokenize import word_tokenize
from collections import Counter

for line in lines:
    
    
    # get rid of end of line char
    line_1 = line.strip()
    
    # get rid of all left string up to =
    line_2= line_1.split('=', 1)[-1]
    
    # subsitute _ with white space  (FIRST!)
    line_3 = line_2. replace('_', ' ') 
    
    # get rid of punctuation except - and _
    #re.sub(pattern, "", line_2) 
    line_4 = line_3.translate(str.maketrans("", "", my_punctuation_except_dash))
        
    #print(line_4)
    
    
    # 1. make line to list
    #my_list = nouns.split()
    
    # tokenize and extract nouns
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = nltk.word_tokenize(line_4)
    my_list = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    #print(my_list)

    
    # 2. chuck if alpha-numeric
    #if len(my_list)<=2:                    # because usually sample names are lone-standing 
    my_list=remove_digit_strings(my_list)   
    #print(my_list)
    
    # Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    refined_list = [lemmatizer.lemmatize(t,'n') for t in my_list] # nouns: plural to singular
    
    
    # remove url-addresses: 
    refined_list = [x for x in refined_list if not x.startswith('http')]
    
    print(refined_list)
    
    
   
    
# =============================================================================
#     # remove stopwords
#     filtered_words = [word for word in my_list if word.lower() not in stopwords.words('english')]
#     #print(filtered_words)
# 
#     # Lemmatization
#     lemmatizer = nltk.WordNetLemmatizer()
#     refined_list = [lemmatizer.lemmatize(t,'v') for t in filtered_words] # verbs: tense to infinitive
#     refined_list = [lemmatizer.lemmatize(t,'n') for t in refined_list] # nouns: plural to singular
#     refined_list = [lemmatizer.lemmatize(t,'s') for t in refined_list] # satellite adj
#     refined_list = [lemmatizer.lemmatize(t,'a') for t in refined_list] # adjectives
#     refined_list = [lemmatizer.lemmatize(t.lower(),'r') for t in refined_list] # adverbs
#     
#     print(refined_list)
# =============================================================================
    
    

    joined=' '.join(refined_list)
    print(joined)
    
    for item in refined_list: 
        if len(refined_list)>0:
            
            bean_bag.append(item.lower())
    
from collections import Counter
out = Counter(bean_bag)
print(out)

        

with open('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/words_occurrence.txt', 'w') as f:
    for k,v in  out.most_common():
        f.write( "{} {}\n".format(k,v) )


    
    
    
    
    
# =============================================================================
#     for g in refined_list:
#         one_mers=one_mers_list.append(g)
# 
#     for g,k in enumerate(refined_list): 
#             
#         if g+1 < len(refined_list):
#             
#             two_mers=refined_list[g]+' '+ refined_list[g+1]
#             two_mers_list.append(two_mers)
#             
#         if g+2 < len(refined_list):
#             
#             three_mers=refined_list[g]+' '+refined_list[g+1]+' '+refined_list[g+2]
#             three_mers_list.append(three_mers)
#             
#         #else: print(two_mers, three_mers) # not sure why there is something getting out of the loop here, but it doesn't seem so! 
# 
# 
# 
# 
# 
# 
#     # map each list against ontologies
#     
#     
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


# =============================================================================
# count = 0
# # Strips the newline character
# for line in lines:
#     count += 1
#     
#     # to show and display line number: 
#     #print("Line{}: {}".format(count, line.strip()))
#     
#     line1 = line.strip()
#     line2= line1.split('=', 1)[-1]
#     #print(line2)
#     
#     # tokenize each line:
#     tokenized_line = word_tokenize(line2)
#     #print(tokenized_lines)
#     
#     for word in tokenized_line:
#         
#         #print(nltk.pos_tag([word]))
#         
#         
#         
#         
#         nouns = [word for word, pos in nltk.pos_tag(word_tokenize(word)) 
#                  if pos.startswith('N')]
#         
#         print(nouns)
#     
# 
# fd = nltk.FreqDist(nouns)
# print(fd)
# 
# fd.tabulate(3)
# =============================================================================
        










