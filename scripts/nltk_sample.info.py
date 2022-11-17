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

import os
home = os.path.expanduser( '~' )

file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/sample.info_50000.txt"
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



lines = [  '>ERS2697201\n',
 'sample_center_name=Wellcome Sanger Institute\n',
 'sample_alias=243d7ba2-8f1b-11e8-ae47-68b59976a384\n',
 'sample_TITLE=3662STDY7591586\n',
 'sample_TAXON_ID=749906\n',
 'sample_COMMON_NAME=gut metagenome\n',
 'sample_SCIENTIFIC_NAME=gut metagenome\n',
 'sample_SUBJECT_ID=3662STDY7591586\n',
 'sample_ArrayExpress-SPECIES=gut metagenome\n',
 'sample_ENA-FIRST-PUBLIC=2020-02-12\n',
 'sample_ENA-LAST-UPDATE=2018-09-03\n',
 'experiments=ERX3909944\n',
 'study=ERP010503\n',
 'study_STUDY_TITLE=BAMBI__Baby_Associated_MicroBiota_of_the_Intestine_\n',
 'study_STUDY_TYPE=\n',
 'study_STUDY_ABSTRACT=http://www.sanger.ac.uk/research/projects/hostmicrobiota/\n',
 'study_STUDY_DESCRIPTION=Study to characterise the microbiota of preterm infants over time These data are part of a pre-publication release. For information on the proper use of pre-publication data shared by the Wellcome Trust Sanger Institute (including details of any publication moratoria), please see http://www.sanger.ac.uk/datasharing/\n',
 'experiment=ERX3909944\n'
    ]

new_stopwords = ["all", "due", "to", "on", "daily"]


from nltk.tokenize import word_tokenize
from collections import Counter

for line in lines:
    
    
    # get rid of end of line char
    line = line.strip()
    
    # get rid of all left string up to =
    line= line.split('=', 1)[-1]
    
    # subsitute _ with white space  (FIRST!)
    line = line. replace('_', ' ') 
    
    # get rid of punctuation except - and _
    #re.sub(pattern, "", line_2) 
    line = line.translate(str.maketrans("", "", my_punctuation_except_dash))
    
    
    #tokenize: 
    from nltk.tokenize import WhitespaceTokenizer
    line=WhitespaceTokenizer().tokenize(line) # It does not tokenize on punctuation.
        
    
    print(line)
    
    
    
    
    text = 'study_STUDY_DESCRIPTION=Study to characterise the microbiota of preterm infants over time These data are part of a pre-publication release. For information on the proper use of pre-publication data shared by the Wellcome Trust Sanger Institute (including details of any publication moratoria), please see http://www.sanger.ac.uk/datasharing/\n'
    
    #tokenize: 
    from nltk.tokenize import WhitespaceTokenizer
    line=WhitespaceTokenizer().tokenize(text) # It does not tokenize on punctuation.
    line
    
    
    
    
    
    
    ##### TOKENIZATION
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize 
    tokens = word_tokenize(text) 
    tokens 

    from nltk.tokenize import RegexpTokenizer
    tokens = RegexpTokenizer('\w+') # punctuation is not considered
    print(tokens.tokenize(text))

    tokens= RegexpTokenizer('\w+|\S') # punctuation is considered
    print(tokens.tokenize(text))

    from nltk.tokenize import WordPunctTokenizer
    print(WordPunctTokenizer().tokenize(text)) # everything is split. it is used when we want to split a text into tokens every time there is either a whitespace or a new line or a tab.






    from nltk.tokenize import WhitespaceTokenizer
    text = 'Would you like to travel to New York?\nThe city is expensive\tbut it is amazing!'
    print(WhitespaceTokenizer().tokenize(text)) # It does not tokenize on punctuation.

    from nltk.tokenize import TreebankWordTokenizer
    text= "If you think you can't keep up-to-date don't @do it! "
    print(TreebankWordTokenizer().tokenize(text)) #hyphenated words into single tokens, contractions are not split. 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# =============================================================================
#     # make line to list
#     line_split = line.split()
# =============================================================================
    
    # remove url-addresses: 
    line_split = [x for x in line_split if not x.startswith('http')]
    
    # chuck if alpha-numeric
    if len(line_split)<=2:      # because usually sample names are lone-standing 
        line_split=remove_digit_strings(line_split)   
    
    
    
    
    
    # tokenize
    #if len(line_split)>0:    
    #    line_split = nltk.word_tokenize(line_split)
    
    print(line_split)
    
    
    
    
    
    
    
    
    
    # extract nouns
    is_noun = lambda pos: pos[:2] == 'NN'
    line_split = [word for (word, pos) in nltk.pos_tag(line_split) if is_noun(pos)] 
    
    
    
    print(line_split)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print(line_split)
    
    
    
    
    
    
    
    print(my_list)

    
    
    
    # Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    refined_list = [lemmatizer.lemmatize(t,'n') for t in my_list] # nouns: plural to singular
    
    
    
    
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
        











