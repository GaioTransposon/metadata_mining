#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:16:17 2023

@author: dgaio
"""

import numpy as np
import pandas as pd
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
from thefuzz import process 
from thefuzz import fuzz
from collections import defaultdict


home = os.path.expanduser( '~' )

file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/sample.info"
out_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/"
file1 = open(file_path, 'r')
lines = file1.readlines()
file1.close()



###
start = time.time()

bean_bag=[]
counter=0


# add: if directories exist, delete them. 
import re




mydic={}
for line in lines: 
    
    # if no line break, then it's the same sample. 
    if line!='\n':     
        
        
        # if line reports sample name, grab sample name: 
        if line.startswith('>'):
            sample_name=line.replace('>', '').strip()
            #print('##########', sample_name)

            mylist=[]
            mydic[sample_name]=[]

        # any other line is the metadata content of the sample. Parse it: 
        else: 
            
            # get rid of end of line char
            line = line.strip()
            
            # get rid of all left string up to =
            line= line.split('=', 1)[-1]
            
            for l in line.split(): 
                
                if re.search('MIGS|MIMS|MIMARK|MISAG|MIMAG|MIUViG', l):
                    #print(sample_name)
                    #print(l)
                    mylist.append(l)
                else:
                    pass
                
            mydic[sample_name]=mylist
      
        
      
# removing keys with empty lists              
mydic_clean = {k: v for k, v in mydic.items() if v != []}
print(mydic_clean)

          
df = pd.DataFrame.from_dict(mydic_clean, orient='index')    

# join all columns into one 
cols = df.columns
df['all'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


# drop all columns except 'all':
df=df.drop(cols, axis = 1)
print(df)


n=0
misag_n=0
marker_n=0
umigs_n=0
migs_n=0
mims_n=0
mimag_n=0
catch_others=0

for index in df.index:
    
    content = df.loc[index, 'all']
    
    
    ###
    misag = re.findall('MISAG', content)  
    migs = re.findall('MIGS', content)  
    umigs = re.findall('UMIGS', content)  # these are from GRC centre in Maryland
    
    
    mimarks = re.findall('MIMARKS', content)      
    m_16S = re.findall('16S', content)  
    mark_gen = re.findall('marker gene', content)  
    
    mims = re.findall('MIMS', content)  
    
    mimag = re.findall('MIMAG', content)  
    ###
    
    
    # print(migs, mimarks, mims, misag, m_16S, mark_gen)
    # print('####')
    
    
    if len(misag)!=0:
        df.loc[index, 'sequencing_type'] = "whole genome sequencing"
        
        misag_n +=1   # tot misag_n=53


    elif len(mimarks)!=0 or len(m_16S)!=0 or len(mark_gen)!=0:
        df.loc[index, 'sequencing_type'] = "metagenomic marker gene"
        
        marker_n+=1   # tot marker_n=78284
    
    
    elif len(umigs)!=0:
        
        df.loc[index, 'sequencing_type'] = "undetermined"
        
        umigs_n+=1   # tot umigs_n=303
        
        
    elif len(migs)!=0 and len(mims)==0 and len(mimag)==0:
        df.loc[index, 'sequencing_type'] = "whole genome sequencing"
        
        migs_n+=1   # tot migs_n=5177
        
        
    elif len(mims)!=0 and len(mimag)==0:
        df.loc[index, 'sequencing_type'] = "shotgun metagenomic"
        
        mims_n+=1   # tot mims_n=37960
    
    
    elif len(mimag)!=0:
        df.loc[index, 'sequencing_type'] = "metagenomic undetermined"
        
        mimag_n+=1   # tot mimag_n=11
        
        
    else: 
        catch_others+=1
        
    
    n+=1
    
# all scanned? check:
misag_n+marker_n+umigs_n+migs_n+mims_n+mimag_n==n

print(df)




my_annot = ["Air",
            "Built-environment", 
            "Host-associated", 
            "Human-associated", 
            "Human-gut",
            "Human-oral", 
            "Human-skin", 
            "Human-vaginal",
            "Microbial mat/biofilm", 
            "Miscellaneous natural or artificial environment",
            "Plant-associated",
            "Sediment",
            "Soil", 
            "Wastewater/sludge",
            "Water",
            "Hydrocarbon resources-cores",
            "Hydrocarbon resources-fluids/swabs"]


for index in df.index:
    
    content = df.loc[index, 'all']
    
    mystring = content.replace(' ','_')
    mystring_list=re.split(r'[,./:_]', mystring) 
    print(mystring_list)
    
    








    
mydic=defaultdict(list)
for i in mystring_list: 
    for j in my_annot:
        
        
        ratio = fuzz.token_set_ratio(i.lower(),j.lower())   
        # if need to make faster: https://github.com/maxbachmann/RapidFuzz

        if mydic[j] == [] or mydic[j] < ratio:
            mydic[j]=ratio

#print(mydic)

pd.DataFrame.from_dict(mydic, orient="index")



# getting all the key-value pairs in the dictionary
result_keyvalpairs = mydic.items()

# converting an object to a list
list_data = list(result_keyvalpairs)

# converting list to an numpy array using numpy array() function
numpy_array = np.array(list_data)




    
    
for index in df.index:
    
    content = df.loc[index, 'all']
    
    mystring = content.replace(' ','_')
    mystring_list=re.split(r'[,./:_]', mystring) 
    
    for i in mystring_list: 
        for j in my_annot:
            ratio = fuzz.ratio(i.lower(),j.lower())
            
            if ratio>80: 
                print(i,j,ratio)
    





# sample origin: 
# 1. Air 
# 2. Built-environment
# 3. Host-associated <-- expand with "mouse" ?
# 4. Human-associated <-- human 
# 5. Human-gut <-- human gut 
# 6. Human-oral <-- human oral
# 7. Human-skin 
# 8. Human-vaginal 
# 9. Microbial mat/biofilm <-- should just "microbial" be mapped to this? 
# 10. Miscellaneous natural or artificial environment <-- map "miscellanous"?
# 11. Plant-associated 
# 12. Sediment   <-- marine sediment? lake sedimen? does it belong here? 
# 13. Soil <-- tundra soil 
# 14. Wastewater/sludge 
# 15. Water <-- expand "lake"? unless it's lake sediment 
# 16. Hydrocarbon resources-cores 
# 17. Hydrocarbon resources-fluids/swabs 
    
    

# sample_metadata

# submitters-made-up: 
# built --> probably meraning built environm (synthetic)
# specimen 
# microbial
# hot springs 
# Unknown
# (uncultured) bacterium / bacteria 
# uncultured
# Bacterial 
# archaeon / Archaea / Archaeal
# Prokaryotes
# Cyanobacterial mat
# rhizosphere
# phyllosphere
# environmental 
# child 
# gut <-- chicken gut, mouse gut, mouse gut 
# feces / STool
# hydrothermal vent
# uncultured organism

# Noctiluca <-- dinoflagellate
# Cephalotes pusillus <-- insect 
# patient <-- attach to human? 
# endodontic <-- it's human-oral 
# Homo sapiens
# dentine --> oral
# sponge 
# human lung 
# mouse / mice / woodrat
# pig
# bovine 
# sheep 
# Sus scrofa
# chicken
# Bubalus bubalis
# leopard frog
# fungi / fungus / fungal (soil fungal?)
# Microtus kikuchii

# rice paddy
# specimen
# microbial community 
# Orchid endophytic fungi
# uncultured bacterium
# uncultured Bacterial/Archaeal
# beach sand
# epibacteria

# aquatic 
# freshwater 
# lake 
# marine
# Groundwater


# anything else: 
# god knows what 

















            