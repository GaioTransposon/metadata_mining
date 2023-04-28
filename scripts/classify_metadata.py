#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:03:34 2023

@author: dgaio
"""


###########################

import os
import time

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
    
    #new_article = vectorizer.transform([text])
    #predicted_class = clf.predict(new_article)
    #print(text)
    #print(predicted_class)
    
    print(text)
    print(predict_class(text, vectorizer, clf))




executionTime = (time.time() - start)
print('Execution time in seconds: ' + str(executionTime))
print('Execution time in minutes: ' + str(executionTime/60)) 

print('Execution time per sample (sec): ' + str(executionTime/counter))
print('Execution time for 2M samples (min): ' + str((executionTime/60)/counter*2000000))
print('Execution time for 2M samples (h): ' + str((executionTime/60/60)/counter*2000000))

    
