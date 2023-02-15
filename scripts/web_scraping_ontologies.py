#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:21:16 2022

@author: dgaio
"""


import os
import requests
from bs4 import BeautifulSoup


##############################################################################

# PART 1: from ontobee website get all downloadable tsv files: 

url = "https://ontobee.org"
response = requests.get(url)
page = str(BeautifulSoup(response.content))

def getURL(page):
    """
    :param page: html of web page (here: Python home page) 
    :return: urls in that page 
    """
    start_link = page.find("a href")
    if start_link == -1:
        return None, 0
    start_quote = page.find('"', start_link)
    end_quote = page.find('"', start_quote + 1)
    url = page[start_quote + 1: end_quote]
    return url, end_quote

list_of_urls=[]
while True:
    url, n = getURL(page)
    page = page[n:]
    if url:
        list_of_urls.append(url)
    else:
        break
list_of_tsv=[]
for i in list_of_urls:
    if 'format=tsv' in  i: 
        list_of_tsv.append(i)
print(list_of_tsv)
len(list_of_tsv)
##############################################################################

# PART 2: download the ontologies (tsv files) in directory - must be in our list of desired ontologies

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('?')[0].split('/')[-1]+".tsv"
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving ontology to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

# open list of ontologies we picked
with open('/Users/dgaio/MicrobeAtlasProject/wanted_ontologies.txt') as file:
    our_list = [line.rstrip() for line in file]

for i in list_of_tsv:
    #if i in our_list:
    download(i, dest_folder="/Users/dgaio/MicrobeAtlasProject/ontologies")
    # else:
    #     print("a new ontology is now on Ontobee: ", i)
    #     print("if you want to download it, type 'yes'")
    #     if input("yes"):
    #         download(i, dest_folder="/Users/dgaio/MicrobeAtlasProject/ontologies")
    #     else: 
    #         pass

##############################################################################





