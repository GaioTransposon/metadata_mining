#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:45:06 2023

@author: dgaio
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd


# It takes on a list of biome related terms and creates urls for the search results page on PubMed
def create_pubmed_search_urls(biome_list):
    
    # keep name of list (hence biome) in memory 
    var_name=biome_list[0]
    
    
    my_terms=[]
    for i in biome_list:
        # when item in the list is a composed term, join it: 
        if len(i.split())==2:
            i=i.split()
            i='+'.join(i)
        else:
            pass
        # compose ncbi url
        term="https://pubmed.ncbi.nlm.nih.gov/?term="+str(i)+"+metagenomics"+"&filter=simsearch1.fha&size=200"
        my_terms.append(term)
    return var_name,str(i),my_terms


water_list = ["water", "wastewater", "water sediment", "river", "lake", 
              "groundwater", "estuary", "sea", "marine", "water reservoir" , 
              "ocean", "brine"]

soil_list = ["field", "agricultural", "paddy", "forest", "farm", "desert", 
             "tundra", "peatland","shrub"]

print(create_pubmed_search_urls(water_list)) 
my_urls=create_pubmed_search_urls(water_list)
len(my_urls[2]) 


    
def give_article_links(these_urls):
    
    biome = these_urls[0]
    search_term = these_urls[1]
    
    for url in these_urls[2]: 
        
        # Send a GET request to the URL and store the response
        response = requests.get(url)
        
        # Parse the HTML content of the response using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Create an empty dataframe to store the results
        df = pd.DataFrame(columns=["url", "biome"])

        # Find all article links on the search results page
        articles_links_fun = soup.find_all("a", class_="docsum-title")

        # Loop through the first 20 article links and scrape the title, abstract, and journal name
        for link in articles_links_fun:
            #print(link)
            # Get the URL for the article page
            article_url = "https://pubmed.ncbi.nlm.nih.gov" + link.get("href")
            
            # Create a new row in the dataframe with the article information
            df = df.append({"url": article_url, "biome": biome, "search_term": search_term}, ignore_index=True)  

    return df


x = give_article_links(my_urls)
print(x)         


           


# Define a function to scrape information from URLs
def scrape_url(df):
    
    df_out = df.copy()
    
    for index, row in df.iterrows():
        # Get the URL from the specified column of the row
        url = row['url']
        #url="https://pubmed.ncbi.nlm.nih.gov/35649310/"
        
        # Send a request to the URL
        response = requests.get(url)
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Scrape the title of the article
        title = soup.find('h1', class_='heading-title').text.strip()
        abstract = soup.find("div", class_="abstract-content selected").text.strip()
        journal = soup.find('div', class_='article-source').text.strip().split('\n')[0]
        
        
        if journal is not None and title is not None and abstract is not None:     # and row['search_term'] in journal.lower() 
                        
            # Add the scraped information to the dataframe as new columns
            df_out.loc[index, 'Title'] = title
            df_out.loc[index, 'Abstract'] = abstract
            df_out.loc[index, 'Journal'] = journal
            
        else:
            pass
            
    return df_out
        
        
    

# Apply the function to each row of the dataframe
y = scrape_url(x)
print(y)
            

    


            
            
            
            







import pandas as pd
import spacy

# load the spaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

# read in the abstracts from a CSV file
df = y

# define a function to extract sample origin information from an abstract
def extract_sample_origin(abstract):
    doc = nlp(abstract)
    sample_origins = []
    for ent in doc.ents:
        if ent.label_ in ["LOC", "ORG"]:
            sample_origins.append(ent.text.lower())
    return sample_origins

# apply the function to the abstract column of the dataframe
df["sample_origins"] = df["Abstract"].apply(extract_sample_origin)

# print the resulting dataframe
print(df.head())




