#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:45:06 2023

@author: dgaio
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set the URL for the search results page on PubMed
url = "https://pubmed.ncbi.nlm.nih.gov/?term=marine+ecology+marine+conservation+water+metagenomics&filter=simsearch1.fha&size=200"


# Send a GET request to the URL and store the response
response = requests.get(url)

# Parse the HTML content of the response using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Create an empty dataframe to store the results
df = pd.DataFrame(columns=["Title", "Abstract", "Journal", "Search Term"])

# Find all article links on the search results page
article_links = soup.find_all("a", class_="docsum-title")


# Loop through the first 20 article links and scrape the title, abstract, and journal name
for link in article_links:
    # Get the URL for the article page
    article_url = "https://pubmed.ncbi.nlm.nih.gov" + link.get("href")
    
    # Send a GET request to the article page and store the response
    article_response = requests.get(article_url)
    
    # Parse the HTML content of the article page using BeautifulSoup
    article_soup = BeautifulSoup(article_response.content, "html.parser")
    
    # Extract the title, abstract, and journal name from the article page
    title = article_soup.find("h1", class_="heading-title")
    abstract = article_soup.find("div", class_="abstract-content selected")
    journal = article_soup.find("span", class_="docsum-journal-citation short-journal-citation").text.strip()
    
    print(journal)
    
    # Check if the journal name includes the word "marine" or "water" (case insensitive)
    if "marine" in journal.lower() or "water" in journal.lower():
        
        # Check if the article has a title, abstract, and journal name
        if title is not None and abstract is not None: #and journal is not None:
            # Extract the text content of the title, abstract, and journal sections
            title = title.text.strip()
            abstract = abstract.text.strip()
            journal = journal
        
            # Create a new row in the dataframe with the article information
            df = df.append({"Title": title, "Abstract": abstract, "Journal": journal, "Search Term": "marine"}, ignore_index=True)  

# Print the dataframe
print(df)
len(df)









