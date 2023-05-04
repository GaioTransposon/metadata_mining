#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:45:06 2023

@author: dgaio
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os





# It takes on a list of biome related terms and creates urls for the search results page on PubMed
def create_scholar_search_urls(biome_list):
    
    # biome_list = water_list
    
    # to scrape google scholar first 10 pages 
    pages=[]
    for i in range(0, 11, 10): # final should be ten pages: 0, 91, 10
        pages.append(i)
 
    # keep name of list (hence biome) in memory 
    var_name=biome_list[0]
    mydic={}
        
    for i in biome_list:
        
        # if biome (e.g.: water) is not the same string, so that water-water won't come out: 
        if var_name != str(i): 
            
            # when item in the list is a composed term, join it: 
            if len(i.split())==2:
                i=i.split()
                i='+'.join(i)
            else:
                pass
            
            # compose ncbi url
            small_list=[]
            for p in pages: 
                # only free full texts
                term="https://scholar.google.com/scholar?start="+str(p)+"&q="+str(i)+"+AND+"+var_name+"+AND+microbiome+AND+metagenomics+&hl=en&as_sdt=0,5"
                #print( var_name,str(i),term )
                small_list.append(term)
                
            mydic[str(i)]=small_list
     
    return var_name,mydic


 


# It takes on a list of biome related terms and creates urls for the search results page on PubMed
def create_pubmed_search_urls(biome_list):

    # to scrape google scholar first 10 pages 
    pages=[]
    for i in range(1, 3, 1): # final should be ten pages: 1, 11, 1
        pages.append(i)
    
    # keep name of list (hence biome) in memory 
    var_name=biome_list[0]
    mydic={}
        
    for i in biome_list:
        
        # if biome (e.g.: water) is not the same string, so that water-water won't come out: 
        if var_name != str(i): 
            
            # when item in the list is a composed term, join it: 
            if len(i.split())==2:
                i=i.split()
                i='+'.join(i)
            else:
                pass
            
            # compose ncbi url
            small_list=[]
            for p in pages: 
                # only free full texts
                term="https://pubmed.ncbi.nlm.nih.gov/?term="+str(i)+"+"+var_name+"+microbiome+metagenomics"+"&filter=simsearch2.ffrft&size=10"+"&page="+str(p)
                #print( var_name,str(i),term )
                small_list.append(term)
                
            mydic[str(i)]=small_list
     
    return var_name,mydic
    
    
           

water_list = ["water", "wastewater", "sediment"] # "river", "lake", 
              #"groundwater", "estuary", "sea", "marine", "reservoir" , 
              #"ocean", "brine"]

soil_list = ["soil", "field", "agricultural"]# "paddy", "forest", "farm", "desert", 
             #"tundra", "peatland","shrub"]

x = create_scholar_search_urls(water_list)
x[0]
x[1]

y = create_pubmed_search_urls(water_list)
y[0]
y[1]
    
    
    

    
def get_article_links(df_with_urls):
    
    # for testing purposes
    #df_with_urls = y
    
    # find out which page it comes from (e.g.: scholar or pubmed)
    which_source=list(df_with_urls[1].items())[0][1][0].split('/')[2].split('.')[0]
    
    
    biome = df_with_urls[0]
    article_links_list =[]
            
    for i in df_with_urls[1]:
        search_term=i
        print(search_term)
        
        for page_url in df_with_urls[1][i]:
            
            # URL for testing:
            #page_url="https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=soil+AND+earth+AND+microbiome+AND+metagenomics+&btnG="
            #page_url="https://pubmed.ncbi.nlm.nih.gov/?term=water+microbiome+metagenomics&filter=simsearch1.fha&size=200"
            print(page_url)
            
            # Send a GET request to the URL and get the HTML content
            response = requests.get(page_url)
            
            if (response.status_code == 200):
                
                # Add a 5-second delay between requests
                time.sleep(5)
                
                
                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')


                # if soup from scholar: 
                if which_source == 'scholar':
                    print('scholar')
                    articles = soup.find_all('div', class_='gs_ri')
                    for article in articles:
                        articles_link_fun = article.find('a')['href']
                        article_links_list.append({'biome': biome, 'search_term': search_term, 'search_url': url, 'article_link': articles_link_fun})
                    print(len(article_links_list))
                    
                # if soup from pubmed: 
                elif which_source == 'pubmed':
                    print('pubmed')
                    articles = soup.find_all('div', class_='docsum-content')
                    for article in articles:
                        articles_link_fun = article.find('a', class_='docsum-title')['href']
                        articles_link_fun = 'https://pubmed.ncbi.nlm.nih.gov' + articles_link_fun
                        article_links_list.append({'biome': biome, 'search_term': search_term, 'search_url': page_url, 'article_link': articles_link_fun})
                    print(len(article_links_list))
                
            else: 
                print('server is complaining. need to wait to send more requests')

                # Add a 10 min delay between requests
                # time.sleep(60*10)


    # Convert the list of dictionaries to a DataFrame
    article_links_df = pd.DataFrame(article_links_list)
    print(article_links_df)
 
    return article_links_df




                

# using scholar: sever complaining whole day
x1 = get_article_links(x)
x1 


# using pubmed: succesfull 
y1 = get_article_links(y)
y1 







def extract_pubmed_article_info(some_df):
    
    # for testing purposes
    #some_df = y1
    
    some_df = some_df.reset_index()  # make sure indexes pair with number of rows

    # add new columns for title and abstract
    some_df['title'] = ""
    some_df['abstract'] = ""
    
    # read dataframe and add new info as new columns: title, abstract, methods
    for index, row in some_df.iterrows():
        
        # send a GET request to the article link
        response = requests.get(row['article_links'])
        
        if (response.status_code == 200):
            
            # Add a 5-second delay between requests
            time.sleep(5)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title_section = soup.find('h1', class_='heading-title')
            abstract_section = soup.find('div', class_='abstract-content selected')
            
            if title_section is not None and abstract_section is not None:

                title = title_section.text.strip()
                abstract = abstract_section.text.strip()
                print(title)
                print(abstract)
                
                some_df.loc[index, 'title'] = title
                some_df.loc[index, 'abstract'] = abstract
                
                
            else:
                print('no info')
                
                some_df.loc[index, 'title'] = None
                some_df.loc[index, 'abstract'] = None
                
        else:
            print('server is complaining. need to wait to send more requests')
            
    return some_df
            
            
        
        
    
y2 = extract_pubmed_article_info(y1) 




import pandas as pd

def evaluate_biome_labels(df):
    for idx, row in df.iterrows():
        print('\n\n\n\n')
        print("Title: ", row['title'])
        print("Abstract: ", row['abstract'])
        if 'Methods' in df.columns:
            print("Methods: ", row['methods'])
        while True:
            answer = input(f"Do you agree with the label '{row['biome']}' under 'biome'? (y or n): ")
            print('\n\n')
            if answer.lower() == 'y':
                df.at[idx, 'confirmed_biome'] = row['biome']
                break
            elif answer.lower() == 'n':
                df.at[idx, 'confirmed_biome'] = None
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
    return df



y3 = evaluate_biome_labels(y2[0:4])
#y3.to_csv('articles_info_for_training.csv', index=False)

y4 = evaluate_biome_labels(y2[3:7])





# if file already exists, open it and concatenate the new dataframe
def update_dataframe(df, output_file):
    
    if os.path.isfile(output_file):
        
        # If the output file exists, load the existing dataframe and concatenate with the new rows
        existing_df = pd.read_csv(output_file)
        # Keep only the rows with titles that don't match
        titles = existing_df['title'].str.lower()
        new_rows = df.loc[~df['title'].str.lower().isin(titles)]
        updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
    else:
        # If the output file doesn't exist, use the new dataframe as-is
        updated_df = df.copy()

    # Evaluate the biome labels
    updated_df = evaluate_biome_labels(updated_df)

    # Save the updated dataframe to a new file
    updated_df.to_csv(output_file, index=False)
    print(f"Saved the updated dataframe to {output_file}")
    return updated_df



# Update the dataframe and save to file
update_dataframe(y4, 'articles_info_for_training.csv')












# list of PubMed article links
article_links = ['https://pubmed.ncbi.nlm.nih.gov/28925579/',
                 'https://pubmed.ncbi.nlm.nih.gov/23577216/',
                 'https://pubmed.ncbi.nlm.nih.gov/29016661/',
                 'https://pubmed.ncbi.nlm.nih.gov/27458453/',
                 'https://pubmed.ncbi.nlm.nih.gov/37073260/']

for link in article_links:
    


    





def extract_article_info(df_with_links_to_articles):
    
    df2 = df2.reset_index()  # make sure indexes pair with number of rows



    # read dataframe and add new info as new columns: title, abstract, methods
    for index, row in df2.iterrows():
        
        # title=...
        # abstract=...
        # methods=...
        print(row['article_url'])
        
        if 'nature.com' in row['article_url']:

            #print(row['article_url'])
            out = scrape_nature(row['article_url'])
            # title = out[0]
            # abstract = out[1]
            # methods = out[2]
            print(out[0]) 
            
            
        elif 'journals.plos' in row['article_url']:
            

            #print(row['article_url'])
            out = scrape_plosone(row['article_url'])
            # title = out[0]
            # abstract = out[1]
            # methods = out[2]
            print(out[0]) 
            
    
                                
            
            
            
            
            

        
        
        
        
        
        


        
        
        
        
        
        




        
        
        
######
def scrape_nature(some_url):
    
    # Send a request to the URL
    response = requests.get(some_url)
    
    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the title of the article
    title = soup.find("h1", class_="c-article-title").get_text().strip()

    # Extract the abstract of the article
    abstract = soup.find("div", class_="c-article-section__content").get_text().strip()

    # Extract the methods section of the article
    methods_heading = soup.find("h2", text="Methods")
    if methods_heading:
        methods_div = methods_heading.find_next_sibling("div")
        methods = methods_div.get_text().strip()
    else:
        methods = ""

    # Print the results
    print("Title: ", title)
    print("Abstract: ", abstract)
    print("Methods: ", methods)
    
    return title,abstract,methods
######
     
######
def scrape_plosone(some_url):
    
    # Send a request to the URL
    response = requests.get(some_url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the title
    title = soup.find('h1', id='artTitle').text.strip()

    # Extract the abstract
    abstract = soup.find('div', class_='abstract-content').text.strip()

    # Extract the methods
    methods = soup.find('div', id='section2', class_='section toc-section').text.strip()

    # Print the results
    print('Title:', title)
    print('Abstract:', abstract)
    print('Methods:', methods)
    
    return title,abstract,methods
######

######
def scrape_sciencedirect(some_url):
    
    # Send a request to the URL
    response = requests.get(some_url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the title
    title = soup.find('h1', id='artTitle').text.strip()

    # Extract the abstract
    abstract = soup.find('div', class_='abstract-content').text.strip()

    # Extract the methods
    methods = soup.find('div', id='section2', class_='section toc-section').text.strip()

    # Print the results
    print('Title:', title)
    print('Abstract:', abstract)
    print('Methods:', methods)
    
    return title,abstract,methods
######



    
    
    
import requests
from bs4 import BeautifulSoup

url = "https://www.frontiersin.org/articles/10.3389/fmicb.2015.00966/full"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract title
title = soup.find("h1").get_text().strip()

# Extract abstract
abstract = soup.find("p").get_text().strip()


soup.find("p", class_="mb0").get_text().strip()



# Extract methods
method_section = soup.find("section", {"aria-label": "Methods"}).find("div", {"class": "NlmCategory"}).get_text().strip()

print(title)
print(abstract)
print(method_section)











        
        
        
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




