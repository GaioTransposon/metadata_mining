#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:45:06 2023

@author: dgaio
"""

import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import re
from collections import Counter
import random





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
    




def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data_list = [line.strip() for line in lines]
    return data_list

def get_random_selection_from_file(file_path, n, random_seed):
    data_list = read_file(file_path)
    first_item = data_list[0]
    random.seed(random_seed)  # Set the random seed
    random_items = random.sample(data_list[1:], n)
    random_selection = [first_item] + random_items
    return random_selection


home = os.path.expanduser('~')

n = 2
# n = int(input("Enter the number of additional random items to select: "))

# random seed of choice
rs = 42
# rs = int(input("Enter the random seed integer to run random selection of items from lists: "))



soil_list = get_random_selection_from_file((home + "/github/metadata_mining/middle_dir/envs_soil.csv"), n, rs)
print(soil_list)

water_list = get_random_selection_from_file((home + "/github/metadata_mining/middle_dir/envs_water.csv"), n, rs)
print(water_list)

plant_list = get_random_selection_from_file((home + "/github/metadata_mining/middle_dir/envs_plant.csv"), n, rs)
print(plant_list)

animal_list = get_random_selection_from_file((home + "/github/metadata_mining/middle_dir/envs_animal.csv"), n, rs)
print(animal_list)



# Usage exmaple: 
x = create_scholar_search_urls(water_list)
x[0]
x[1]


r = create_pubmed_search_urls(soil_list)
y = create_pubmed_search_urls(water_list)
u = create_pubmed_search_urls(plant_list)
k = create_pubmed_search_urls(animal_list)
    



# Get article links if not already present in old file: 
home = os.path.expanduser('~')
myfile = home + "/github/metadata_mining/middle_dir/pubmed_articles_info_for_training.csv"
old_df = pd.read_csv(myfile)  # Read the older DataFrame

def get_article_links(df_with_urls):
    # for testing purposes
    # df_with_urls = y

    # find out which page it comes from (e.g.: scholar or pubmed)
    which_source = list(df_with_urls[1].items())[0][1][0].split('/')[2].split('.')[0]

    biome = df_with_urls[0]
    article_links_list = []

    for i in df_with_urls[1]:
        search_term = i
        print(search_term)

        for page_url in df_with_urls[1][i]:
            # URL for testing:
            # page_url="https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=soil+AND+earth+AND+microbiome+AND+metagenomics+&btnG="
            # page_url="https://pubmed.ncbi.nlm.nih.gov/?term=water+microbiome+metagenomics&filter=simsearch1.fha&size=200"
            print(page_url)

            # Send a GET request to the URL and get the HTML content
            response = requests.get(page_url)

            if response.status_code == 200:
                # Add a 5-second delay between requests
                time.sleep(5)

                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # if soup from scholar:
                if which_source == 'scholar':
                    print('scholar')
                    articles = soup.find_all('div', class_='gs_ri')
                    for article in articles:
                        article_link = article.find('a')['href']
                        article_links_list.append({'biome': biome, 'search_term': search_term, 'search_url': page_url, 'article_link': article_link})
                    print(len(article_links_list))

                # if soup from pubmed:
                elif which_source == 'pubmed':
                    print('pubmed')
                    articles = soup.find_all('div', class_='docsum-content')
                    for article in articles:
                        article_link = article.find('a', class_='docsum-title')['href']
                        article_link = 'https://pubmed.ncbi.nlm.nih.gov' + article_link
                        article_links_list.append({'biome': biome, 'search_term': search_term, 'search_url': page_url, 'article_link': article_link})
                    print(len(article_links_list))

            else:
                print('server is complaining. need to wait to send more requests')

                # Add a 10 min delay between requests
                # time.sleep(60*10)

    # Convert the list of dictionaries to a DataFrame
    article_links_df = pd.DataFrame(article_links_list)
    
    # Convert the article link columns to lowercase for case-insensitive comparison
    article_links_df['article_link_lower'] = article_links_df['article_link'].str.lower()
    old_links = old_df['article_link'].str.lower()
    
    # Filter out rows with article links already present in the older DataFrame
    article_links_df = article_links_df[~article_links_df['article_link_lower'].isin(old_links)]

    print(article_links_df)
    
    return article_links_df




# using scholar: succesfull
x1 = get_article_links(x)
x1 

# using pubmed: succesfull 
r1 = get_article_links(r)
y1 = get_article_links(y)
u1 = get_article_links(u)
k1 = get_article_links(k)



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
        response = requests.get(row['article_link'])
        
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
            
            
        
        
r2 = extract_pubmed_article_info(r1) 
y2 = extract_pubmed_article_info(y1) 
u2 = extract_pubmed_article_info(u1) 
k2 = extract_pubmed_article_info(k1) 







def depict_biome(biome):
    if biome == 'plant':
        return '🌿'  # Leaf symbol
    elif biome == 'water':
        return '💧'  # Droplet symbol
    elif biome == 'animal':
        return '🐾'  # Animal symbol
    elif biome == 'soil':
        return '🌱'  # Soil symbol
    else:
        return '❓'  # Unknown or unsupported biome

# evaluate biome:
def evaluate_biome_labels(df):
    for idx, row in df.iterrows():
        print('\n\n\n\n')
        print("Title: ", row['title'])
        print("Abstract: ", row['abstract'])
        if 'Methods' in df.columns:
            print("Methods: ", row['methods'])
        while True:
            sys.stdout.flush()  # Flush the output buffer
            biome = row['biome']
            biome_symbol = depict_biome(biome)
            answer = input(f"Do you agree with the label '{biome_symbol}' ({biome}) under 'biome'? (y or n): ")
            print('\n\n')
            if answer.lower() == 'y':
                df.at[idx, 'confirmed_biome'] = biome
                break
            elif answer.lower() == 'n':
                df.at[idx, 'confirmed_biome'] = None
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
    
    return df



r3 = evaluate_biome_labels(r2)

y3 = evaluate_biome_labels(y2)

u3 = evaluate_biome_labels(u2)

k3 = evaluate_biome_labels(k2)

# if an unwanted article has been added, remove it as follows:
#y3.drop(y3[y3['title'] == "Impact of treated wastewater irrigation on antibiotic resistance in the soil microbiome"].index, inplace = True)




def update_dataframe(updated_df, output_file):
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_titles = existing_df['title'].str.lower().tolist()
        new_df = updated_df[~updated_df['title'].str.lower().isin(existing_titles)]
        if len(new_df) > 0:
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            existing_df.to_csv(output_file, index=False)
            print(f"{len(new_df)} new rows added to {output_file}")
            return existing_df
        else:
            print("No new rows added to the existing file.")
            return existing_df
    else:
        updated_df.to_csv(output_file, index=False)
        print(f"File saved as {output_file}")
        return updated_df


# Update the dataframe and save to file
home=os.path.expanduser('~')
myfile=home+"/github/metadata_mining/middle_dir/pubmed_articles_info_for_training.csv"     

r4 = update_dataframe(r3, myfile)
y4 = update_dataframe(y3, myfile)
u4 = update_dataframe(u3, myfile)
k4 = update_dataframe(k3, myfile)






# inspect my training set: 
def analyze_file(filepath):
    # Read the file into a DataFrame
    df = pd.read_csv(filepath)
    
    # Drop the rows where confirmed_biome has NaN values
    df = df.dropna(subset=['confirmed_biome'])

    # Count the number of rows per group
    group_counts = df['confirmed_biome'].value_counts()
    
    # Construct the labels for the pie chart
    labels = [f"{group} ({count})" for group, count in zip(group_counts.index, group_counts.values)]

    # Count the total number of words per group in the "title", "abstract" and "total" columns
    word_count_df = df.groupby('confirmed_biome')[['title', 'abstract']].agg(lambda x: x.str.split().apply(len).sum())
    word_count_df['total'] = word_count_df['title'] + word_count_df['abstract']

    # Create a figure with a 1x2 grid of subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Plot the pie chart on the left subplot
    ax1.pie(group_counts.values, labels=labels, autopct='%1.1f%%')
    ax1.set_title('Number of articles for model training')

    # Plot the bar chart on the right subplot
    word_count_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Distribution of word counts per biome')
    ax2.set_xlabel('biome')
    ax2.set_ylabel('word count')
    ax2.legend(['Title', 'Abstract', 'Total'], loc='best')

    # Print the total word counts per group
    print('Total word counts by Group:')
    print(word_count_df)

    # Show the plot
    plt.show()



analyze_file(myfile)









# =============================================================================
# def extract_article_info(df_with_links_to_articles):
#     
#     df2 = df2.reset_index()  # make sure indexes pair with number of rows
# 
# 
# 
#     # read dataframe and add new info as new columns: title, abstract, methods
#     for index, row in df2.iterrows():
#         
#         # title=...
#         # abstract=...
#         # methods=...
#         print(row['article_url'])
#         
#         if 'nature.com' in row['article_url']:
# 
#             #print(row['article_url'])
#             out = scrape_nature(row['article_url'])
#             # title = out[0]
#             # abstract = out[1]
#             # methods = out[2]
#             print(out[0]) 
#             
#             
#         elif 'journals.plos' in row['article_url']:
#             
# 
#             #print(row['article_url'])
#             out = scrape_plosone(row['article_url'])
#             # title = out[0]
#             # abstract = out[1]
#             # methods = out[2]
#             print(out[0]) 
# =============================================================================
            
    
                                
            
            
            
            
            

        
        
        
        
        
        


        
        
        
        
        
        




        
        
        
# =============================================================================
# ######
# def scrape_nature(some_url):
#     
#     # Send a request to the URL
#     response = requests.get(some_url)
#     
#     # Parse the HTML content of the page with BeautifulSoup
#     soup = BeautifulSoup(response.content, "html.parser")
# 
#     # Extract the title of the article
#     title = soup.find("h1", class_="c-article-title").get_text().strip()
# 
#     # Extract the abstract of the article
#     abstract = soup.find("div", class_="c-article-section__content").get_text().strip()
# 
#     # Extract the methods section of the article
#     methods_heading = soup.find("h2", text="Methods")
#     if methods_heading:
#         methods_div = methods_heading.find_next_sibling("div")
#         methods = methods_div.get_text().strip()
#     else:
#         methods = ""
# 
#     # Print the results
#     print("Title: ", title)
#     print("Abstract: ", abstract)
#     print("Methods: ", methods)
#     
#     return title,abstract,methods
# ######
#      
# ######
# def scrape_plosone(some_url):
#     
#     # Send a request to the URL
#     response = requests.get(some_url)
#     
#     # Parse the HTML content using BeautifulSoup
#     soup = BeautifulSoup(response.content, 'html.parser')
# 
#     # Extract the title
#     title = soup.find('h1', id='artTitle').text.strip()
# 
#     # Extract the abstract
#     abstract = soup.find('div', class_='abstract-content').text.strip()
# 
#     # Extract the methods
#     methods = soup.find('div', id='section2', class_='section toc-section').text.strip()
# 
#     # Print the results
#     print('Title:', title)
#     print('Abstract:', abstract)
#     print('Methods:', methods)
#     
#     return title,abstract,methods
# ######
# 
# ######
# def scrape_sciencedirect(some_url):
#     
#     # Send a request to the URL
#     response = requests.get(some_url)
#     
#     # Parse the HTML content using BeautifulSoup
#     soup = BeautifulSoup(response.content, 'html.parser')
# 
#     # Extract the title
#     title = soup.find('h1', id='artTitle').text.strip()
# 
#     # Extract the abstract
#     abstract = soup.find('div', class_='abstract-content').text.strip()
# 
#     # Extract the methods
#     methods = soup.find('div', id='section2', class_='section toc-section').text.strip()
# 
#     # Print the results
#     print('Title:', title)
#     print('Abstract:', abstract)
#     print('Methods:', methods)
#     
#     return title,abstract,methods
# ######
# =============================================================================



    
 
    

            
            
            
            
            
            
            
            
            
        
        
        
        






            
            
            
            









