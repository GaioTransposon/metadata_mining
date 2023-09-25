#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:00:54 2023

@author: dgaio
"""



# # run as: 
# python ~/github/metadata_mining/scripts/get_ncbi_metadata_entrez.py  \
#         --work_dir '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/' \
#             --sample_info_biome_pmid "sample.info_biome_pmid.csv" \
#                 --output_file "sample.info_biome_pmid_title_abstract.csv" \
#                         --figure "sample.info_biome_pmid_title_abstract.pdf"
    

import pandas as pd
import os
import gzip
import argparse  
import ast
import xml.etree.ElementTree as ET
import time
import pickle
import matplotlib.pyplot as plt
import requests
from xml.etree import ElementTree as ET
import time
import pandas as pd


####################

parser = argparse.ArgumentParser(description='Process XML files.')

parser.add_argument('--work_dir', type=str, required=True, help='path to work directory')
parser.add_argument('--sample_info_biome_pmid', type=str, required=True, help='path to input df')
parser.add_argument('--output_file', type=str, required=True, help='name of output file')
parser.add_argument('--figure', type=str, required=True, help='name of figure file')

args = parser.parse_args()

# Prepend work_dir to all the file paths
sample_info_biome_pmid = os.path.join(args.work_dir, args.sample_info_biome_pmid)
output_file = os.path.join(args.work_dir, args.output_file)
figure = os.path.join(args.work_dir, args.figure)


# # for testing purposes
# work_dir = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/"
# sample_info_biome_pmid = os.path.join(work_dir, "sample.info_biome_pmid.csv")
# output_file = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.csv")
# figure = os.path.join(work_dir, "sample.info_biome_pmid_title_abstract.pdf")
# ###################


############ 1. open input df
s = pd.read_csv(sample_info_biome_pmid)

############ 2. get a list of unique PMIDs

# Convert string representation of lists to actual lists
s['pmid'] = s['pmid'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

# Flatten the lists and get unique PMIDs
unique_pmids = pd.Series([pmid for sublist in s['pmid'].dropna() for pmid in sublist]).unique().tolist()
len(unique_pmids)
type(unique_pmids)

############ 3. get info from Entrez 


def extract_all_text(element):
    if element is None:
        return None
    
    texts = []
    for t in element.itertext():
        if t:
            texts.append(t.strip())
    return " ".join(texts).strip()  # Notice the space inside " ".join()



def fetch_articles(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    df = pd.DataFrame(columns=["pmid", "title", "abstract"])
    pmids_no_title = []
    pmids_no_abstract = []
    pmids_no_title_no_abstract = []

    # Splitting the list of pmids into batches of 200
    for i in range(0, len(pmids), 200):
        batch_pmids = pmids[i:i+200]
        params = {
            "db": "pubmed",
            "id": ",".join(batch_pmids),
            "retmode": "xml"
        }
        response = requests.get(base_url, params=params)
        root = ET.fromstring(response.content)

        new_rows = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text
            title_element = article.find(".//ArticleTitle")
            abstract_element = article.find(".//AbstractText")

            #title = title_element.text.strip() if title_element is not None and title_element.text and title_element.text.strip() != "" else None
            #abstract = abstract_element.text.strip() if abstract_element is not None and abstract_element.text and abstract_element.text.strip() != "" else None

            title = extract_all_text(title_element)
            abstract = extract_all_text(abstract_element)


            #title = title_element.text if title_element is not None else None
            #abstract = abstract_element.text if abstract_element is not None else None

            # Checking if title and/or abstract are missing and populating the lists
            if title is None and abstract is not None:
                pmids_no_title.append(pmid)
            elif title is not None and abstract is None:
                pmids_no_abstract.append(pmid)
            elif title is None and abstract is None:
                pmids_no_title_no_abstract.append(pmid)

            new_rows.append({"pmid": pmid, "title": title, "abstract": abstract})

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Pausing for 3 seconds between requests and print progress
        print(f"Processed {i + len(batch_pmids)} out of {len(pmids)} PMIDs...")
        time.sleep(3)

    print(f"\nPMIDs with no title: {len(pmids_no_title)}")
    print(f"PMIDs with no abstract: {len(pmids_no_abstract)}")
    print(f"PMIDs with both missing: {len(pmids_no_title_no_abstract)}")

    return df, pmids_no_title, pmids_no_abstract, pmids_no_title_no_abstract




# Sample PMIDs
pmids_sample = unique_pmids # ["30074988", "31651534", "31661069", "33004443",  "34566912", "29359185", "32595912"]

start_time = time.time()
df_retrieved, pmids_no_title, pmids_no_abstract, pmids_no_title_no_abstract = fetch_articles(pmids_sample)
end_time = time.time()

elapsed_time = end_time - start_time
articles_per_second = len(df_retrieved) / elapsed_time

print(f"\nTime taken: {elapsed_time:.2f} seconds")
print(f"Rate: {articles_per_second:.2f} articles per second")

print(f"PMIDs processed: {len(df_retrieved)}")
print(f"PMIDs without title: {len(pmids_no_title)}")
print(f"PMIDs without abstract: {len(pmids_no_abstract)}")
print(f"PPMIDs without title and abstract: {len(pmids_no_title_no_abstract)}")




############ 4. Merge ncbi metadata to our original dataframe and select 1 (oldest) pmid per sample: 

def select_pmid(group):
    sorted_group = group.sort_values('pmid')
    oldest_was_skipped = False
    skip_reason = None

    for idx, row in sorted_group.iterrows():
        title = row['title'].lower()
        matched_keyword = next((keyword for keyword in exclude_keywords if keyword in title), None)
        
        if not matched_keyword:
            if oldest_was_skipped:
                skipped_and_picked.append((sorted_group.iloc[0]['pmid'], sorted_group.iloc[0]['title'], 
                                           row['pmid'], row['title'], skip_reason))
            return row
        elif idx == sorted_group.index[0]:
            oldest_was_skipped = True
            skip_reason = matched_keyword

    return sorted_group.iloc[0]

# Define the exclusion list
exclude_keywords = ["protocol", "protocols", "method", "methods", "procedure", "procedures", "library", "libraries"]

# Initialize the results list
skipped_and_picked = []

# Process
selected = (
    s.explode('pmid')
    .merge(df_retrieved, on='pmid', how='left')
    .reindex(columns=['sample', 'biome', 'pmid', 'title', 'abstract'])
    .dropna(subset=['title', 'abstract'])
    .groupby('sample').apply(select_pmid).reset_index(drop=True)
)

skipped_and_picked_df = (
    pd.DataFrame(skipped_and_picked, columns=['Skipped_PMID', 'Skipped_Title', 'Picked_PMID', 'Picked_Title', 'Reason_For_Skip_oldest_pmid'])
    .drop_duplicates(subset=['Skipped_PMID', 'Picked_PMID'])
)

print('''For these samples, the second oldest pmid has been picked,\n 
      rather than the oldest, as the oldest contained lab-related terms\n 
      (indicative of protocol description''')
print(skipped_and_picked_df)

print("Unique pmids per biome:")
print(selected.groupby('biome')['pmid'].nunique())


# Save
selected.to_csv(os.path.join(output_file), index=False)
print("Output file succesfully written")



############ 5. Plot: 


# 1. For each biome, count the number of samples with at least one associated PMID
sample_counts = selected.groupby('biome')['sample'].nunique().reset_index().rename(columns={'sample': 'samples_with_t+a'})

# 2. Count the number of unique PMIDs per biome
pmid_counts = selected.groupby('biome')['pmid'].nunique().reset_index().rename(columns={'pmid': 'unique_pmids'})

# 3. Merge the two dataframes
biome_summary = sample_counts.merge(pmid_counts, on='biome')

# 4. Plot


plt.figure(figsize=(15, 10))  # Increase figure size

# First, plot the samples_with_t+a bars
bars1 = plt.bar(biome_summary['biome'], biome_summary['samples_with_t+a'], label='Samples with Title & Abstract', color='b')

# Then, plot the unique_pmids as a stacked bar on top
bars2 = plt.bar(biome_summary['biome'], biome_summary['unique_pmids'], bottom=biome_summary['samples_with_t+a'], label='Unique PMIDs', color='r')

# Add annotations on top of the bars with adjusted position and reduced font size
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/4, yval + 0.5, round(yval, 2), ha="center", va="bottom", fontsize=10, color='b')

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + 3*bar.get_width()/4, yval + bar.get_y() + 0.5, round(yval, 2), ha="center", va="bottom", fontsize=10, color='r')

plt.title('Comparison of samples with title & abstract vs unique PMIDs per biome')
plt.ylabel('Count')
plt.xlabel('Biome')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title="Metrics")

# Save the plot
plt.savefig(figure, dpi=300, bbox_inches='tight')

plt.show()

print("Plotted !")












# Step 1: Expand the 's' dataframe
s_exploded = s.explode('pmid')

# Step 2: Merge 
merged_df = s_exploded.merge(df_retrieved, on='pmid', how='left')

# Step 3: Reorder columns
ordered_df = merged_df[['sample', 'biome', 'pmid', 'title', 'abstract']]

# Step 4: Filter out rows where the title or abstract is NaN
filtered_df = ordered_df.dropna(subset=['title', 'abstract'])







z = filtered_df[1:200]





df = z

mylist = ['ERS492621']

a = df['sample'].isin(mylist)
df = df[a]






import torch
print("Number of GPUs available:", torch.cuda.device_count())
import transformers
from transformers import BertTokenizer, BertModel
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# or: model = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")







import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
# conda install -c conda-forge accelerate


model_name = "meta-llama/Llama-2-7b-chat-hf"    #other possibilities: meta-llama/Llama-2-70b      #or: meta-llama/Llama-2-70b-chat-hf
base_model_path="./huggingface/llama7B"

with open('huggingface_token.txt', 'r') as file:
    my_huggingface_token = file.read().strip()


tokenizer = AutoTokenizer.from_pretrained(model_name, token=my_huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=my_huggingface_token,
                                             #device_map='auto', # for this to run we need the package accelerate which is now installing 
                                             torch_dtype=torch.float16)



#Save the model and the tokenizer to your PC
model.save_pretrained(base_model_path, from_pt=True) 
tokenizer.save_pretrained(base_model_path, from_pt=True)
# done 







from transformers import  LlamaForCausalLM, LlamaTokenizer, pipeline


model_uploaded = LlamaForCausalLM.from_pretrained(base_model_path)
tokenizer_uploaded = LlamaTokenizer.from_pretrained(base_model_path)
# this last gave a problem:
# UnboundLocalError: cannot access local variable 'sentencepiece_model_pb2' where it is not associated with a value











from transformers import pipeline


llama_pipeline = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto")
    



def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    print("Chatbot:", sequences[0]['generated_text'])



if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
get_llama_response(prompt)











print(model.device) # model is in cuda or cpu 


    
    
    
    







