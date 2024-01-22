#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:55:10 2024

@author: dgaio
"""


import openai
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import csv
import plotly.graph_objects as go
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# compute cosine similarity
def custom_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# get embeddings from OpenAI
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
    return np.array([embedding['embedding'] for embedding in response['data']])

# rescale opacities
def rescale_opacity(opacities, new_min=0.3, new_max=1.0):
    old_min, old_max = min(opacities), max(opacities)
    return [new_min + (new_max - new_min) * ((opacity - old_min) / (old_max - old_min)) for opacity in opacities]

# open api key
api_key_path ='/Users/dgaio/my_api_key'
with open(api_key_path, "r") as file:
    openai.api_key = file.read().strip()


################################


input_gold_dict = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gold_dict.pkl"

with open(input_gold_dict, 'rb') as file:
    gold_dict = pickle.load(file)
gold_dict = gold_dict[0]

# Convert gold_dict to DataFrame
gold_dict_df = pd.DataFrame(gold_dict.items(), columns=['Sample ID', 'tuple_data'])
gold_dict_df['pmid'] = gold_dict_df['tuple_data'].apply(lambda x: x[0])
gold_dict_df['biome'] = gold_dict_df['tuple_data'].apply(lambda x: x[1])
gold_dict_df.drop(columns='tuple_data', inplace=True)


################################


# path
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240117_1755.txt'

# read file and extract summaries and smaple IDs
texts = []
sample_ids = []
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        sample_ids.append(row[0])  # sample IDs
        texts.append(row[1])       # sample's summary 



# get embeddings
text_embeddings = get_embeddings(texts)

# Compute similarities among all embeddings
all_embeddings = np.vstack((text_embeddings))

similarity_matrix = cosine_similarity(all_embeddings)


# Define biome colors
biome_colors = {"plant": "green", "water": "blue", "animal": "pink", "soil": "brown"}

################################


# K-means 

# Define your variables for coloring
color_by_cluster = 0  # Set to 1 for coloring by cluster, 0 otherwise
color_by_biome = 1    # Set to 1 for coloring by biome, 0 otherwise


# K-means Clustering
n_clusters = 10  # Change this number based on your expectation of distinct clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(all_embeddings)

# UMAP for dimensionality reduction
reducer = umap.UMAP()   # evt: random_state=42

transformed_embeddings = reducer.fit_transform(all_embeddings)


import matplotlib
import matplotlib.pyplot as plt

# Generate a list of 10 distinct colors
n_colors = 10  # Adjust this if you have more than 10 clusters
colors = plt.cm.get_cmap('tab10', n_colors)

# Convert colors to hex format for Plotly
hex_colors = [matplotlib.colors.rgb2hex(colors(i)) for i in range(n_colors)]



# Initialize plot data and results
plot_data = []
results = []


# Process each text embedding for plotting
for i, embedding in enumerate(transformed_embeddings):
    # Calculate the max similarity for the current embedding
    max_similarity = max(similarity_matrix[i])

    # It's a text sample
    sample_id = sample_ids[i]
    text_sample = texts[i]
    
    # Safely get biome from gold_dict_df
    biome = gold_dict_df.loc[gold_dict_df['Sample ID'] == sample_id, 'biome'].values

    text_sample = texts[i]
    text = f"ID: {sample_id}<br>Summary: {text_sample[:200]}..." if len(text_sample) > 200 else f"ID: {sample_id}<br>Summary: {text_sample}"

    # Populate your results DataFrame
    results.append({
        "Sample ID": sample_id, 
        "Biome": biome[0] if biome.size > 0 else "Unknown", 
        "Cluster": clusters[i],
        "Summary": text_sample  # Add the full summary text
    })
    
    # Determine color based on user selection
    if color_by_cluster:
        color = hex_colors[clusters[i]]
    elif color_by_biome:
        biome_color = biome_colors.get(biome[0], 'grey') if biome.size > 0 else 'grey'
        color = biome_color
    else:
        color = 'grey'  # Default color

    plot_data.append({
        "x": embedding[0],
        "y": embedding[1],
        "text": text,
        "color": color,
        "opacity": max_similarity
    })
    
# Create a Plotly scatter plot
fig = go.Figure()

# Initialize a set to keep track of clusters already plotted
plotted_clusters = set()

# Add points to the scatter plot with legend entries
for i, data in enumerate(plot_data):
    cluster_num = clusters[i]  # Use loop index for cluster number
    if cluster_num not in plotted_clusters:
        # Add trace with a legend entry
        fig.add_trace(go.Scatter(
            x=[data['x']], 
            y=[data['y']], 
            mode='markers',
            marker=dict(color=hex_colors[cluster_num], size=7),
            text=data['text'],
            hoverinfo='text',
            name=f'Cluster {cluster_num}'  # Set the name for the legend
        ))
        plotted_clusters.add(cluster_num)
    else:
        # Add trace without a legend entry
        fig.add_trace(go.Scatter(
            x=[data['x']], 
            y=[data['y']], 
            mode='markers',
            marker=dict(color=hex_colors[cluster_num], size=7),
            text=data['text'],
            hoverinfo='text',
            showlegend=False
        ))

# Update layout with titles and labels
fig.update_layout(
    title='Text Embeddings Visualization with UMAP',
    xaxis_title='UMAP 1',
    yaxis_title='UMAP 2'
)

# Show the plot
fig.show()

# Save the figure as an HTML file
fig.write_html("text_embeddings_visualization.html")


# Create DataFrame for samples only
df = pd.DataFrame(results)

# Display the DataFrame
print(df)

################################



################################

# TD-IDF 

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Assuming 'df' is your DataFrame with 'Cluster' and 'Summary' columns
clustered_texts = df.groupby('Cluster')['Metadata'].apply(lambda texts: ' '.join(texts))     ####### or Metadata

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

# Fit and transform the texts
tfidf_matrix = vectorizer.fit_transform(clustered_texts)

# Get feature names to use for displaying top words
feature_names = np.array(vectorizer.get_feature_names_out())

# Dictionary to keep track of word frequency across clusters
word_cluster_count = {}

# Function to update the word count dictionary
def update_word_cluster_count(cluster_num, top_words):
    for word in top_words:
        if word in word_cluster_count:
            word_cluster_count[word].add(cluster_num)
        else:
            word_cluster_count[word] = {cluster_num}

# Function to get top words for each cluster
def get_top_words(cluster_num, top_n):
    row = np.squeeze(tfidf_matrix[cluster_num].toarray())
    top_word_indices = np.argsort(row)[::-1][:top_n]
    top_words = feature_names[top_word_indices]
    # Update the word count
    update_word_cluster_count(cluster_num, top_words)
    return top_words

# Assuming the number of clusters is known as 'n_clusters'
for cluster in range(n_clusters):
    get_top_words(cluster, 10)  # Adjust 10 if needed

# Now filter out words that appear in more than 2 clusters
filtered_words = {word for word, clusters in word_cluster_count.items() if len(clusters) <= 2}

# Redefine get_top_words to exclude frequent words
def get_filtered_top_words(cluster_num, top_n):
    row = np.squeeze(tfidf_matrix[cluster_num].toarray())
    all_word_indices = np.argsort(row)[::-1]
    top_words = []
    for idx in all_word_indices:
        if feature_names[idx] in filtered_words and len(top_words) < top_n:
            top_words.append(feature_names[idx])
    return top_words

# Display filtered top words for each cluster
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    print(get_filtered_top_words(cluster, 3))  # Adjust 10 if needed



################################
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Assuming 'df' is your DataFrame with 'Cluster' and 'Summary' columns
clustered_texts = df.groupby('Cluster')['Metadata'].apply(lambda texts: ' '.join(texts))

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=None, stop_words='english')

# Fit and transform the texts
tfidf_matrix = vectorizer.fit_transform(clustered_texts)

# Get feature names to use for displaying top words
feature_names = np.array(vectorizer.get_feature_names_out())

# Dictionary to keep track of word frequency across clusters
word_cluster_count = {}

# Populate the word count dictionary
for cluster in range(tfidf_matrix.shape[0]):
    row = np.squeeze(tfidf_matrix[cluster].toarray())
    for word_index in row.nonzero()[0]:
        word = feature_names[word_index]
        if word in word_cluster_count:
            word_cluster_count[word].add(cluster)
        else:
            word_cluster_count[word] = {cluster}

# Filter out words that appear in 5 or more clusters
unique_words = {word for word, clusters in word_cluster_count.items() if len(clusters) < 5}

# Function to get top unique words for each cluster
def get_top_unique_words(cluster_num, top_n):
    row = np.squeeze(tfidf_matrix[cluster_num].toarray())
    all_word_indices = np.argsort(row)[::-1]
    top_words = []
    for idx in all_word_indices:
        if feature_names[idx] in unique_words:
            top_words.append(feature_names[idx])
            if len(top_words) == top_n:
                break
    return top_words

# Assuming the number of clusters is known as 'n_clusters'
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    print(get_top_unique_words(cluster, 10))  # Change 10 to get more or fewer top words


################################

# SHAP


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare your data
X = text_embeddings  # Text embeddings as features
y = clusters  # Cluster labels as target

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)


import shap # Version: 0.44.0
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize JS visualization in the notebook (if you're using Jupyter)
shap.initjs()

# Create a SHAP explainer and calculate SHAP values
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test)


# Assuming 'texts' is your list of text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Now you can use vectorizer.get_feature_names_out() since the vectorizer has been fitted
feature_names = vectorizer.get_feature_names_out()
print(feature_names[1600:1650])  # Print first 10 feature names to check


# Now this should work without error
shap.summary_plot(shap_values[3], X_test, feature_names=feature_names)


################################


################################


# wordclouds 

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame with 'Cluster' and 'Summary' columns
# Create a dictionary to store text for each cluster
cluster_texts = {}
for cluster in df['Cluster'].unique():
    texts = df[df['Cluster'] == cluster]['Summary'].str.cat(sep=' ')
    cluster_texts[cluster] = texts

# Generate word clouds for each cluster
for cluster, texts in cluster_texts.items():
    print(cluster)
    wordcloud = WordCloud(width=800, height=800, 
                          background_color='white', 
                          min_font_size=10).generate(texts)
    
    # Plot the WordCloud image                        
    plt.figure(figsize=(8, 8), facecolor=None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad=0) 
  
    plt.title(f'Word Cloud for Cluster {cluster}')
    plt.show()

    # # Save the word cloud to a file
    # wordcloud.to_file(f'wordcloud_cluster_{cluster}.png')


################################





import os
import csv
import openai
import numpy as np

# Your existing functions
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
    return np.array([embedding['embedding'] for embedding in response['data']])

def fetch_metadata_from_sample(sample):
    folder_name = f"dir_{sample[-3:]}"
    folder_path = os.path.join(METADATA_DIRECTORY, folder_name)
    metadata_file_path = os.path.join(folder_path, f"{sample}_clean.txt")
    with open(metadata_file_path, 'r') as file:
        return file.read()

# Constants
file_path = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/gpt_clean_output_nspb200_chunksize1500_modelgpt-3.5-turbo-1106_temp1.0_maxtokens4096_topp0.75_freqp0.25_presp1.5_dt20240117_1755.txt'
METADATA_DIRECTORY = "/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_split_dirs"

# Read file and extract sample IDs
sample_ids = []
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        sample_ids.append(row[0])  # sample IDs

# Fetch metadata for each sample and create embeddings
metadata_texts = []
metadata_embeddings = []
for sample_id in sample_ids:
    metadata = fetch_metadata_from_sample(sample_id)
    metadata_texts.append(metadata)

metadata_embeddings = get_embeddings(metadata_texts)


# Compute similarities among all metadata embeddings
all_metadata_embeddings = np.vstack((metadata_embeddings))
similarity_matrix = cosine_similarity(all_metadata_embeddings)

# K-means Clustering on metadata embeddings
n_clusters = 10  # Adjust based on your expectation of distinct clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(all_metadata_embeddings)

# UMAP for dimensionality reduction on metadata embeddings
reducer = umap.UMAP()  # evt: random_state=42
transformed_metadata_embeddings = reducer.fit_transform(all_metadata_embeddings)




import matplotlib
import matplotlib.pyplot as plt

# Generate a list of 10 distinct colors
n_colors = 10  # Adjust this if you have more than 10 clusters
colors = plt.cm.get_cmap('tab10', n_colors)

# Convert colors to hex format for Plotly
hex_colors = [matplotlib.colors.rgb2hex(colors(i)) for i in range(n_colors)]



# Initialize plot data and results
plot_data = []
results = []


# Assuming transformed_metadata_embeddings is the result of UMAP on metadata_embeddings
for i, embedding in enumerate(transformed_metadata_embeddings):
    # Calculate the max similarity for the current metadata embedding
    max_similarity = max(similarity_matrix[i])

    # It's a metadata sample
    sample_id = sample_ids[i]
    metadata_sample = metadata_texts[i]
    
    # Safely get biome from gold_dict_df
    biome = gold_dict_df.loc[gold_dict_df['Sample ID'] == sample_id, 'biome'].values

    metadata_sample = metadata_texts[i]
    text = f"ID: {sample_id}<br>Metadata: {metadata_sample[:200]}..." if len(metadata_sample) > 200 else f"ID: {sample_id}<br>Metadata: {metadata_sample}"

    # Populate your results DataFrame
    results.append({
        "Sample ID": sample_id, 
        "Biome": biome[0] if biome.size > 0 else "Unknown", 
        "Cluster": clusters[i],
        "Metadata": metadata_sample  # Add the full metadata text
    })
    
    # Determine color based on user selection
    if color_by_cluster:
        color = hex_colors[clusters[i]]
    elif color_by_biome:
        biome_color = biome_colors.get(biome[0], 'grey') if biome.size > 0 else 'grey'
        color = biome_color
    else:
        color = 'grey'  # Default color

    plot_data.append({
        "x": embedding[0],
        "y": embedding[1],
        "text": text,
        "color": color,
        "opacity": max_similarity
    })


# Create a Plotly scatter plot
fig = go.Figure()

# Initialize a set to keep track of clusters already plotted
plotted_clusters = set()

# Add points to the scatter plot with legend entries
for i, data in enumerate(plot_data):
    cluster_num = clusters[i]  # Use loop index for cluster number
    if cluster_num not in plotted_clusters:
        # Add trace with a legend entry
        fig.add_trace(go.Scatter(
            x=[data['x']], 
            y=[data['y']], 
            mode='markers',
            marker=dict(color=hex_colors[cluster_num], size=7),
            text=data['text'],
            hoverinfo='text',
            name=f'Cluster {cluster_num}'  # Set the name for the legend
        ))
        plotted_clusters.add(cluster_num)
    else:
        # Add trace without a legend entry
        fig.add_trace(go.Scatter(
            x=[data['x']], 
            y=[data['y']], 
            mode='markers',
            marker=dict(color=hex_colors[cluster_num], size=7),
            text=data['text'],
            hoverinfo='text',
            showlegend=False
        ))

# Update layout with titles and labels
fig.update_layout(
    title='Text Embeddings Visualization with UMAP - from raw (cleaned) metadata',
    xaxis_title='UMAP 1',
    yaxis_title='UMAP 2'
)

# Show the plot
fig.show()

# Save the figure as an HTML file
fig.write_html("text_embeddings_visualization.html")


# Create DataFrame for samples only
df = pd.DataFrame(results)

# Display the DataFrame
print(df)




    
    
    
    
# =============================================================================
# import matplotlib.pyplot as plt
# 
# # Take average SHAP values per feature across all samples
# mean_shap_values = np.abs(shap_values[0]).mean(axis=0)
# sorted_indices = np.argsort(mean_shap_values)
# 
# # Plot
# plt.figure(figsize=(10, 8))
# plt.title("Feature Importance based on SHAP values")
# plt.barh(range(20), mean_shap_values[sorted_indices][-20:], align='center')  # Top 20 features
# plt.yticks(range(20), feature_names[sorted_indices][-20:])
# plt.xlabel("SHAP Value (mean absolute value)")
# plt.show()
# =============================================================================


