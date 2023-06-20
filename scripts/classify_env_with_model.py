#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:15:15 2023

@author: dgaio
"""

import os
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')



def cut_abstract_and_balance_rows(df, output_file):
    # Find the length of the shortest sentence in the "abstract" column
    shortest_abstract_length = min(df['abstract'].apply(lambda x: len(x.split())))

    # Cut the strings in the "abstract" column to the length of the shortest sentence
    df['abstract'] = df['abstract'].apply(lambda x: ' '.join(x.split()[:shortest_abstract_length]) if pd.notnull(x) else x)

    # Count the number of rows for each biome
    biome_counts = df['confirmed_biome'].value_counts()

    # Find the minimum count among the biomes
    min_biome_count = min(biome_counts)

    # Create an empty dataframe to store the balanced data
    balanced_df = pd.DataFrame()

    # Randomly sample rows for each biome to obtain the minimum count
    for biome in biome_counts.index:
        biome_rows = df[df['confirmed_biome'] == biome]
        sampled_rows = biome_rows.sample(n=min_biome_count, random_state=42)  # Adjust the random_state as desired
        balanced_df = pd.concat([balanced_df, sampled_rows])

    # Write the balanced dataframe to a CSV file
    balanced_df.to_csv(output_file, index=False)


# Example usage
file_path = os.path.expanduser('~/github/metadata_mining/middle_dir/pubmed_articles_info_for_training.csv')
input_df = pd.read_csv(file_path).dropna(subset=['confirmed_biome'])
output_file = '~/github/metadata_mining/middle_dir/pubmed_articles_info_for_training_subsampled.csv'  # Replace 'output_data.csv' with your desired output file name
cut_abstract_and_balance_rows(input_df, output_file)



def filter_nouns(text):
    """
    Filters out non-noun words from the given text using POS tagging.
    
    Parameters:
        text (str): Input text to filter.
    
    Returns:
        str: Filtered text containing only nouns.
    """
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    filtered_tokens = [token for token, tag in tagged_tokens if tag.startswith('N')]
    
    # Lemmatize the filtered tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)






def prep_and_split():
    """
    Prepares the training data by performing steps 1 to 5.
    
    Returns:
        The preprocessed text features (X) and the corresponding labels (y).
    """
    file_path = os.path.expanduser('~/github/metadata_mining/middle_dir/pubmed_articles_info_for_training_subsampled.csv')

    # Step 1: Collect and prepare the data
    data = pd.read_csv(file_path).dropna(subset=['confirmed_biome'])

    # Step 3: Group the data by confirmed_biome and sample an equal number of rows for each biome
    grouped_data = data.groupby('confirmed_biome')
    sampled_data = grouped_data.apply(lambda x: x.sample(grouped_data.size().min(), random_state=42))

    # Step 4: Feature extraction and selection
    texts = sampled_data['abstract'] + ' ' + sampled_data['title']
    filtered_texts = texts.apply(filter_nouns)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(filtered_texts)
    y = sampled_data['confirmed_biome']

    # Apply feature selection using SelectKBest
    k = 100  # Number of top features to select
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.5, stratify=y, random_state=42)

    
    return X_train, X_test, y_train, y_test, vectorizer, selector


def train_classifier_model(X_train, y_train, model_name):
    """
    Trains a classifier model using the specified training dataset and model name.
    
    Parameters:
        X (array-like): The training features.
        y (array-like): The training labels.
        model_name (str): Name of the classifier model to train. Should be one of:
            'Multinomial Naive Bayes', 'Random Forest Classifier', 'Support Vector Machines', or
            'Gradient Boosting Classifier'.
            
    Returns:
        A tuple containing the trained classifier model and the SelectKBest object.
    """
    
    # Step 6: Train the model
    if model_name == "Multinomial Naive Bayes":
        clf = MultinomialNB()
    elif model_name == "Random Forest Classifier":
        clf = RandomForestClassifier(random_state=42)
    elif model_name == "Support Vector Machines":
        clf = SVC(random_state=42)
    elif model_name == "Gradient Boosting Classifier":
        clf = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    clf.fit(X_train, y_train)

    # Step 7: Evaluate the model on the training set
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Step 8: Evaluate the model on the testing set
    y_test_pred = clf.predict(X_test)
    target_names = ['animal', 'plant', 'soil', 'water']

    # Calculate the confusion matrix
    conf_mat = confusion_matrix(y_test, y_test_pred, labels=target_names)

    # Print the confusion matrix
    print("Confusion Matrix -", model_name)
    print("{:<10s}".format(""), end="")
    print("".join("{:<10s}".format(target) for target in target_names))
    for i, target in enumerate(target_names):
        print("{:<10s}".format(target), end="")
        print("".join("{:<10d}".format(conf_mat[i, j]) for j in range(len(target_names))))

    # Calculate the number of correct and incorrect predictions
    correct_predictions = np.trace(conf_mat)
    incorrect_predictions = y_test.shape[0] - correct_predictions
    
    # Print the number of correct and incorrect predictions
    print("\n", model_name)
    print("Train Accuracy:", train_accuracy)
    print("Correct Predictions:", correct_predictions)
    print("Incorrect Predictions:", incorrect_predictions)
    
    # Return the trained model
    return clf





def get_top_features_per_biome(trained_model, vectorizer, selector, n=10):
    selected_indices = selector.get_support(indices=True)
    feature_names = vectorizer.get_feature_names_out()

    if isinstance(trained_model, MultinomialNB):
        if hasattr(trained_model, 'feature_log_prob_'):
            feature_log_prob = trained_model.feature_log_prob_
            top_biomes = trained_model.classes_

            for i, biome in enumerate(top_biomes):
                print(f"Top {n} features for biome: {biome}")
                top_indices = np.argsort(feature_log_prob[i])[::-1][:n]
                top_features = [feature_names[idx] for idx in selected_indices[top_indices]]
                for feature in top_features:
                    print(feature)
                print()
        else:
            print("Feature information not available for this model.")
    elif isinstance(trained_model, RandomForestClassifier):
        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:n]
            top_features = [feature_names[idx] for idx in selected_indices[top_indices]]
            print(f"Top {n} features:")
            for feature in top_features:
                print(feature)
        else:
            print("Feature information not available for this model.")
    elif isinstance(trained_model, GradientBoostingClassifier):
        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:n]
            top_features = [feature_names[idx] for idx in selected_indices[top_indices]]
            print(f"Top {n} features:")
            for feature in top_features:
                print(feature)
        else:
            print("Feature information not available for this model.")
    else:
        print("Feature information not available for this model.")






X_train, X_test, y_train, y_test, vectorizer, selector = prep_and_split()




mnb_model = train_classifier_model(X_train, y_train, "Multinomial Naive Bayes")
get_top_features_per_biome(mnb_model, vectorizer, selector, 20)

rf_model = train_classifier_model(X_train, y_train, "Random Forest Classifier")
get_top_features_per_biome(rf_model, vectorizer, selector, 20)

gb_model = train_classifier_model(X_train, y_train, "Gradient Boosting Classifier")
get_top_features_per_biome(gb_model, vectorizer, selector, 20)

svm_model = train_classifier_model(X_train, y_train, "Support Vector Machines")
get_top_features_per_biome(svm_model, vectorizer, selector, 20)





# Ensembling the prediction: 

# Create a list of (model_name, model) tuples
models = [
    ("Multinomial Naive Bayes", mnb_model),
    ("Random Forest Classifier", rf_model),
    ("Support Vector Machines", svm_model),
    ("Gradient Boosting Classifier", gb_model)
]

# Create a VotingClassifier ensemble with the trained models
ensemble_model = VotingClassifier(estimators=models, voting="hard")

# Fit the ensemble model on the training data
ensemble_model.fit(X_train, y_train)

# Evaluate the ensemble model on the testing set
y_test_pred = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_test_pred)

# Print the ensemble accuracy
print("Ensemble Accuracy:", ensemble_accuracy)


##########
# How the ensembled model performs: 

# Step 7: Evaluate the model on the training set
y_train_pred = ensemble_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Step 8: Evaluate the model on the testing set
y_test_pred = ensemble_model.predict(X_test)
target_names = ['animal', 'plant', 'soil', 'water']

# Calculate the confusion matrix
conf_mat = confusion_matrix(y_test, y_test_pred, labels=target_names)

# Print the confusion matrix
print("Confusion Matrix -", "ensembled")
print("{:<10s}".format(""), end="")
print("".join("{:<10s}".format(target) for target in target_names))
for i, target in enumerate(target_names):
    print("{:<10s}".format(target), end="")
    print("".join("{:<10d}".format(conf_mat[i, j]) for j in range(len(target_names))))

# Calculate the number of correct and incorrect predictions
correct_predictions = np.trace(conf_mat)
incorrect_predictions = y_test.shape[0] - correct_predictions

# Print the number of correct and incorrect predictions
print("\n", "ensembled")
print("Train Accuracy:", train_accuracy)
print("Correct Predictions:", correct_predictions)
print("Incorrect Predictions:", incorrect_predictions)
##########


# =============================================================================
# def extract_nouns(text):
#     """
#     Extracts only nouns from the given text using POS tagging.
#     
#     Parameters:
#         text (str): The input text to extract nouns from.
#     
#     Returns:
#         The extracted nouns as a space-separated string.
#     """
#     tokens = word_tokenize(text)
#     tagged_tokens = nltk.pos_tag(tokens)
#     nouns = [token for token, tag in tagged_tokens if tag.startswith('N')]
#     return ' '.join(nouns)
# =============================================================================




def preprocess_text(text):
    """
    Preprocesses the input text by extracting text after "=",
    replacing punctuation (except dashes) with white spaces,
    converting to lowercase, replacing underscores with white spaces,
    removing lines with empty text after "=", removing numbers,
    removing words that are a mix of numbers and letters,
    removing standalone dashes,
    and removing extra whitespaces.
    
    Parameters:
        text (str): The input text to preprocess.
    
    Returns:
        The preprocessed text.
    """
    # Remove text before and including "="
    text = re.sub(r".*?=", "", text)

    # Replace punctuation (except dashes) with white spaces
    punctuation_except_dash = f"[{re.escape(string.punctuation.replace('-', ''))}]"
    text = re.sub(punctuation_except_dash, " ", text)

    # Convert to lowercase
    text = text.lower()

    # Replace underscores with white spaces
    text = text.replace("_", " ")

    # Remove lines with empty text after "="
    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    text = "\n".join(lines)

    # Remove numbers
    text = re.sub(r"\b\d+\b", "", text)

    # Remove words that are a mix of numbers and letters
    text = re.sub(r"\b\w*\d\w*\b", "", text)

    # Remove standalone dashes
    text = re.sub(r"(?<!\w)-(?!\w)", "", text)

    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text.strip())

    # Extract only nouns from the text
    text = filter_nouns(text)

    return text




def predict_biome(model, vectorizer, selector, text):
    """
    Predicts the biome for a given text using the trained classifier model, TfidfVectorizer, and feature selector.
    
    Parameters:
        model: Trained classifier model.
        vectorizer: TfidfVectorizer object used for feature extraction.
        selector: SelectKBest object used for feature selection.
        text (str): Text to predict the biome for.
    
    Returns:
        The predicted biome for the given text.
    """
    preprocessed_text = preprocess_text(text)
    
    # Transform the preprocessed text using the vectorizer
    X = vectorizer.transform([preprocessed_text])
    
    # Apply feature selection to the transformed text
    X_selected = selector.transform(X)
    
    # Predict the biome
    predicted_biome = model.predict(X_selected)[0]
    
    return predicted_biome










# Directory containing the split files
input_directory = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/'

# Trained classifier models and vectorizers
models = {
    'Multinomial Naive Bayes': mnb_model,
    'Random Forest Classifier': rf_model,
    'Gradient Boosting Classifier': gb_model,
    'Support Vector Machines': svm_model,
    'Ensembled model':ensemble_model
}

# # Initialize the dictionary to store the predictions
# predictions = {'Sample Number': []}



# Initialize the dictionary to store the predictions
predictions = {'Sample Number': [], 'metadata': []}

# Iterate through directories starting with "dir_"
for dir_name in os.listdir(input_directory):
    if dir_name.startswith('dir_'):
        dir_path = os.path.join(input_directory, dir_name)

        # Iterate through the files in the directory
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)

            if file_name.endswith('.txt'):
                # Read the file and extract the sample number
                with open(file_path, 'r') as text_file:
                    lines = text_file.readlines()
                sample_number = lines[0].strip().replace('>', '')

                # Preprocess the text
                text = ' '.join(lines[1:])  # Combine lines excluding the first line
                preprocessed_text = preprocess_text(text)

                # Perform predictions for each model
                for model_name, model in models.items():
                    # Predict the biome
                    predicted_biome = predict_biome(model, vectorizer, selector, preprocessed_text)
                    # Append the predicted biome to the corresponding column
                    if model_name not in predictions:
                        predictions[model_name] = []
                    predictions[model_name].append(predicted_biome)

                # Add the sample number and text to the list of predictions
                predictions['Sample Number'].append(sample_number)
                predictions['metadata'].append(text)

# =============================================================================
# 
# # Iterate through directories starting with "dir_"
# for dir_name in os.listdir(input_directory):
#     if dir_name.startswith('dir_'):
#         dir_path = os.path.join(input_directory, dir_name)
# 
#         # Iterate through the files in the directory
#         for file_name in os.listdir(dir_path):
#             file_path = os.path.join(dir_path, file_name)
# 
#             if file_name.endswith('.txt'):
#                 # Read the file and extract the sample number
#                 with open(file_path, 'r') as text_file:
#                     lines = text_file.readlines()
#                 sample_number = lines[0].strip().replace('>', '')
# 
#                 # Preprocess the text
#                 text = ' '.join(lines[1:])  # Combine lines excluding the first line
#                 preprocessed_text = preprocess_text(text)
# 
#                 # Perform predictions for each model
#                 for model_name, model in models.items():
#                     # Predict the biome
#                     predicted_biome = predict_biome(model, vectorizer, selector, preprocessed_text)
#                     # Append the predicted biome to the corresponding column
#                     if model_name not in predictions:
#                         predictions[model_name] = []
#                     predictions[model_name].append(predicted_biome)
# 
#                 # Add the sample number to the list of predictions
#                 predictions['Sample Number'].append(sample_number)
# 
# =============================================================================



# Create a DataFrame from the predictions dictionary
predictions_df = pd.DataFrame(predictions)

predictions_df.to_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/predictions.csv', index=False)

# Define the biome categories
biome_categories = ['water', 'soil', 'animal', 'plant']

# Calculate and print the sum of occurrences per biome category and model
for model_name, model_predictions in predictions.items():
    print(f"Model: {model_name}")
    for category in biome_categories:
        category_count = model_predictions.count(category)
        print(f"{category.capitalize()}: {category_count}")
    print()

















