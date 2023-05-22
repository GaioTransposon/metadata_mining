#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:15:15 2023

@author: dgaio
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def train_classifier_model(model_name):
    """
    Trains a classifier model using the specified training dataset file and model name.
    
    Parameters:
        model_name (str): Name of the classifier model to train. Should be one of:
            'Multinomial Naive Bayes', 'Random Forest Classifier', 'Support Vector Machines', or
            'Gradient Boosting Classifier'.
            
    Returns:
        A tuple containing the trained classifier model and the TfidfVectorizer object.
    """
    file_path = os.path.expanduser('~/github/metadata_mining/middle_dir/pubmed_articles_info_for_training.csv')

    # Step 1: Collect and prepare the data
    data = pd.read_csv(file_path).dropna(subset=['confirmed_biome'])

    # Step 2: Define the classes
    classes = ['water', 'animal', 'soil', 'plant']
    
    # Step 3: Group the data by confirmed_biome and sample an equal number of rows for each biome
    grouped_data = data.groupby('confirmed_biome')
    sampled_data = grouped_data.apply(lambda x: x.sample(grouped_data.size().min(), random_state=42))

    # Step 4: Feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sampled_data['abstract'] + ' ' + sampled_data['title'])
    y = sampled_data['confirmed_biome']

    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

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

    # Step 7: Evaluate the model on the testing set
    y_pred = clf.predict(X_test)
    target_names = sorted(data['confirmed_biome'].unique())

    # Calculate the confusion matrix
    conf_mat = np.zeros((len(target_names), len(target_names)), dtype=np.int32)
    for i, j in zip(y_test, y_pred):
        conf_mat[target_names.index(i), target_names.index(j)] += 1

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
    print("Correct Predictions:", correct_predictions)
    print("Incorrect Predictions:", incorrect_predictions)
    
    # Return the trained model and vectorizer
    return clf, vectorizer







train_classifier_model("Multinomial Naive Bayes")
train_classifier_model("Random Forest Classifier")
train_classifier_model("Support Vector Machines")
train_classifier_model("Gradient Boosting Classifier")



# maybe it's better to train the models and use them all, then pick the biome they most agree on for a testing string. 


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import re
import string

def preprocess_text(text):
    """
    Preprocesses the input text by removing punctuation, converting to lowercase, and removing extra whitespaces.
    
    Parameters:
        text (str): The input text to preprocess.
    
    Returns:
        The preprocessed text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub("\s+", " ", text)  # Remove extra whitespaces
    
    return text

def predict_biome(model, vectorizer, text):
    """
    Predicts the biome for a given text using the trained classifier model and TfidfVectorizer.
    
    Parameters:
        model: Trained classifier model.
        vectorizer: TfidfVectorizer object used for feature extraction.
        text (str): Text to predict the biome for.
    
    Returns:
        The predicted biome for the given text.
    """
    preprocessed_text = preprocess_text(text)
    
    # Transform the preprocessed text using the vectorizer
    X = vectorizer.transform([preprocessed_text])
    
    # Predict the biome
    predicted_biome = model.predict(X)[0]
    
    return predicted_biome

# Example usage
model_name = "Multinomial Naive Bayes"  # Choose the model name you used during training
trained_model, vectorizer = train_classifier_model(model_name)

text_to_predict = "This is a sample text for prediction."  # Replace with your own text
predicted_biome = predict_biome(trained_model, vectorizer, text_to_predict)
print("Predicted Biome:", predicted_biome)





# Example usage
text_to_predict = "This is a sample text for prediction."  # Replace with your own text
preprocessed_text = preprocess_text(text_to_predict)

# Transform the preprocessed text using the vectorizer
X = vectorizer.transform([preprocessed_text])

# Predict the biome using the trained model
predicted_biome = trained_model.predict(X)[0]
print("Predicted Biome:", predicted_biome)



# Check the vocabulary learned by the vectorizer:
vocab = vectorizer.get_feature_names()
print("Vocabulary:", vocab)

if model_name in ["Random Forest Classifier", "Gradient Boosting Classifier"]:
    importances = trained_model.feature_importances_
    feature_importances = list(zip(vocab, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 Feature Importances:")
    for feature, importance in feature_importances[:10]:
        print(feature, importance)
elif model_name == "Multinomial Naive Bayes":
    feature_log_prob = trained_model.feature_log_prob_
    top_biomes = trained_model.classes_
    for i, biome in enumerate(top_biomes):
        print("Top 10 Features for Biome:", biome)
        top_features_indices = feature_log_prob[i].argsort()[::-1][:10]
        for feature_index in top_features_indices:
            print(vocab[feature_index])
        print()
else:
    print("Feature importances/coefficients not available for this model.")

    
train_predictions = trained_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

from sklearn.metrics import accuracy_score

# Assuming you have the training features and labels as X_train and y_train

# Calculate the training predictions
train_predictions = trained_model.predict(X_train)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)


