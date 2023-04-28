#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:15:15 2023

@author: dgaio
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np


home = os.path.expanduser( '~' )

file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/articles.csv"



# Step 1: Collect and prepare the data
data = pd.read_csv(file_path)
X = data['text']
y = data['class']

# Step 2: Define the classes
classes = ['marine', 'human', 'host-associated', 'plant', 'soil']

# Step 3: Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Step 4: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)


# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
target_names = sorted(data['class'].unique())

# Calculate the confusion matrix
conf_mat = np.zeros((len(target_names), len(target_names)), dtype=np.int32)
for i, j in zip(y_test, y_pred):
    conf_mat[target_names.index(i), target_names.index(j)] += 1

# Print the confusion matrix
print("Confusion Matrix:")
print(" ", end="")
for target in target_names:
    print(f"{target:10s}", end="")
print()
for i, target in enumerate(target_names):
    print(f"{target:10s}", end="")
    for j in range(len(target_names)):
        print(f"{conf_mat[i, j]:10d}", end="")
    print()


# Step 6: Use the model
new_article = vectorizer.transform(['This is a new article about marine life.'])
predicted_class = clf.predict(new_article)
print(predicted_class)


###########################
# Edited to print "unknown":
    
# This script loads the data, preprocesses it, 
# splits it into training and testing sets, trains a Naive Bayes classifier, 
# evaluates the model using a confusion matrix, and outputs 
# the predicted class for each sample in the test set. 
# If the classifier is uncertain about the predicted class for a sample, 
# it outputs "unknown".


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the data
data = pd.read_csv(file_path)

# Step 2: Preprocess the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['class']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
target_names = sorted(data['class'].unique())

# Calculate the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred, labels=target_names)

# Print the confusion matrix
print("Confusion Matrix:")
print(" ", end="")
for target in target_names:
    print(f"{target:10s}", end="")
print()
for i, target in enumerate(target_names):
    print(f"{target:10s}", end="")
    for j in range(len(target_names)):
        print(f"{conf_mat[i, j]:10d}", end="")
    print()

# Output uncertain predictions as "unknown"
for i in range(len(y_pred)):
    if np.max(clf.predict_proba(X_test)[i]) < 0.5:
        print(f"Uncertain prediction for sample {i + 1}: unknown")
    else:
        print(f"Predicted class for sample {i + 1}: {y_pred[i]}")

###########################
    
# as a function:   
    
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def predict_class(text, vectorizer, clf):

    # Step 2: Preprocess the data (using the pre-trained vectorizer)
    X_input = vectorizer.transform([text])

    # Step 3: Predict the class of the input text (using the pre-trained classifier)
    y_pred = clf.predict(X_input)[0]
    
    # Output uncertain predictions as "unknown"
    if np.max(clf.predict_proba(X_input)) < 0.2:   # higher means more doubtful 
        return "unknown"
    else:
        return y_pred
    
    
# Preprocess the data and train the classifier
file_path = home+"/cloudstor/Gaio/MicrobeAtlasProject/articles.csv"
data = pd.read_csv(file_path)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['class']
clf = MultinomialNB()
clf.fit(X, y)
    
    
text = "A new study examines the effects of climate change on marine ecosystems. The study analyzes data..."

# Predict the class of the input text
predicted_class = predict_class(text, vectorizer, clf)

# Print the predicted class
print("Predicted class:", predicted_class)


###########################



    


