#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:38:07 2023

@author: dgaio
"""


# =============================================================================
# # https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/NLP/Multi-class-text-classifica_fine-tuning-distilbert.ipynb
# 
# # check out later: 
# # https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490f1d192379
# 
# 
# 
# import transformers
# print(transformers.__version__)
# from transformers import DistilBertTokenizer
# 
# # alternative to: 
# from transformers import TFDistilBertForSequenceClassification
# # made an edit to modeling_tf_utils.py: commented out: from keras.saving.hdf5_format import save_attributes_to_hdf5_group
# 
# 
# from transformers import TextClassificationPipeline
# 
# import tensorflow as tf
# import pandas as pd
# import json
# import gc
# 
# from sklearn.model_selection import train_test_split
# 
# import re
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stopw = stopwords.words('english')
# 
# import seaborn as sns
# import matplotlib.pyplot as plt
# from plotly.offline import iplot
# 
# from tqdm import tqdm
# 
# 
# 
# root_path = '/Users/dgaio/Downloads/bbc-text.csv'
# df = pd.read_csv(root_path)
# df.head()
# 
# df.shape
# df.dtypes
# 
# 
# 
# 
# 
# # =============================================================================
# # ###
# # # Histogram of the count of text
# # df['count'] = df['text'].apply(lambda x: len(x.split()))
# # df.head()
# # 
# # plt.figure(figsize= (8, 8))
# # 
# # sns.displot(df['count'])
# # 
# # plt.xlim(0, 1000)
# # 
# # plt.xlabel('The num of words ', fontsize = 16)
# # plt.title("The Number of Words Distribution", fontsize = 18)
# # plt.show()
# # ###
# # 
# # ###
# # # Bar plot for each of the new category
# # category_count = df['category'].value_counts()
# # 
# # categories = category_count.index
# # 
# # categories
# # 
# # category_count
# # 
# # category_count.index
# # 
# # fig = plt.figure(figsize= (12, 5))
# # 
# # ax = fig.add_subplot(111)
# # 
# # sns.barplot(x = category_count.index, y = category_count )
# # 
# # for a, p in enumerate(ax.patches):
# #     ax.annotate(f'{categories[a]}\n' + format(p.get_height(), '.0f'), xy = (p.get_x() + p.get_width() / 2.0, p.get_height()), xytext = (0,-25), size = 13, color = 'white' , ha = 'center', va = 'center', textcoords = 'offset points', bbox = dict(boxstyle = 'round', facecolor='none',edgecolor='white', alpha = 0.5) )
# #     
# # plt.xlabel('Categories', size = 15)
# # 
# # plt.ylabel('The Number of News', size= 15)
# # 
# # plt.xticks(size = 12)
# # 
# # plt.title("The number of News by Categories" , size = 18)
# # 
# # plt.show()
# # ###
# # =============================================================================
# 
# 
# df['category'].unique()
# 
# df['encoded_text'] = df['category'].astype('category').cat.codes
# 
# df.head(10)
# 
# data_texts = df['text'].to_list()
# 
# data_labels = df['encoded_text'].to_list()
# 
# 
# ###
# # Train Test SPlit
# 
# # 20% for validation (80% for training - actually 79.2% because 80%-0.8%)
# train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size = 0.2, random_state = 0 )
# 
# # 1% of the training set is set aside to create a test set (this is 0.8% of the total data): 
# train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size = 0.01, random_state = 0 )
# ###
# 
# 
# ###
# # Model Definition
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# 
# # Tokenize: 
# # truncation set to true truncates to 512 tokens (that's for DistilBert)
# # adding pads to make texts of equal length
# # about padding: In general, deep learning models, like the transformer models used in NLP, are smart enough to understand that padding tokens do not hold any meaningful information, so they won't heavily rely on these tokens to make predictions.
# train_encodings = tokenizer(train_texts, truncation = True, padding = True  )
# 
# val_encodings = tokenizer(val_texts, truncation = True, padding = True )
# 
# # what to do to handle larger text: 
# # option 4 seems best - need to find pout how: 
# # 1.Chunking: Break your text into chunks of maximum length 512 and run each chunk separately through the model. Then combine the outputs in some way (like averaging the predictions).
# # 2.Sliding Window: Similar to chunking, but instead of disjoint chunks, you have an overlap between chunks. This can mitigate the issue of cutting off context between chunks.
# # 3.Use a different model: Some transformer models like Longformer or BigBird are designed specifically to handle long sequences.
# # 4.Text summarization: Use a summarization model to condense your text to a length that fits into the model's limit.
# 
# 
# 
# 
# 
# ###
# # these just creates objects for Tensorflow to handle 
# train_dataset = tf.data.Dataset.from_tensor_slices((
#     dict(train_encodings),
#     train_labels
# ))
# 
# val_dataset = tf.data.Dataset.from_tensor_slices((
#     dict(val_encodings),
#     val_labels
# ))
# ###
# 
# 
# # Loading the pre-trained model and setting it up to have 5 output neurons:
# model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
# 
# 
# from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
# 
# 
# # setting hyperparameters: 
# training_args = TFTrainingArguments(
#     output_dir='./results6',          # saving model predictions and checkpoints
#     num_train_epochs=7,               # number of times the model sees each sample
#     per_device_train_batch_size=16,   # 16 samples (from training data) bundled together and fed to the model at each step 
#     per_device_eval_batch_size=64,    # 64 samples (from validation data) bundled together and fed to the model at each step 
#     warmup_steps=500,                 # 👶 to increase the learning rate linearly (you don't want to learn a lot at the start) 
#     weight_decay=float(1e-5),         # to avoid overfitting to the training data, weights are decreased at each training step
#     logging_dir='./logs',            
#     eval_steps=100                   # evaluation every 100 steps 
# )
# 
# with training_args.strategy.scope():
#     trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 5 )
# 
# 
# trainer = TFTrainer(
#     model=trainer_model,                 
#     args=training_args,                  
#     train_dataset=train_dataset,         
#     eval_dataset=val_dataset,            
# )
# 
# 
# trainer.train()
# trainer.evaluate()
# 
# 
# # =============================================================================
# # # Saving & Loading the model
# # save_directory = "/Users/dgaio/github/metadata_mining/middle_dir/saved_models" 
# # 
# # model.save_pretrained(save_directory)
# # 
# # tokenizer.save_pretrained(save_directory)
# # 
# # 
# # # Loading Pre-Trained Model
# # tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)
# # 
# # model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)
# # test_text = test_texts[0]
# # 
# # test_text
# # 
# # 
# # predict_input = tokenizer_fine_tuned.encode(
# #     test_text,
# #     truncation = True,
# #     padding = True,
# #     return_tensors = 'tf'    
# # )
# # 
# # output = model_fine_tuned(predict_input)[0]
# # 
# # prediction_value = tf.argmax(output, axis = 1).numpy()[0]
# # 
# # prediction_value
# # =============================================================================
# =============================================================================
################################################################################
################################################################################

# just using a classifer:
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# if you have GPU access, you can set the device parameter to 0 to use the GPU, which will speed up model performance.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# continue later as it s 2 GB: 
# https://medium.com/@eijaz/introduction-to-text-classification-using-transformers-be55d49cdd88

################################################################################
################################################################################


# finetuning a pretrained model:   
# https://huggingface.co/learn/nlp-course/chapter3/2?fw=tf


import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# =============================================================================
# # Same as before
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# batch = dict(tokenizer(sequences, padding=True, truncation=True, return_tensors="tf"))
# 
# # This is new
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# labels = tf.convert_to_tensor([1, 1])
# model.train_on_batch(batch, labels)
# =============================================================================

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]

raw_train_dataset.features

# transform sentrnces to numbers: Tokenization: 
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])


# =============================================================================
# # example of tokenizing two sentences to "merge" them:
# # because we here have 2 snetnces....
# inputs = tokenizer("This is the first sentence.", "This is the second one.")
# inputs
# # If we decode the IDs inside input_ids back to words:
# tokenizer.convert_ids_to_tokens(inputs["input_ids"])
# =============================================================================

# Tokenizing inputs: 
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets


# Padding: 
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")


# =============================================================================
# samples = tokenized_datasets["train"][:8]
# samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
# [len(x) for x in samples["input_ids"]]
# 
# batch = data_collator(samples)
# {k: v.shape for k, v in batch.items()}
# =============================================================================


tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)


from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


from tensorflow.keras.losses import SparseCategoricalCrossentropy


model.compile(
    optimizer="adam",   # standard optimizer for DL these days 
    loss=SparseCategoricalCrossentropy(from_logits=True), # later look into this loss, once you know more
    metrics=["accuracy"],  # to see metrics on the fly. can also ask for f1? 
)


model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs = 2 # epochs 
)

# =============================================================================
# # output: 
# # Epoch 1/2
# # 459/459 [==============================] - 400s 872ms/step - loss: 0.6903 - accuracy: 0.6295 - val_loss: 0.6755 - val_accuracy: 0.6838
# # Epoch 2/2
# # 459/459 [==============================] - 399s 869ms/step - loss: 0.6827 - accuracy: 0.6311 - val_loss: 0.7708 - val_accuracy: 0.3162
# 
# # Training loss: Decreased from 0.6903 to 0.6827. This means that the model is getting better at predicting the labels of the training set.
# # Training accuracy: Increased slightly from 0.6295 to 0.6311. This also indicates that the model is getting slightly better at correctly classifying the training samples.
# # Validation loss: Increased from 0.6755 to 0.7708. This means that the model's predictions for the validation set are getting worse.
# # Validation accuracy: Decreased significantly from 0.6838 to 0.3162. This means that the model is getting worse at correctly classifying the validation samples.
# # The increase in validation loss and decrease in validation accuracy could be a sign of overfitting. Overfitting is a common problem in machine learning where the model learns to perform very well on the training data but fails to generalize to new, unseen data (like your validation set). In other words, the model is learning the specific features and noise in the training set, but it's not learning the underlying patterns that would help it perform well on new data.
# # You might want to consider strategies to prevent overfitting, such as adding regularization, using dropout layers, early stopping, or gathering more data. It might also be helpful to try different architectures, learning rates, or other hyperparameters.
# 
# =============================================================================




# =============================================================================
# # ! alternatives to Adam (advanced): 
# # adam learning rate is way too high: 0.001
# # changing it to 20x less: 0.00005  is better 
# # secondly, it is better to lower the learning rate over the 
# # course of the training: a process called decaying or annealing. 
# # PolynomialDecay linearly lowers the learning rate. 
# 
# from tensorflow.keras.optimizers.schedules import PolynomialDecay
# 
# batch_size = 8
# num_epochs = 3
# # The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# # by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# # not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
# num_train_steps = len(tf_train_dataset) * num_epochs
# lr_scheduler = PolynomialDecay(
#     initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
# )
# from tensorflow.keras.optimizers import Adam
# 
# opt = Adam(learning_rate=lr_scheduler)
# 
# import tensorflow as tf
# 
# model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
# 
# # Then, we fit again:
# model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)
# 
# =============================================================================

###
# predicting on raw texts: 
def predict(text1, text2, model, tokenizer):
    # Tokenize the text
    encoded_text = tokenizer(text1, text2, truncation=True, padding=True, return_tensors="tf")

    # Make a prediction
    logits = model(encoded_text)[0]
    # Get the predicted class
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probabilities, axis=-1)

    return predicted_class.numpy(), probabilities.numpy()


text1 = "This is the first sentence."
text2 = "something else entirely"

predicted_class, probabilities = predict(text1, text2, model, tokenizer)

print("Predicted class:", predicted_class)
print("Probabilities:", probabilities)
###



###
# predicting on whole data: 
preds = model.predict(tf_validation_dataset)["logits"]
# logits are outputs of the last layer of the network, 
# before softmax is applied

# apply softmax: 
probabilities = tf.nn.softmax(preds)

# turn probabilities into class predictions by picking the biggest probab for each output (argmax) 
class_preds = np.argmax(probabilities, axis = 1)

# digging into the model: 
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=class_preds, references=raw_datasets["validation"]["label"])
# accuracy: % of times model predicions are correct
# f1 score: how well the model trades off precision and recall
###










#####
# Saving 

# 1. 
# save the model (architecture and weights)
model.save('path_to_my_model')
# load the model later with: 
new_model = tf.keras.models.load_model('path_to_my_model')


# 2. 
# If you're using a transformer model from the Hugging Face library, save model and associated tokenizer: 
model.save_pretrained('path_to_my_model')
tokenizer.save_pretrained('path_to_my_model')
# later load as: 
model = TFAutoModelForSequenceClassification.from_pretrained('path_to_my_model')
tokenizer = AutoTokenizer.from_pretrained('path_to_my_model')
#####




################################################################################







subsampling_percentage = 0.5  # Adjust this to the percentage you want
num_train_epochs = 3

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import clear_output
clear_output(wait=True)




from IPython.display import clear_output
clear_output(wait=True)

df = pd.read_csv('/Users/dgaio/Downloads/bbc-text.csv')

# Convert categorical labels to integers
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Determine number of categories
num_categories = df['category'].nunique()

# Calculate total subsampling size
total_subsampling = int(len(df) * subsampling_percentage)

# Determine subsample size per category
samples_per_category = total_subsampling // num_categories

# Initialize an empty dataframe to store subsamples
df_subsampled = pd.DataFrame()

# Perform subsampling
for category in df['category'].unique():
    df_category = df[df['category'] == category]
    df_subsampled = pd.concat([df_subsampled, df_category.sample(n=samples_per_category)])

df = df_subsampled

# List of sentences (replace with your data)
sentences = df['text'].tolist()
# Corresponding labels (replace with your data)
labels = df['category'].tolist()

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(labels)))

# Tokenize sentences
inputs = tokenizer(sentences, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Prepare dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Split data into train, validation, and test datasets
train_sentences, temp_sentences, train_labels, temp_labels = train_test_split(sentences, labels, test_size=0.3, random_state=42)
val_sentences, test_sentences, val_labels, test_labels = train_test_split(temp_sentences, temp_labels, test_size=0.5, random_state=42)
train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_sentences, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_sentences, truncation=True, padding=True, max_length=128)

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)
test_dataset = TextDataset(test_encodings, test_labels)



from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d") + '_frac' + str(subsampling_percentage) + '_' + str(num_train_epochs) + 'epochs'

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=num_train_epochs, # increase the number of epochs for better visualization
    per_device_train_batch_size=16,  # batch size for training. The model will update its parameters after seeing 16 examples.
    per_device_eval_batch_size=64,   # batch size for evaluation. The model will evaluate its performance on 64 examples at a time.
    warmup_steps=500,                # During the first 500 steps of training, the learning rate will gradually increase. This can sometimes improve the final performance of the model.
    weight_decay=0.01,               # regularization parameter. It can help prevent the model from overfitting to the training data.
    logging_dir=log_dir,             # logs for TensorBoard.
    logging_steps=10,                # Log and save the model every 10 steps: a step is an iteration over a batch of the training dataset.
    evaluation_strategy="steps",  # Evaluate the model every 10 steps
    eval_steps=10,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics, # add this line to compute metrics during training
)

# Train the model
trainer.train()

subsampling_percentage
len(df)
samples_per_category

trainer.evaluate()






# Apply model to test data
test_predictions = trainer.predict(test_dataset)

# Get predicted categories
predicted_categories = test_predictions.predictions.argmax(-1)

# Calculate confusion matrix
cm = confusion_matrix(test_labels, predicted_categories)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()









# =============================================================================
# # Save the model
# model.save_pretrained('./my_model')
# 
# # Load the model
# model = DistilBertForSequenceClassification.from_pretrained('./my_model')
# 
# =============================================================================




from transformers import pipeline

# Convert the model outputs to original category names.
def get_predictions(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return le.inverse_transform(torch.argmax(probs, dim=1).detach().cpu().numpy())

# Usage:
texts = ["This is about business", "Another sample sentence."]
predictions = get_predictions(texts)
print(predictions)





