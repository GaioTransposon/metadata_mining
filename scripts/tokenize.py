#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:07:31 2022
@author: danielagaio
"""

import matplotlib
import emoji # conda install -c conda-forge emoji 
import sklearn # conda install scikit-learn 
import pandas
import plotly # conda install plotly     
import wordcloud # conda install -c conda-forge wordcloud
import networkx # conda install -c anaconda networkx
import wordnet # pip
#import emoji-data-python # pip
import autocorrect # pip
import nltk 

# pip install jupyterlab
# pip install notebook

##### TOKENIZATION
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
text = "I'm making coffee." 
tokens = word_tokenize(text) 
tokens 

from nltk.tokenize import RegexpTokenizer
tokens = RegexpTokenizer('\w+') # punctuation is not considered
print(tokens.tokenize("I can't COME NOW."))

tokens= RegexpTokenizer('\w+|\S') # punctuation is considered
print(tokens.tokenize("I can’t COME NOW."))

tokens = RegexpTokenizer('[A-Z]\w+')
print(tokens.tokenize("I can't COME NOW."))

from nltk.tokenize import WordPunctTokenizer
text="p.s. I'd love to come!"
print(WordPunctTokenizer().tokenize(text)) # everything is split. it is used when we want to split a text into tokens every time there is either a whitespace or a new line or a tab.

from nltk.tokenize import WhitespaceTokenizer
text = 'Would you like to travel to New York?\nThe city is expensive\tbut it is amazing!'
print(WhitespaceTokenizer().tokenize(text)) # It does not tokenize on punctuation.

from nltk.tokenize import TreebankWordTokenizer
text= "If you think you can't keep up-to-date don't @do it! "
print(TreebankWordTokenizer().tokenize(text)) #hyphenated words into single tokens, contractions are not split. 

# Other languages: 
from nltk.tokenize import sent_tokenize
text = 'Θέλω να πάω μια βόλτα. Έχει ζέστη.'
greek = sent_tokenize(text, language='greek')
greek

from nltk.tokenize import word_tokenize
text = 'Θέλω να πάω μια βόλτα. Έχει ζέστη.'
greek = word_tokenize(text, language='greek')
greek
#####

##### Frequency distribution 
from nltk.probability import FreqDist
text = "The cat was under the table. Today it's not."
tokenized = word_tokenize(text, language='english')
tokenized
fdist=FreqDist(tokenized)
fdist

##### Plot the Freq distribution
import matplotlib.pyplot as plt
#fdist.plot(30,cumulative=False)

##### Remove digits: 2 ways 
ori_text = "@ I'm making 200 coffee <for> everione at #work & you're only // making coffee for yourself 😊.?https://edition.cnn.com/" 
text= ''.join(c for c in ori_text if not c.isdigit()) # or use c.isnumeric()
text
#####

import re # import re module
text = re.sub(r'\d+','', ori_text) 
text
#####

##### Removing hyperlinks 
import re # import re module
re.sub(r"http\S+", "", ori_text) 
#####

##### Expand contractions
import re 
def decontracted(text):
    text = re.sub(r"\'re", " are", text) 
    text = re.sub(r"\'m", " am", text) 
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"\'d", " had", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text
decontracted(ori_text)
#####

##### Remove punctuation and special chars
import string # import the string module
string.punctuation # print a set of punctuation characters
text= "".join(c for c in ori_text if not c in string.punctuation)
text

#####

##### Remove emoji
import emoji # import the module emoji

def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)

get_emoji_regexp().sub("",ori_text)# with the method emoji.get_emoji_regexp() replace any match of emoji with the replacement string ""
#####

##### Navigate through emoji data: 
import emoji_data_python
emoji_data_python.emoji_data
emoji_data_python.find_by_name('flag')
emoji_data_python.find_by_shortname('flag')
emoji_data_python.unified_to_char('1F600')
#####

##### White spaces removal 
ori_text.strip() # remove leading and trailing whitespaces 
" ".join(ori_text.split()) # removes all white spaces (except for the ones dividing the items of the list)
#####

##### Spelling correction
from autocorrect import Speller # from the module autocorrect import the class Speller
spell = Speller(lang='en') # create an instance of the class Speller
spell(ori_text)
#####

##### Remove noise in one go: 
import nltk
import string
import re
import emoji
from autocorrect import Speller
spell = Speller(lang='en')
def clean_text(text):
    text= ''.join(c for c in text if not c.isdigit())
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'m", " am", text)
    text = ''.join(c for c in text if c not in string.punctuation)
    text = get_emoji_regexp().sub("",text)
    text = " ".join(text.split())
    text = spell(text)
    return text
clean_text(ori_text)
#####

# Conversion to upper and lower case 
text = "She reads many books at HOME and I think she is the quickest READER I know"
text.lower()
text.upper()

# Stemming (removing affixes (prefixes and suffixes)   --> Way too strong!
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer() 
stemmer.stem('active') 
stemmer.stem('better')
stemmer.stem('friendly')
stemmer.stem('susceptibility')

# Lemmatization
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
lemma.lemmatize('runs') 
lemma.lemmatize('better')
lemma.lemmatize('better', pos = 'a')   # a for adjective 

# Stop words
#nltk.download('stopwords')
from nltk.corpus import stopwords
all_stopwords = stopwords.words('english') # a list of english stop words 
all_stopwords.remove("she") # remove 'she' from the list of stopwords
tokens = word_tokenize(text)
text_wo_stopwords= [word for word in tokens if word not in stopwords.words()]
text_wo_stopwords

# Normalise text in one go: 
def norm_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]
    return tokens

norm_text(text)

# TF_IDF: term frequency 
corpus = ['The cats eat fish and dogs eat veggies','Leo catches fish every day','Once a day I eat fish']
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


cv=CountVectorizer() 
word_count_vector=cv.fit_transform(corpus) 
tokens = cv.get_feature_names_out() 

print(word_count_vector.shape)
print(tokens)
print(len(tokens))
print(word_count_vector.toarray())


# Word count in a dataframe
doc_names = ['Doc{:d}'.format(index) for index, _ in enumerate(word_count_vector)]
df = pd.DataFrame(data=word_count_vector.toarray(), index=doc_names, columns=tokens) 
df

# Calculate Term Frequency
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=False, norm='l1')
X = tfidf_vectorizer.fit_transform(corpus) 
df = pd.DataFrame(X.toarray(), index=doc_names, columns=tfidf_vectorizer.get_feature_names_out())
df

#####
# Inverse Document Frequency (IDF)

# Higher values for words that occur rarely, lower values for frequently occurring terms 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=tokens,columns=["idf_weights"])
df_idf

# exactly as above, but "shorter" written
tf = TfidfVectorizer(use_idf=True)
tf.fit_transform(corpus)
idf = tf.idf_
df_idf = pd.DataFrame(idf, index=cv.get_feature_names_out(),columns=["idf_weights"])
df_idf
#####

#####
# Text similarity (can be useful to cluster samples based on metadata)
# nb: it's solely based on frequency of the words, not sentiment. 
# in my case to obtain clusters based on keywords I can just make PCA based on the keywords! 
text1 = 'The President communicates nicely with all his colleagues.'
text2 = 'The President has a very close relationship with the staff.'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = [text1, text2]
cv=CountVectorizer()
word_count_vector=cv.fit_transform(corpus)
tokens = cv.get_feature_names_out()
print(word_count_vector.shape)
print(tokens)
print(len(tokens))
print(word_count_vector.toarray())
#####



