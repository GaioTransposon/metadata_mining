# -*- coding: utf-8 -*-


class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"
    
   
# training set
train_x=["i love the book book", "this is a great book", "the fit is great", "i love the shoes"]
# to what categories each correspond
train_y=[Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

###########################################################################

# create vector
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))            # binary=True why? if a word appears twice, it will still be 0-1
                                                     # If you want to predict categories using n-grams instead of 1-grams
                                                     # just add ngram_range=(1,2)
train_x_vectors = vectorizer.fit_transform(train_x)      


print(vectorizer.get_feature_names_out())
#['book', 'fit', 'great', 'is', 'love', 'shoes', 'the', 'this']

# ['book' 'book book' 'fit' 'fit is' 'great' 'great book' 'is' 'is great'
#  'love' 'love the' 'shoes' 'the' 'the book' 'the fit' 'the shoes' 'this'
#  'this is']               # this is when using 1,2 n-grams

print(train_x_vectors.toarray()) 
# [[1 0 0 0 1 0 1 0]                 # rows are the sentences of the train set
#  [1 0 1 1 0 0 0 1]                 # columns are the feature names 
#  [0 1 1 1 0 0 1 0]                  
#  [0 0 0 0 1 1 1 0]]

# [[1 1 0 0 0 0 0 0 1 1 0 1 1 0 0 0 0]   # when using 1,2 n-grams 
#  [1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1]
#  [0 0 1 1 1 0 1 1 0 0 0 1 0 1 0 0 0]
#  [0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0]]



###########################################################################

# training 
from sklearn import svm 

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

###########################################################################
# predicting 
test_x = vectorizer.transform(['I like the book'])
print(clf_svm.predict(test_x))
#['BOOKS']

test_x = vectorizer.transform(['I love the story'])
print(clf_svm.predict(test_x))
#['CLOTHING']    # "story" never seen before so wrongly predicted

test_x = vectorizer.transform(['I love the books'])
print(clf_svm.predict(test_x))
#['CLOTHING']    # "book" appears in our training set but not "books", so wrongly predicted






