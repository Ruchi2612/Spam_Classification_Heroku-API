# -*- coding: utf-8 -*-
"""
Created on Sun May 23 04:48:35 2021

@author: Ruchilekha
"""

# 1. Import Libraries
import pandas as pd
import  pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_csv("spam.csv")
#df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# 2. Features & Labels
df['label'] = df['type'].map({'ham':0, 'spam':1})
x = df['text']
Y = df['label']

# 3. Extract features with CountVectorizer 
cv = CountVectorizer()
X = cv.fit_transform(x)

# 4. Saving model to disk
pickle.dump(cv, open('transform.pkl','wb'))

# 5. Splitting train-test dataset 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# 6. Fitting model with training data
clf = MultinomialNB()
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# 7. Saving model to disk
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename,'wb')) 