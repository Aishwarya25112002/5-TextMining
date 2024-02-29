# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:34:57 2023

@author: hp
"""

#               NLP Textpresentation

#NLP ENDOCDING
#ONE HOT ENCODING
#bag of word(BOW)
#TF-IDF



#bag of word
#this BoW converts unstructured data to structured form

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus=['At least seven idian pharma companies are working to develop vaccine against the corana virus.','The deadly virus that has already infected more than 14 million globally']
bag_of_word_model=CountVectorizer()
print(bag_of_word_model.fit_transform(corpus).todense())

bag_of_word_df=pd.DataFrame(bag_of_word_model.fit_transform(corpus).todense())
#this will create dataframe

bag_of_word_df.columns=sorted(bag_of_word_model.vocabulary_)
bag_of_word_df.head()
#################################
#bag of words model small

bag_of_word_model_small=CountVectorizer(max_features=5)

bag_of_word_df_small=pd.DataFrame(bag_of_word_model_small.fit_transform(corpus).todense())
#this will create dataframe

bag_of_word_df_small.columns=sorted(bag_of_word_model_small.vocabulary_)
bag_of_word_df.head()

####################################################

#
import pandas as pd
import numpy as np
#read cvs
df=pd.read_csv("C:/9-Text Mining/spam.csv")

#check first 10 record
df.head()
#total no. of spam and ham
df.Category.value_counts()
#create one more column comprises o and 1
#name of column is spam
df['spam']=df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

#####
#
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(df.Message,df.spam,test_size=0.2)

#let us check the shape of xtrain and x_test data
x_train.shape
x_test.shape

#let us check the type of x_train and y_train

type(x_train)
type(y_train)

######################################333

#create bag of words representation using countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
x_train_cv=v.fit_transform(x_train.values)
x_train_cv.shape

#####################
#train the naive bayes model

from sklearn.naive_bayes import MultinomialNB
#intialize the model
model=MultinomialNB()
#train the model
model.fit(x_train,y_train)

#####################33
#creating bag of words representation using countvectorizer
#of x_test
x_test_cv=v.transform(x_test)
######################
from sklearn.metrics import classification_report
y_pred=model.predict(x_test_cv)
print(classification_report(y_test, y_pred))

#######################################################


