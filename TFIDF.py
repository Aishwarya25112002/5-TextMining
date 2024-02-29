# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:51:59 2023

@author: hp
"""

#How to use TFIDF

import pandas as pd
from sklearn.feature_extraction.text import TFidTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus=['The mouse had a tiny little mouse','the cat saw the mouse','The cat catch te mouse']
#step 1 initialize count vector

cv=CountVectorizer()
#To count the total number of TF

word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape

#how the next step apply in IDF

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

#this matrix is in row matrix form let us convert the dataframe
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names_out(),columns=["idf_weights"])

#sort ascending

df_idf=df_idf.sort_values(by=['idf_weights'])
df_idf

###############################################

from sklearn.feature_extraction.text import TfidfVectorizer

corpus=["The eating pizza,loki is eatin pizza,Ironman ate pizza already","Apple is announcing the ipone tomarrow","Tesla is announcing new model-3 tommarow","microsoft announce new surface tommarow","Amazon is announce new eco-dot announcing"]

#let's create the vectorize and fit the corpus and transform that according
v=TfidfVectorizer()
v.fit(corpus)
transform_output=v.transform(corpus)
#let print the vocabulary

print(v.vocabulary_)
#let print the idf of each other

all_feature

