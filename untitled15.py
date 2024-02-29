# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:18:15 2023

@author: hp
"""

import pandas as pd
#Read the data into  pandas dataframe
df = pd.read_csv('C:/2-Dataset/Ecomerce_data.csv')
df=pd.read_csv("C:/8-PCA/.csv")
df
print(df.shape)
df.head(5)
#check the distribution of labels
df['label'].value_counts()
#Add the new column which gives a unique number to each of these label
df['label_num'] = df['label'].map({
    'Household':0,
    'Books':1,
    'Electronics':2,
    'Clothing & Accessories': 3})

#checking the results
df.head(5)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    df.Text,
    df.label_num,
    test_size=0.2 #20% sample will go to test dataset
    random_state=2022,
    stratify=df.label_num
)

print('Shape of X_train: ',x_train.shape)
print("shape of x_test: ",x_test.shape)
y_train.value_counts()
y_test.value_counts()

###################################
#Apply classifier

from sklearn.neighbours import KNeighborsClassifier
from sklearn.pipeline import pipeline
from sklearn.metrics import classification_report


#create pipeline object

clf=pipeline([('vectorizer_tfidf',TfidfVectorizer()),
              ('KNN',KNeighboursClassifier())])
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)










