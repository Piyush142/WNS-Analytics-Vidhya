# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 03:38:29 2018

@author: user
"""

import pandas as pd
#import quandl
import sklearn 
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
import numpy as np
df = pd.read_csv('train.csv')
# df = fillna()
#print(df.head())
df[['education']] = df[['education']].astype(str)
df['previous_year_rating'] = df['previous_year_rating'].fillna(3)
C = np.array(df)
X =C[0:54808,0:13]
#print(X[0:10])
Y = C[0:54808,13:14]
#print(Y)
#print(len(Y))
for i in range(1,6):
    le = preprocessing.LabelEncoder()
    le.fit(X[:,i])
    Z = le.transform(X[:,i])
    print(Z)
    le = preprocessing.LabelEncoder()
    le.fit(X[:,i])
    X[:,i] = le.transform(X[:,i])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size = 0.2)
clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
