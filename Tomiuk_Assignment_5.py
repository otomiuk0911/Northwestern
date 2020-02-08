# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 05:25:37 2020

@author: o_tom
"""

# =============================================================================
#  Data preparation, exploration, visualization (10 points)
# =============================================================================

#Data Prep

#Import packages
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
import time


data = pd.read_csv('D:/Northwestern/MSDS 422/Week 5/Assignment 5/train.csv') #import training set

#Data overview
print(data.info())
data.dtypes
print(data.describe())
data.head(n=4)

X = data.loc[:,data.columns!= 'label']
y = data['label']

# =============================================================================
# Step 1: Random Forest Classifier
# =============================================================================
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

rfc = RandomForestClassifier(n_estimators = 1000, random_state = 0)
rfc.fit(X,y)
rfc_pred = rfc.predict(X)

print ("My program took", time.time() - start_time, "to run")

print('Mean Absolute Error:', metrics.mean_absolute_error(y, rfc_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, rfc_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, rfc_pred)))

Xt = pd.read_csv('D:/Northwestern/MSDS 422/Week 5/Assignment 5/test.csv') #import test set

y_testpred1 = rfc.predict(Xt)
model1 = pd.DataFrame()
model1['ImageID'] = Xt.index+1
model1['Label'] = (y_testpred1)
model1.to_csv('D:/Northwestern/MSDS 422/Week 5/Assignment 5/Model1.csv', encoding='utf-8', index=False)

# =============================================================================
# Step 2: Principal Components Analysis (PCA) on the combined training and test set data
# =============================================================================

merge = data.append(Xt, sort=False, ignore_index=True) #merge train and test sets
Xm = merge.loc[:,data.columns!= 'label']

from sklearn.decomposition import PCA

start_time = time.time()

pca = PCA(n_components=155) #I changed the n_components input to get to roughly 0.95 explained variance
pca_components = pca.fit_transform(Xm)
sum(pca.explained_variance_ratio_)
pca.components_

print ("My program took", time.time() - start_time, "to run")

Xtrain2 = pd.DataFrame(pca_components).head(42000) #get teh reduced training set
Xtest2 = pd.DataFrame(pca_components).tail(28000) #get the reduced test set

start_time = time.time()

rfc.fit(Xtrain2,y)
rfc_pred2 = rfc.predict(Xtrain2)

print ("My program took", time.time() - start_time, "to run")

print('Mean Absolute Error:', metrics.mean_absolute_error(y, rfc_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(y, rfc_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, rfc_pred2)))

y_testpred2 = rfc.predict(Xtest2)
model2 = pd.DataFrame()
model2['ImageID'] = pd.DataFrame(y_testpred2).index+1
model2['Label'] = (y_testpred2)
model2.to_csv('D:/Northwestern/MSDS 422/Week 5/Assignment 5/Model2.csv', encoding='utf-8', index=False)




