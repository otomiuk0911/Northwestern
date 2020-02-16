# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 05:25:37 2020

@author: o_tom
"""

# =============================================================================
# Step 1: Data preparation, exploration, visualization (10 points)
# =============================================================================
pip install pandas
pip install sklearn
pip install matplotlib

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
data.isna().sum().sum()

X = data.loc[:,data.columns!= 'label']
y = data['label']

# =============================================================================
# Step 2: Random Forest Classifier
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
# Step 3: Principal Components Analysis (PCA) on the combined training and test set data
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

# =============================================================================
# Step 4: Rerunning the reduced model with the label data from the training set
# =============================================================================

Xtrain2 = pd.DataFrame(pca_components).head(42000) #get the reduced training set
Xtest2 = pd.DataFrame(pca_components).tail(28000) #get the reduced test set

start_time = time.time()

rfc = RandomForestClassifier(n_estimators = 1000, random_state = 0)
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

# =============================================================================
# Neural Networks Model 1 - 3 Layers, 392 Nodes
# =============================================================================

import tensorflow as tf
from keras import models
from keras import layers
from keras.utils import to_categorical

# declare the training data placeholders
X = data.loc[:,data.columns!= 'label']
# now declare the output data placeholder - 10 digits
y = to_categorical(data['label'])

network1 = models.Sequential()
network1.add(layers.Dense(392, activation='relu', input_shape=(784,))) # rectified linear unit activation function - ReLU
network1.add(layers.Dense(392, activation='relu' ))
network1.add(layers.Dense(392, activation='relu' ))
network1.add(layers.Dense(10, activation='softmax')) # for multiclass classification use softmax

network1.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

start_time = time.time()
network1.fit(X,y, epochs=20, batch_size=100)
print ("My program took", time.time() - start_time, "to run")

nn_testpred1 = network1.predict_classes(Xt)
nn1 = pd.DataFrame()
nn1['ImageID'] = pd.DataFrame(Xt).index+1
nn1['Label'] = (nn_testpred1)
nn1.to_csv('D:/Northwestern/MSDS 422/Week 6/Assignment 6/Model1.csv', encoding='utf-8', index=False)

# =============================================================================
# Neural Networks Model 2 - 3 Layers, 784 Nodes
# =============================================================================

import tensorflow as tf
from keras import models
from keras import layers
from keras.utils import to_categorical

# declare the training data placeholders
X = data.loc[:,data.columns!= 'label']
# now declare the output data placeholder - 10 digits
y = to_categorical(data['label'])

network2 = models.Sequential()
network2.add(layers.Dense(784, activation='relu', input_shape=(784,))) # rectified linear unit activation function - ReLU
network2.add(layers.Dense(784, activation='relu' ))
network2.add(layers.Dense(784, activation='relu' ))
network2.add(layers.Dense(10, activation='softmax')) # for multiclass classification use softmax

network2.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

start_time = time.time()
network2.fit(X,y, epochs=20, batch_size=100)
print ("My program took", time.time() - start_time, "to run")

nn_testpred2 = network2.predict_classes(Xt)
nn2 = pd.DataFrame()
nn2['ImageID'] = pd.DataFrame(Xt).index+1
nn2['Label'] = (nn_testpred2)
nn2.to_csv('D:/Northwestern/MSDS 422/Week 6/Assignment 6/Model2.csv', encoding='utf-8', index=False)


# =============================================================================
# Neural Networks Model 3 - 6 Layers, 392 Nodes
# =============================================================================

import tensorflow as tf
from keras import models
from keras import layers
from keras.utils import to_categorical

# declare the training data placeholders
X = data.loc[:,data.columns!= 'label']
# now declare the output data placeholder - 10 digits
y = to_categorical(data['label'])

network3 = models.Sequential()
network3.add(layers.Dense(392, activation='relu', input_shape=(784,))) # rectified linear unit activation function - ReLU
network3.add(layers.Dense(392, activation='relu' ))
network3.add(layers.Dense(392, activation='relu' ))
network3.add(layers.Dense(392, activation='relu' ))
network3.add(layers.Dense(392, activation='relu' ))
network3.add(layers.Dense(392, activation='relu' ))
network3.add(layers.Dense(10, activation='softmax')) # for multiclass classification use softmax

network3.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

start_time = time.time()
network3.fit(X,y, epochs=20, batch_size=100)
print ("My program took", time.time() - start_time, "to run")

nn_testpred3 = network3.predict_classes(Xt)
nn3 = pd.DataFrame()
nn3['ImageID'] = pd.DataFrame(Xt).index+1
nn3['Label'] = (nn_testpred3)
nn3.to_csv('D:/Northwestern/MSDS 422/Week 6/Assignment 6/Model3.csv', encoding='utf-8', index=False)

# =============================================================================
# Neural Networks Model 4 - 6 Layers, 784 Nodes
# =============================================================================

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# declare the training data placeholders
X = data.loc[:,data.columns!= 'label']
# now declare the output data placeholder - 10 digits
y = to_categorical(data['label'])
\

network4 = models.Sequential()
network4.add(layers.Dense(784, activation='relu', input_shape=(784,))) # rectified linear unit activation function - ReLU
network4.add(layers.Dense(784, activation='relu' ))
network4.add(layers.Dense(784, activation='relu' ))
network4.add(layers.Dense(784, activation='relu' ))
network4.add(layers.Dense(784, activation='relu' ))
network4.add(layers.Dense(784, activation='relu' ))
network4.add(layers.Dense(10, activation='softmax')) # for multiclass classification use softmax

network4.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

start_time = time.time()
network4.fit(X,y, epochs=20, batch_size=100)
print ("My program took", time.time() - start_time, "to run")

nn_testpred4 = network4.predict_classes(Xt)
nn4 = pd.DataFrame()
nn4['ImageID'] = pd.DataFrame(Xt).index+1
nn4['Label'] = (nn_testpred4)
nn4.to_csv('D:/Northwestern/MSDS 422/Week 6/Assignment 6/Model4.csv', encoding='utf-8', index=False)







# =============================================================================
# 
# =============================================================================
import tensorflow as tf

Xmatrix = pd.DataFrame.as_matrix(X, columns =0)

data_tensor = tf.convert_to_tensor(X.values)


784/2




plt.figure()
plt.imshow(X[0])
plt.colorbar()
plt.grid(False)
plt.show()


from sklearn import preprocessing

scaler = preprocessing.StandardScaler(X)

scaler = preprocessing.StandardScaler()

Xa = pd.array(X)
xtf = tf.constant(X)

Xarray = tf.constant(X)

# =============================================================================
# 
# =============================================================================
#############

# Import `tensorflow` 
import tensorflow as tf 

# Initialize placeholders 
X = data.loc[:,data.columns!= 'label']
# now declare the output data placeholder - 10 digits
y = to_categorical(data['label'])

# Flatten the input data
#images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(X, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#############

#test_loss, test_acc = network.evaluate(test_images, test_labels)
#print('test_acc:', test_acc, 'test_loss', test_loss)






# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100


Xx = 

