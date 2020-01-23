# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 05:38:15 2020

@author: o_tom
"""

# =============================================================================
#  Data preparation
# =============================================================================

#Data Prep

#Import packages
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sn
import matplotlib.pyplot as plt


data = pd.read_csv('D:/Northwestern/MSDS 422/Week 3/Titanic - Assignment 3/train.csv')

#Data overview
print(data.info())
print(data.dtypes)
print(data.describe())
print(data.head(n=4))
print(data.columns)


data = data.fillna(data.mean())
data.isnull().sum()

# =============================================================================
# EDA and Graphing
# =============================================================================

def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(10,10))
    sn.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)  
  
corr_chart(df_corr = data) 

#Prep for Barcharts
barcharts = pd.DataFrame() 
barcharts = data[['Survived', 'Pclass', 'Sex', 'SibSp', 'Embarked']]

for column in barcharts.columns[1:]:
    sn.set(style="whitegrid")
    fig, ax = plt.subplots()
    ax = sn.barplot(x=column, y="Survived", data=barcharts, capsize=.2)
    
#Hist for Age
sn.set(style="whitegrid")
ax = sn.distplot(data.Age)

#Check binary is ordinal
sn.countplot(x='Survived', data=data)
#need info for fare and survived. Need info for age and survived, 

# =============================================================================
# Create Dummy Variables
# =============================================================================
#Selecting Categorical Variables 
data.loc[data['Sex'] == "male", 'd1_sex(m)'] = 1 #Female is the default
data.loc[data['Sex'] != "male", 'd1_sex(m)'] = 0 

data.loc[data['Embarked'] == "S", 'd1_Embarked(S)'] = 1 # Q is the default
data.loc[data['Embarked'] != "S", 'd1_Embarked(S)'] = 0 
data.loc[data['Embarked'] == "C", 'd2_Embarked(C)'] = 1 
data.loc[data['Embarked'] != "C", 'd2_Embarked(C)'] = 0 

data.loc[data['Pclass'] == "1", 'd1_Pclass'] = 1 #3 is the default
data.loc[data['Pclass'] != "1", 'd1_Pclass'] = 0 
data.loc[data['Pclass'] == "2", 'd2_Pclass'] = 1 
data.loc[data['Pclass'] != "2", 'd2_Pclass'] = 0 

data.loc[data['SibSp'] == "1", 'd1_SibSp'] = 1 #6 or more is the default
data.loc[data['SibSp'] != "1", 'd1_SibSp'] = 0 
data.loc[data['SibSp'] == "2", 'd2_SibSp'] = 1 
data.loc[data['SibSp'] != "2", 'd2_SibSp'] = 0 
data.loc[data['SibSp'] == "3", 'd3_SibSp'] = 1 
data.loc[data['SibSp'] != "3", 'd3_SibSp'] = 0 
data.loc[data['SibSp'] == "4", 'd4_SibSp'] = 1 
data.loc[data['SibSp'] != "4", 'd4_SibSp'] = 0 
data.loc[data['SibSp'] == "5", 'd5_SibSp'] = 1 
data.loc[data['SibSp'] != "5", 'd5_SibSp'] = 0 

data.loc[data['Parch'] == "1", 'd1_Parch'] = 1 #6 or more is the default
data.loc[data['Parch'] != "1", 'd1_Parch'] = 0 
data.loc[data['Parch'] == "2", 'd2_Parch'] = 1 
data.loc[data['Parch'] != "2", 'd2_Parch'] = 0 
data.loc[data['Parch'] == "3", 'd3_Parch'] = 1 
data.loc[data['Parch'] != "3", 'd3_Parch'] = 0 
data.loc[data['Parch'] == "4", 'd4_Parch'] = 1 
data.loc[data['Parch'] != "4", 'd4_Parch'] = 0 
data.loc[data['Parch'] == "5", 'd5_Parch'] = 1 
data.loc[data['Parch'] != "5", 'd5_Parch'] = 0 

# =============================================================================
# Logistic Regression
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn import metrics 
from sklearn import preprocessing

X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'd1_Embarked(S)', 'd2_Embarked(C)']]
X = scale(X)
y = data.ix


LogReg = LogisticRegression()
LogReg.fit(X,y)

#check for independence between features
sn.regplot(x='Age', y='Fare', data = data, scatter = True )
sn.regplot(x='SibSp', y='Parch', data = data, scatter = True )




