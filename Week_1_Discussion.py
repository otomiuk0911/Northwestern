# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:34:09 2020

@author: o_tom
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

discussion_1 = pd.read_csv("D:/Northwestern/MSDS 422/Week 1/train.csv") 

discussion_1.info()
discussion_1.dtypes
discussion_1.describe()

# Descriptive Statistics for two variables
discussion_1.Age.min()
discussion_1.Age.mean()
discussion_1.Age.max()


discussion_1.Fare.min()
discussion_1.Fare.mean()
discussion_1.Fare.max()


#Histograms for two variables
plt.hist(discussion_1.Fare)      
plt.title('Histogram of Passenger Fares', size=15)
plt.show


plt.hist(discussion_1.Age)      
plt.title('Histogram of Passenger Age', size=15)
plt.show

#Scatter plot of Age and Fare

Age_Fare = sn.scatterplot(x = 'Age',y = 'Fare',data = discussion_1)
plt.title('Scatter of Age and Fare', size=15)
plt.show()
