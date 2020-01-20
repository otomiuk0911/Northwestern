# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Based off of Jump-Start Example: Python analysis of MSPA Software Survey by Tom Miller and Kelsey O'Neil
# Update 2020-01-12 by Olaf Tomiuk new scatterplots, new bar charts, new histograms, new courses_completed column, new extra transformation code added



# =============================================================================
# Data Preparation
# =============================================================================

# external libraries for visualizations and data manipulation
# ensure that these packages have been installed prior to calls
#Import packages
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#Import Data
data = pd.read_csv("D:/Northwestern/MSDS 422/Week 1/mspa-software-survey-case-python-v005/mspa-survey-data.csv") 

#Data overview
data.info()
data.dtypes
data.describe()
data.head(n=4)

#Rename columns to shorter, more usable format
data = data.rename(index=str, columns={
    'Personal_JavaScalaSpark': 'My_Java',
    'Personal_JavaScriptHTMLCSS': 'My_JS',
    'Personal_Python': 'My_Python',
    'Personal_R': 'My_R',
    'Personal_SAS': 'My_SAS',
    'Professional_JavaScalaSpark': 'Prof_Java',
    'Professional_JavaScriptHTMLCSS': 'Prof_JS',
    'Professional_Python': 'Prof_Python',
    'Professional_R': 'Prof_R',
    'Professional_SAS': 'Prof_SAS',
    'Industry_JavaScalaSpark': 'Ind_Java',
    'Industry_JavaScriptHTMLCSS': 'Ind_JS',
    'Industry_Python': 'Ind_Python',
    'Industry_R': 'Ind_R',
    'Industry_SAS': 'Ind_SAS'})

#Create new column to count programming courses  (slightly different than courses completed)
data['Prog_Courses_Completed'] = data[['PREDICT400','PREDICT401','PREDICT410','PREDICT411', 'PREDICT413','PREDICT420', 'PREDICT422','PREDICT450', 'PREDICT451','PREDICT452', 'PREDICT453','PREDICT454','PREDICT455','PREDICT456','PREDICT457','OtherPython', 'OtherR', 'OtherSAS', 'Other']].count(axis=1)

#fix NA values in "Courses_Completed" column with value from Prog_Courses_Completed column
data['Courses_Completed'].fillna(data['Prog_Courses_Completed'], inplace=True)

#Fix NA values for Course Interest Columns
data[['Python_Course_Interest','Foundations_DE_Course_Interest', 'Analytics_App_Course_Interest','Systems_Analysis_Course_Interest']] = data[['Python_Course_Interest','Foundations_DE_Course_Interest', 'Analytics_App_Course_Interest','Systems_Analysis_Course_Interest']].fillna(data.mean())

#Check to make sure nan values are replaced with the mean for their respective columns
data.info()

# =============================================================================
# EDA & Graphing
# =============================================================================
#Correlation Matrix
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
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


#Min, avg, max values for Software preferences
Preferences = pd.DataFrame()
Preferences['Software'] = []
Preferences['Min'] = []
Preferences['Mean'] = []
Preferences['Max'] = []

# define subset DataFrame for analysis of software preferences 
software_df = data.loc[:, 'My_Java':'Ind_SAS']

for i in range(15):
            Preferences = Preferences.append(pd.Series([software_df.columns[i], (software_df[software_df.columns[i]]).min(), (software_df[software_df.columns[i]]).mean(), (software_df[software_df.columns[i]]).max()], index=['Software', 'Min', 'Mean', 'Max']),ignore_index=True)     

#Break preferences out into personal, professional, and Industry
pers_pref = Preferences.loc[0:4]     
prof_pref = Preferences.loc[5:9]   
ind_pref = Preferences.loc[10:14]   

#plot preferences
pers_pref.plot(x="Software", y=["Min", "Mean", "Max"], kind="bar")
plt.title('Personal Software Preferences', size=15)
plt.show

prof_pref.plot(x="Software", y=["Min", "Mean", "Max"], kind="bar")
plt.title('Professional Software Preferences', size=15)
plt.show

ind_pref.plot(x="Software", y=["Min", "Mean", "Max"], kind="bar")
plt.title('Industry Software Preferences', size=15)
plt.show


#Create Labels for future plots
survey_df_labels = [
    'Personal Preference for Java/Scala/Spark',
    'Personal Preference for Java/Script/HTML/CSS',
    'Personal Preference for Python',
    'Personal Preference for R',
    'Personal Preference for SAS',
    'Professional Java/Scala/Spark',
    'Professional JavaScript/HTML/CSS',
    'Professional Python',
    'Professional R',
    'Professional SAS',
    'Industry Java/Scala/Spark',
    'Industry Java/Script/HTML/CSS',
    'Industry Python',
    'Industry R',
    'Industry SAS'        
]   

#Scatter Plots
# create a set of scatter plots for personal preferences
for i in range(5):
    for j in range(5):
        if i != j:
            file_title = software_df.columns[i] + '_and_' + software_df.columns[j]
            plot_title = software_df.columns[i] + ' and ' + software_df.columns[j]
            fig, axis = plt.subplots()
            axis.set_xlabel(survey_df_labels[i])
            axis.set_ylabel(survey_df_labels[j])
            plt.title(plot_title)
            scatter_plot = axis.scatter(software_df[software_df.columns[i]], 
            software_df[software_df.columns[j]],
            facecolors = 'none', 
            edgecolors = 'blue') 
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)  
            
# create a set of scatter plots for Professional preferences
for i in range(5,10):
    for j in range(5,10):
        if i != j:
            file_title = software_df.columns[i] + '_and_' + software_df.columns[j]
            plot_title = software_df.columns[i] + ' and ' + software_df.columns[j]
            fig, axis = plt.subplots()
            axis.set_xlabel(survey_df_labels[i])
            axis.set_ylabel(survey_df_labels[j])
            plt.title(plot_title)
            scatter_plot = axis.scatter(software_df[software_df.columns[i]], 
            software_df[software_df.columns[j]],
            facecolors = 'none', 
            edgecolors = 'blue') 
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)  
            
# create a set of scatter plots for industry preferences          
for i in range(10,15):
    for j in range(10,15):
        if i != j:
            file_title = software_df.columns[i] + '_and_' + software_df.columns[j]
            plot_title = software_df.columns[i] + ' and ' + software_df.columns[j]
            fig, axis = plt.subplots()
            axis.set_xlabel(survey_df_labels[i])
            axis.set_ylabel(survey_df_labels[j])
            plt.title(plot_title)
            scatter_plot = axis.scatter(software_df[software_df.columns[i]], 
            software_df[software_df.columns[j]],
            facecolors = 'none', 
            edgecolors = 'blue') 
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)  
            
            
# Create a set of Histograms for Preferences
        
for i in range(15):
            file_title = software_df.columns[i] 
            plot_title = software_df.columns[i] 
            fig, axis = plt.subplots()
            axis.set_xlabel(survey_df_labels[i])
            axis.set_ylabel('count')
            plt.title(plot_title)
            histogram = plt.hist(data[software_df.columns[i]]),
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)  
            

#Create histgram for number of programming courses taken
courses_hist = plt.hist(data['Prog_Courses_Completed'])      
plt.title('Number of Programming Courses Completed', size=15)
plt.show

# Check mode and create histogram for number of courses completed

data['Courses_Completed'].mode()

#bins specified
courses_hist = plt.hist(data['Courses_Completed'], bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])      
plt.title('Number of Courses Completed', size=15)
plt.show

#bins not specified
courses_hist = plt.hist(data['Courses_Completed'])      
plt.title('Number of Courses Completed (Bins not specified)', size=15)
plt.show

#Create Histogram for new course interest
#Python_Course_Interest
courses_hist = plt.hist(data['Python_Course_Interest'])      
plt.title('Python_Course_Interest', size=15)
plt.show

#Data Engineering Course Interest
courses_hist = plt.hist(data['Foundations_DE_Course_Interest'])      
plt.title('Foundations_DE_Course_Interest', size=15)
plt.show

# Analytics_App_Course_Interest
courses_hist = plt.hist(data['Analytics_App_Course_Interest'])      
plt.title('Analytics_App_Course_Interest', size=15)
plt.show

#Systems_Analysis_Course_Interest
courses_hist = plt.hist(data['Systems_Analysis_Course_Interest'])      
plt.title('Systems_Analysis_Course_Interest', size=15)
plt.show

#plot new course interest
Interest = pd.DataFrame()
Interest['Course_Interest'] = []
Interest['Min'] = []
Interest['Mean'] = []
Interest['Max'] = []

Course_Interest = data.loc[:, 'Python_Course_Interest':'Systems_Analysis_Course_Interest']

Course_Interest = Course_Interest.rename(index=str, columns={
    'Python_Course_Interest': 'Python_Course',
    'Foundations_DE_Course_Interest': 'Foundations_DE',
    'Analytics_App_Course_Interest': 'Analytics_App',
    'Systems_Analysis_Course_Interest': 'System_Analysis'})

for i in range(4):
            Interest = Interest.append(pd.Series([Course_Interest.columns[i], (Course_Interest[Course_Interest.columns[i]]).min(), (Course_Interest[Course_Interest.columns[i]]).mean(), (Course_Interest[Course_Interest.columns[i]]).max()], index=['Course_Interest', 'Min', 'Mean', 'Max']),ignore_index=True)     

Interest.plot(x="Course_Interest", y=["Min", "Mean", "Max"], kind="bar")
plt.title('New Course Interest', size=15)
plt.show

# =============================================================================
# Variable Transformations and Scaling
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

# transformations a la Scikit Learn
# select variable to examine, eliminating missing data codes
X = data['Courses_Completed'].dropna()
X = np.array(X)
X = X.reshape(-1,1)

# Seaborn provides a convenient way to show the effects of transformations
# on the distribution of values being transformed
# Documentation at https://seaborn.pydata.org/generated/seaborn.distplot.html

unscaled_fig, ax = plt.subplots()
sn.distplot(X).set_title('Unscaled')
unscaled_fig.savefig('Transformation-Unscaled' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

standard_fig, ax = plt.subplots()
sn.distplot(StandardScaler().fit_transform(X)).set_title('StandardScaler')
standard_fig.savefig('Transformation-StandardScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

minmax_fig, ax = plt.subplots()
sn.distplot(MinMaxScaler().fit_transform(X)).set_title('MinMaxScaler')
minmax_fig.savefig('Transformation-MinMaxScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None) 

#Log doesn't work with 0s in the dataset
 
log_fig, ax = plt.subplots()
sn.distplot(np.log(X)).set_title('NaturalLog')
log_fig.savefig('Transformation-NaturalLog' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None) 

#new scaling techniques

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler

maxabs_fig, ax = plt.subplots()
sn.distplot(MaxAbsScaler().fit_transform(X)).set_title('MaxAbsScaler')
maxabs_fig.savefig('Transformation-MaxAbsScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None) 

PowerTrans_fig, ax = plt.subplots()
sn.distplot(PowerTransformer(method='yeo-johnson').fit_transform(X)).set_title('PowerTransformer')
PowerTrans_fig.savefig('Transformation-PowerTransformer' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None) 

Robust_fig, ax = plt.subplots()
sn.distplot(RobustScaler().fit_transform(X)).set_title('RobustScaler')
Robust_fig.savefig('Transformation-RobustScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None) 


















