# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 05:30:47 2020

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


data = pd.read_csv('D:/Northwestern/MSDS 422/Week 2/Assignment 2/train.csv')

#Data overview
print(data.info())
data.dtypes
print(data.describe())
data.head(n=4)


#Create new variables from existing data
data['HouseAge'] = data['YrSold'] - data['YearBuilt']
data['logSalePrice'] = np.log(data['SalePrice'])
data['Bathrooms'] = data['FullBath'] + 0.5*data['HalfBath']
data['Quality_Index'] = data['OverallQual']*data['OverallCond']

#Deal with missing values
data.drop(columns =['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], inplace = True) #too many missing values in columns
data = data.fillna(data.mean())
#EDA .............................

#Graphing
    
#Correlation Matrix
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(30,30))
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

#EDA for Categorical variables
cat_data = data.select_dtypes(include = 'object')
cat_data['SalePrice'] = data['SalePrice']

cols = ['SalePrice']  + [col for col in cat_data if col != 'SalePrice']
cat_data = cat_data[cols]

for column in cat_data.columns[1:]:
    sn.set()
    fig, ax = plt.subplots()
    sn.set(style="ticks")
    sn.boxplot(x=column, y='SalePrice', data=cat_data)
    sn.despine(offset=10, trim=True)
    fig.set_size_inches(6,6)
    
#EDA for Continuous Variables
cont_data = data.select_dtypes(include = ['float64'])
cont_data['SalePrice'] = data['SalePrice']

cols = ['SalePrice']  + [col for col in cont_data if col != 'SalePrice']
cont_data = cont_data[cols]

for column in cont_data.columns[1:]:
    sn.set()
    fig, ax = plt.subplots()
    sn.set(style="ticks")
    sn.scatterplot(x=column, y='SalePrice', data=cont_data)
    sn.despine(offset=10, trim=True)
    fig.set_size_inches(6,6)
    
    
#EDA for Integer Variables
int_data = data.select_dtypes(include = ['int64'])
int_data['SalePrice'] = data['SalePrice']

cols = ['SalePrice']  + [col for col in int_data if col != 'SalePrice']
int_data = int_data[cols]

for column in int_data.columns[1:]:
    sn.set()
    fig, ax = plt.subplots()
    sn.set(style="ticks")
    sn.scatterplot(x=column, y='SalePrice', data=int_data)
    sn.despine(offset=10, trim=True)
    fig.set_size_inches(6,6)

for column in int_data.columns[1:]:
    sn.set()
    fig, ax = plt.subplots()
    sn.set(style="ticks")
    sn.boxplot(x=column, y='SalePrice', data=int_data)
    sn.despine(offset=10, trim=True)
    fig.set_size_inches(6,6)

# =============================================================================
# Review research design and modeling methods (10 points)
# =============================================================================
#Selecting Numerical Variables  
X = pd.concat([cont_data, int_data], axis=1) #merge cont, int and dummies
X = X.drop(['SalePrice', 'logSalePrice'], axis = 1)
y = data['SalePrice']

#Feature Selection Using SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression


select_feature = SelectKBest(score_func=f_regression, k=20).fit(X,y)
selected_features = pd.DataFrame({'Feature':list(X.columns),'Scores':select_feature.scores_})
selected_features = selected_features.sort_values(by='Scores', ascending=False)

#Feature importance from ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#Selecting Categorical Variables 
data.loc[data['KitchenQual'] == "Ex", 'd1_KitchenQual'] = 1 #Other is base (Fa & Po)
data.loc[data['KitchenQual'] != "Ex", 'd1_KitchenQual'] = 0 
data.loc[data['KitchenQual'] == "Gd", 'd2_KitchenQual'] = 1 
data.loc[data['KitchenQual'] != "Gd", 'd2_KitchenQual'] = 0 
data.loc[data['KitchenQual'] == "TA", 'd3_KitchenQual'] = 1 
data.loc[data['KitchenQual'] != "TA", 'd3_KitchenQual'] = 0 

data.loc[data['CentralAir'] == "Y", 'd1_CentralAir'] = 1 #No is base
data.loc[data['CentralAir'] != "Y", 'd1_CentralAir'] = 0 

data.loc[data['ExterQual'] == "Ex", 'd1_ExterQual'] = 1 #Other is base (Fa)
data.loc[data['ExterQual'] != "Ex", 'd1_ExterQual'] = 0 
data.loc[data['ExterQual'] == "Gd", 'd2_ExterQual'] = 1 
data.loc[data['ExterQual'] != "Gd", 'd2_ExterQual'] = 0 
data.loc[data['ExterQual'] == "TA", 'd3_ExterQual'] = 1 
data.loc[data['ExterQual'] != "TA", 'd3_ExterQual'] = 0 

data.loc[data['BsmtQual'] == "Ex", 'd1_BsmtQual'] = 1 #Other is base (Fa & TA)
data.loc[data['BsmtQual'] != "Ex", 'd1_BsmtQual'] = 0 
data.loc[data['BsmtQual'] == "Gd", 'd2_BsmtQual'] = 1 
data.loc[data['BsmtQual'] != "Gd", 'd2_BsmtQual'] = 0 


# =============================================================================
# Getting to the final Model
# =============================================================================
final_data = data[['logSalePrice','OverallQual', 'GrLivArea', 'GarageArea', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]




X = final_data[['OverallQual', 'GrLivArea', 'GarageArea', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

y = final_data['logSalePrice']


#Standardize the Numeric Data - wait to select maybe
from sklearn import preprocessing
# Get column names first
names = X.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=names)

X.shape
y.shape

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

#Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

 
Model1 = LinearRegression().fit(X, y) #training the algorithm
accuracy1 = Model1.score(X,y)
print(accuracy1)

y_pred1 = np.exp(Model1.predict(X))
Model1_df = pd.DataFrame({'Actual': data.SalePrice, 'Predicted': y_pred1})
Model1_df

# Regression metrics
explained_variance1=metrics.explained_variance_score(data.SalePrice, y_pred1)
mean_absolute_error1=metrics.mean_absolute_error(data.SalePrice, y_pred1) 
mse1=metrics.mean_squared_error(data.SalePrice, y_pred1) 
mean_squared_log_error1=metrics.mean_squared_log_error(data.SalePrice, y_pred1)
median_absolute_error1=metrics.median_absolute_error(data.SalePrice, y_pred1)
r21=metrics.r2_score(data.SalePrice, y_pred1)

data.SalePrice.min()
y_pred1.min()

print('explained_variance: ', round(explained_variance1,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error1,4))
print('r2: ', round(r21,4))
print('MAE: ', round(mean_absolute_error1,4))
print('MSE: ', round(mse1,4))
print('RMSE: ', round(np.sqrt(mse1),4))


reg_score = []
reg_score=pd.DataFrame(reg_score)
reg_score['CV_MSE'] = cross_val_score(LinearRegression(), X, y, scoring='neg_mean_squared_error', cv = 5)
reg_score['CV_RMSE'] = np.sqrt(-(reg_score.CV_MSE))


#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

####Try this
ridge=Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-08, 1e-03,1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X,y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

##try this
Model2 = Ridge(alpha=0.01).fit(X, y) #training the algorithm
accuracy2 = Model2.score(X,y)
print(accuracy2)

y_pred2 = np.exp(Model2.predict(X))
Model2_df = pd.DataFrame({'Actual': data.SalePrice, 'Predicted': y_pred2})
Model2_df


ridge_score = []
ridge_score=pd.DataFrame(ridge_score)
ridge_score['CV_MSE'] = cross_val_score(Ridge(), X, y, scoring='neg_mean_squared_error', cv = 5)
ridge_score['CV_RMSE'] = np.sqrt(-(ridge_score.CV_MSE))

explained_variance2=metrics.explained_variance_score(data.SalePrice, y_pred2)
mean_absolute_error2=metrics.mean_absolute_error(data.SalePrice, y_pred2) 
mse2=metrics.mean_squared_error(data.SalePrice, y_pred2) 
mean_squared_log_error2=metrics.mean_squared_log_error(data.SalePrice, y_pred2)
median_absolute_error2=metrics.median_absolute_error(data.SalePrice, y_pred2)
r22=metrics.r2_score(data.SalePrice, y_pred2)

data.SalePrice.min()
y_pred2.min()

print('explained_variance: ', round(explained_variance2,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error2,4))
print('r2: ', round(r22,4))
print('MAE: ', round(mean_absolute_error2,4))
print('MSE: ', round(mse2,4))
print('RMSE: ', round(np.sqrt(mse2),4))


#Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso = Lasso(max_iter=50e5)  
parameters = {'alpha': [1e-15, 1e-10, 1e-08, 1e-03,1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)
lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


Model3 = Lasso(alpha=1e-15, max_iter=50e5).fit(X, y) #training the algorithm
accuracy3 = Model3.score(X,y)
print(accuracy3)

y_pred3 = np.exp(Model3.predict(X))
Model3_df = pd.DataFrame({'Actual': data.SalePrice, 'Predicted': y_pred3})
Model3_df


lasso_score = []
lasso_score=pd.DataFrame(lasso_score)
lasso_score['CV_MSE'] = cross_val_score(Lasso(alpha=1e-15, max_iter=50e5), X, y, scoring='neg_mean_squared_error', cv = 5)
lasso_score['CV_RMSE'] = np.sqrt(-(lasso_score.CV_MSE))

explained_variance3=metrics.explained_variance_score(data.SalePrice, y_pred3)
mean_absolute_error3=metrics.mean_absolute_error(data.SalePrice, y_pred3) 
mse3=metrics.mean_squared_error(data.SalePrice, y_pred3) 
mean_squared_log_error3=metrics.mean_squared_log_error(data.SalePrice, y_pred3)
median_absolute_error3=metrics.median_absolute_error(data.SalePrice, y_pred3)
r23=metrics.r2_score(data.SalePrice, y_pred3)

data.SalePrice.min()
y_pred3.min()

print('explained_variance: ', round(explained_variance3,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error3,4))
print('r2: ', round(r23,4))
print('MAE: ', round(mean_absolute_error3,4))
print('MSE: ', round(mse3,4))
print('RMSE: ', round(np.sqrt(mse3),4))








# =============================================================================
# Test the models on test set
# =============================================================================

test = pd.read_csv('D:/Northwestern/MSDS 422/Week 2/Assignment 2/test.csv')

test['HouseAge'] = test['YrSold'] - test['YearBuilt']
test['Bathrooms'] = test['FullBath'] + 0.5*test['HalfBath']


#Deal with missing values
test.drop(columns =['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace = True) #too many missing values in columns
test = test.fillna(test.mean())

test.shape

print(test.info())
test.dtypes
print(test.describe())
test.head(n=4)
#EDA .............................


ynew = Model1.predict(Xnew)
Xnew=pd.get_dummies(test)
