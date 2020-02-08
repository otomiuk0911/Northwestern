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


data = pd.read_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/train.csv')

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
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
  
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
X = pd.concat([cont_data, int_data], axis=1) #merge cont, int variables
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
# Final Dataset #1
# =============================================================================
final_data = data[['logSalePrice','OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

#Remove Outliers
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(final_data))
print(z)

Q1 = final_data.quantile(0.25)
Q3 = final_data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

final_data1 = final_data[(z < 3).all(axis=1)]

final_data.shape
final_data1.shape



X1 = final_data1[['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

y1 = final_data1['logSalePrice']


#Standardize the Numeric Data - wait to select maybe
from sklearn import preprocessing
# Get column names first
names = X1.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
X1 = scaler.fit_transform(X1)
X1 = pd.DataFrame(X1, columns=names)

X1.shape
y1.shape

# =============================================================================
# Final Dataset #1 Regression #1
# =============================================================================

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

#Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

Model1 = LinearRegression().fit(X1, y1) #training the algorithm
accuracy1 = Model1.score(X1,y1)
print(accuracy1)

y_pred1 = np.exp(Model1.predict(X1))
Model1_df = pd.DataFrame({'Actual': np.exp(final_data1.logSalePrice), 'Predicted': y_pred1})
Model1_df

#quick check on min and max predictions vs actual results
np.exp(final_data1.logSalePrice).min()
y_pred1.min()
np.exp(final_data1.logSalePrice).max()
y_pred1.max()

# Regression metrics
explained_variance1=metrics.explained_variance_score(np.exp(final_data1.logSalePrice), y_pred1)
mean_absolute_error1=metrics.mean_absolute_error(np.exp(final_data1.logSalePrice), y_pred1) 
mse1=metrics.mean_squared_error(np.exp(final_data1.logSalePrice), y_pred1) 
mean_squared_log_error1=metrics.mean_squared_log_error(np.exp(final_data1.logSalePrice), y_pred1)
median_absolute_error1=metrics.median_absolute_error(np.exp(final_data1.logSalePrice), y_pred1)
r21=metrics.r2_score(np.exp(final_data1.logSalePrice), y_pred1)

#Score results
print('explained_variance: ', round(explained_variance1,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error1,4))
print('r2: ', round(r21,4))
print('MAE: ', round(mean_absolute_error1,4))
print('MSE: ', round(mse1,4))
print('RMSE: ', round(np.sqrt(mse1),4))

#Cross Validation
reg_score = []
reg_score=pd.DataFrame(reg_score)
reg_score['CV_MSE'] = cross_val_score(LinearRegression(), X1, y1, scoring='neg_mean_squared_error', cv = 5)
reg_score['CV_RMSE'] = np.sqrt(abs(reg_score.CV_MSE))
print(np.sqrt(abs(reg_score)))

# =============================================================================
# Final Dataset #1 Ridge Regression #1
# =============================================================================

#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge(max_iter=50e5)
parameters = {'alpha': [1e-15, 1e-10, 1e-08, 1e-03,1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]} #Test Multiple Alphas
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X1,y1)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

Model2 = Ridge(alpha=1).fit(X1, y1) #training the algorithm with best paramters
accuracy2 = Model2.score(X1,y1)
print(accuracy2)

y_pred2 = np.exp(Model2.predict(X1))
Model2_df = pd.DataFrame({'Actual': np.exp(final_data1.logSalePrice), 'Predicted': y_pred2})
Model2_df

#quick check on min and max predictions vs actual results
np.exp(final_data1.logSalePrice).min()
y_pred2.min()
np.exp(final_data1.logSalePrice).max()
y_pred2.max()

#Score results
explained_variance2=metrics.explained_variance_score(np.exp(final_data1.logSalePrice), y_pred2)
mean_absolute_error2=metrics.mean_absolute_error(np.exp(final_data1.logSalePrice), y_pred2) 
mse2=metrics.mean_squared_error(np.exp(final_data1.logSalePrice), y_pred2) 
mean_squared_log_error2=metrics.mean_squared_log_error(np.exp(final_data1.logSalePrice), y_pred2)
median_absolute_error2=metrics.median_absolute_error(np.exp(final_data1.logSalePrice), y_pred2)
r22=metrics.r2_score(np.exp(final_data1.logSalePrice), y_pred2)

print('explained_variance: ', round(explained_variance2,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error2,4))
print('r2: ', round(r22,4))
print('MAE: ', round(mean_absolute_error2,4))
print('MSE: ', round(mse2,4))
print('RMSE: ', round(np.sqrt(mse2),4))

#Cross Validation
ridge_score = []
ridge_score=pd.DataFrame(ridge_score)
ridge_score['CV_MSE'] = cross_val_score(Ridge(), X1, y1, scoring='neg_mean_squared_error', cv = 5)
ridge_score['CV_RMSE'] = np.sqrt(abs(ridge_score.CV_MSE))
print(np.sqrt(abs(ridge_score)))

# =============================================================================
# Final Dataset #1 Lasso Regression #1
# =============================================================================

#Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

#Test Multiple parameters
lasso = Lasso(max_iter=50e5)  
parameters = {'alpha': [1e-15, 1e-10, 1e-08, 1e-03,1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)
lasso_regressor.fit(X1,y1)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


Model3 = Lasso(alpha=1e-08, max_iter=50e5).fit(X1, y1) #training the algorithm
accuracy3 = Model3.score(X1,y1)
print(accuracy3)

y_pred3 = np.exp(Model3.predict(X1))
Model3_df = pd.DataFrame({'Actual': np.exp(final_data1.logSalePrice), 'Predicted': y_pred3})
Model3_df

#quick check on min and max predictions vs actual results
np.exp(final_data1.logSalePrice).min()
y_pred3.min()
np.exp(final_data1.logSalePrice).max()
y_pred3.max()

#Score results
explained_variance3=metrics.explained_variance_score(np.exp(final_data1.logSalePrice), y_pred3)
mean_absolute_error3=metrics.mean_absolute_error(np.exp(final_data1.logSalePrice), y_pred3) 
mse3=metrics.mean_squared_error(np.exp(final_data1.logSalePrice), y_pred3) 
mean_squared_log_error3=metrics.mean_squared_log_error(np.exp(final_data1.logSalePrice), y_pred3)
median_absolute_error3=metrics.median_absolute_error(np.exp(final_data1.logSalePrice), y_pred3)
r23=metrics.r2_score(np.exp(final_data1.logSalePrice), y_pred3)

print('explained_variance: ', round(explained_variance3,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error3,4))
print('r2: ', round(r23,4))
print('MAE: ', round(mean_absolute_error3,4))
print('MSE: ', round(mse3,4))
print('RMSE: ', round(np.sqrt(mse3),4))

#Cross Validation
lasso_score = []
lasso_score=pd.DataFrame(lasso_score)
lasso_score['CV_MSE'] = cross_val_score(Lasso(alpha=1e-08, max_iter=50e5), X1, y1, scoring='neg_mean_squared_error', cv = 5)
lasso_score['CV_RMSE'] = np.sqrt(-(lasso_score.CV_MSE))

# =============================================================================
# Final Dataset #1 Random Forest Regression #1
# =============================================================================

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators = 100000, random_state = 0) # number of trees in the forest
RFR.fit(X1, y1)
RFR_pred1 = np.exp(RFR.predict(X1))

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(np.exp(y1), RFR_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(np.exp(y1), RFR_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.exp(y1), RFR_pred1)))

# =============================================================================
# Final Dataset #1 Hyperparameter Tuning Random Forest Regression #1
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV 
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of trees in random forest
max_features = ['auto', 'sqrt'] # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
bootstrap = [True, False] # Method of selecting samples for training each tree

random_grid = {'n_estimators': n_estimators, # Create the random grid
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
RFR = RandomForestRegressor()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = RFR, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X1, y1)

rf_random.best_params_ #get the best parameters

RFR2 = RandomForestRegressor(n_estimators = 2000, max_features = 'sqrt', max_depth = 100, min_samples_split = 5, 
                            min_samples_leaf = 1, bootstrap = True, random_state = 0) 

rf_best = RFR2.fit(X1, y1)
RFR_pred2 = np.exp(RFR2.predict(X1))

print('Mean Absolute Error:', metrics.mean_absolute_error(np.exp(y1), RFR_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(np.exp(y1), RFR_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.exp(y1), RFR_pred2)))

# =============================================================================
# Final Dataset #1 XGBoost #1
# =============================================================================
from sklearn.ensemble import GradientBoostingRegressor

params = {'n_estimators': 100000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GBR = GradientBoostingRegressor(**params)
GBR.fit(X1, y1)
GBR_pred1 = np.exp(GBR.predict(X1))

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(np.exp(y1), GBR_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(np.exp(y1), GBR_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.exp(y1), GBR_pred1)))

# =============================================================================
# Final Dataset #1 Hyperparameter Tuning XGBoost #1
# =============================================================================
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of trees in random forest
max_features = ['auto', 'sqrt'] # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
alpha = [1e-15, 1e-10, 1e-08, 1e-03,1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]

random_grid = {'n_estimators': n_estimators, # Create the random grid
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'alpha': alpha}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
GBR = GradientBoostingRegressor()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
gbr_random = RandomizedSearchCV(estimator = GBR, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
gbr_random.fit(X1, y1)

gbr_random.best_params_ #get the best parameters

GBR2 = GradientBoostingRegressor(n_estimators = 400, max_features = 'sqrt', max_depth = 10, min_samples_split = 5, 
                            min_samples_leaf = 4, random_state = 0, alpha = 1e-15) 

gbr_best = GBR2.fit(X1, y1)
GBR_pred2 = np.exp(GBR2.predict(X1))

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(np.exp(y1), GBR_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(np.exp(y1), GBR_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.exp(y1), GBR_pred2)))

# =============================================================================
# Test the models on test set
# =============================================================================

#Import test set
test = pd.read_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/test.csv')

#create dummies for test set
test['HouseAge'] = test['YrSold'] - test['YearBuilt']
test['Bathrooms'] = test['FullBath'] + 0.5*test['HalfBath']
test['Quality_Index'] = test['OverallQual']*test['OverallCond']

test.loc[data['KitchenQual'] == "Ex", 'd1_KitchenQual'] = 1 #Other is base (Fa & Po)
test.loc[data['KitchenQual'] != "Ex", 'd1_KitchenQual'] = 0 
test.loc[data['KitchenQual'] == "Gd", 'd2_KitchenQual'] = 1 
test.loc[data['KitchenQual'] != "Gd", 'd2_KitchenQual'] = 0 
test.loc[data['KitchenQual'] == "TA", 'd3_KitchenQual'] = 1 
test.loc[data['KitchenQual'] != "TA", 'd3_KitchenQual'] = 0 

test.loc[data['CentralAir'] == "Y", 'd1_CentralAir'] = 1 #No is base
test.loc[data['CentralAir'] != "Y", 'd1_CentralAir'] = 0 

test.loc[data['ExterQual'] == "Ex", 'd1_ExterQual'] = 1 #Other is base (Fa)
test.loc[data['ExterQual'] != "Ex", 'd1_ExterQual'] = 0 
test.loc[data['ExterQual'] == "Gd", 'd2_ExterQual'] = 1 
test.loc[data['ExterQual'] != "Gd", 'd2_ExterQual'] = 0 
test.loc[data['ExterQual'] == "TA", 'd3_ExterQual'] = 1 
test.loc[data['ExterQual'] != "TA", 'd3_ExterQual'] = 0 

test.loc[data['BsmtQual'] == "Ex", 'd1_BsmtQual'] = 1 #Other is base (Fa & TA)
test.loc[data['BsmtQual'] != "Ex", 'd1_BsmtQual'] = 0 
test.loc[data['BsmtQual'] == "Gd", 'd2_BsmtQual'] = 1 
test.loc[data['BsmtQual'] != "Gd", 'd2_BsmtQual'] = 0 

#Select variables for test set
Xtest = test[['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

Xtest = Xtest.fillna(test.mean())

#Standardize the Numeric Data - wait to select maybe
from sklearn import preprocessing
# Get column names first
names = Xtest.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
Xtest = scaler.fit_transform(Xtest)
Xtest = pd.DataFrame(Xtest, columns=names)

y_testpred1 = np.exp(Model1.predict(Xtest))
model1 = pd.DataFrame()
model1['Id'] = test['Id']
model1['SalePrice'] = y_testpred1
model1.to_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/Model1.csv', encoding='utf-8', index=False)

y_testpred2 = np.exp(Model2.predict(Xtest))
model2 = pd.DataFrame()
model2['Id'] = test['Id']
model2['SalePrice'] = y_testpred2
model2.to_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/Model2.csv', encoding='utf-8', index=False)

y_testpred3 = np.exp(Model3.predict(Xtest))
model3 = pd.DataFrame()
model3['Id'] = test['Id']
model3['SalePrice'] = y_testpred3
model3.to_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/Model3.csv', encoding='utf-8', index=False)

y_testpred4 = RFR.predict(Xtest)
model4 = pd.DataFrame()
model4['Id'] = test['Id']
model4['SalePrice'] = np.exp(y_testpred4)
model4.to_csv('D:/Northwestern/MSDS 422/Week 4/Ames 2 Assignment 4/Model4.csv', encoding='utf-8', index=False)

y_testpred5 = RFR2.predict(Xtest)
model5 = pd.DataFrame()
model5['Id'] = test['Id']
model5['SalePrice'] = np.exp(y_testpred5)
model5.to_csv('D:/Northwestern/MSDS 422/Week 4/Ames 2 Assignment 4/Model5.csv', encoding='utf-8', index=False)

y_testpred6 = GBR.predict(Xtest)
model6 = pd.DataFrame()
model6['Id'] = test['Id']
model6['SalePrice'] = np.exp(y_testpred6)
model6.to_csv('D:/Northwestern/MSDS 422/Week 4/Ames 2 Assignment 4/Model6.csv', encoding='utf-8', index=False)

y_testpred7 = GBR2.predict(Xtest)
model7 = pd.DataFrame()
model7['Id'] = test['Id']
model7['SalePrice'] = np.exp(y_testpred7)
model7.to_csv('D:/Northwestern/MSDS 422/Week 4/Ames 2 Assignment 4/Model7.csv', encoding='utf-8', index=False)

# =============================================================================
# 
# =============================================================================









GBR.fit(RFR_X1, RFR_y1)
mse = mean_squared_error(RFR_y1, GBR.predict(RFR_X1))
print("MSE: %.4f" % mse)


# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(GBR.staged_predict(RFR_X1)):
    test_score[i] = GBR.loss_(RFR_y1, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, GBR.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# Plot feature importance
feature_importance = GBR.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, RFR_X1.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# =============================================================================
# Test the models on test set
# =============================================================================

test = pd.read_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/test.csv')

test['HouseAge'] = test['YrSold'] - test['YearBuilt']
test['Bathrooms'] = test['FullBath'] + 0.5*test['HalfBath']
test['Quality_Index'] = test['OverallQual']*test['OverallCond']

test.loc[data['KitchenQual'] == "Ex", 'd1_KitchenQual'] = 1 #Other is base (Fa & Po)
test.loc[data['KitchenQual'] != "Ex", 'd1_KitchenQual'] = 0 
test.loc[data['KitchenQual'] == "Gd", 'd2_KitchenQual'] = 1 
test.loc[data['KitchenQual'] != "Gd", 'd2_KitchenQual'] = 0 
test.loc[data['KitchenQual'] == "TA", 'd3_KitchenQual'] = 1 
test.loc[data['KitchenQual'] != "TA", 'd3_KitchenQual'] = 0 

test.loc[data['CentralAir'] == "Y", 'd1_CentralAir'] = 1 #No is base
test.loc[data['CentralAir'] != "Y", 'd1_CentralAir'] = 0 

test.loc[data['ExterQual'] == "Ex", 'd1_ExterQual'] = 1 #Other is base (Fa)
test.loc[data['ExterQual'] != "Ex", 'd1_ExterQual'] = 0 
test.loc[data['ExterQual'] == "Gd", 'd2_ExterQual'] = 1 
test.loc[data['ExterQual'] != "Gd", 'd2_ExterQual'] = 0 
test.loc[data['ExterQual'] == "TA", 'd3_ExterQual'] = 1 
test.loc[data['ExterQual'] != "TA", 'd3_ExterQual'] = 0 

test.loc[data['BsmtQual'] == "Ex", 'd1_BsmtQual'] = 1 #Other is base (Fa & TA)
test.loc[data['BsmtQual'] != "Ex", 'd1_BsmtQual'] = 0 
test.loc[data['BsmtQual'] == "Gd", 'd2_BsmtQual'] = 1 
test.loc[data['BsmtQual'] != "Gd", 'd2_BsmtQual'] = 0 

Xtest = test[['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

Xtest = Xtest.fillna(test.mean())

#Standardize the Numeric Data - wait to select maybe
from sklearn import preprocessing
# Get column names first
names = Xtest.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
Xtest = scaler.fit_transform(Xtest)
Xtest = pd.DataFrame(Xtest, columns=names)

y_testpred1 = np.exp(Model1.predict(Xtest))
model1 = pd.DataFrame()
model1['Id'] = test['Id']
model1['SalePrice'] = y_testpred1
model1.to_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/Model1.csv', encoding='utf-8', index=False)

y_testpred2 = np.exp(Model2.predict(Xtest))
model2 = pd.DataFrame()
model2['Id'] = test['Id']
model2['SalePrice'] = y_testpred2
model2.to_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/Model2.csv', encoding='utf-8', index=False)

y_testpred3 = np.exp(Model3.predict(Xtest))
model3 = pd.DataFrame()
model3['Id'] = test['Id']
model3['SalePrice'] = y_testpred3
model3.to_csv('D:/Northwestern/MSDS 422/Week 2/Ames Housing Assignment 2/Model3.csv', encoding='utf-8', index=False)

y_testpred4 = np.exp(RFR.predict(Xtest_RFR))
model4 = pd.DataFrame()
model4['Id'] = test['Id']
model4['SalePrice'] = y_testpred4
model4.to_csv('D:/Northwestern/MSDS 422/Week 4/Ames 2 Assignment 4/Model4.csv', encoding='utf-8', index=False)

# =============================================================================
# Randon Forest Regression
# =============================================================================
final_data_RFR1 = data[['logSalePrice','OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

RFR_X1 = data[['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

RFR_y1 = data.SalePrice

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators = 1800, random_state = 0) # number of trees in the forest
RFR.fit(RFR_X1, RFR_y1)
RFR_pred1 = RFR.predict(RFR_X1)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(RFR_y1, RFR_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(RFR_y1, RFR_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(RFR_y1, RFR_pred1)))

#Run model on test set

Xtest_RFR = test[['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

Xtest_RFR = Xtest_RFR.fillna(test.mean())

y_testpred4 = RFR.predict(Xtest_RFR)
model4 = pd.DataFrame()
model4['Id'] = test['Id']
model4['SalePrice'] = y_testpred4
model4.to_csv('D:/Northwestern/MSDS 422/Week 4/Ames 2 Assignment 4/Model4.csv', encoding='utf-8', index=False)

# =============================================================================
# Hyperparameter Tuning the Random Forest in Python
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV 
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of trees in random forest
max_features = ['auto', 'sqrt'] # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
bootstrap = [True, False] # Method of selecting samples for training each tree
random_grid = {'n_estimators': n_estimators, # Create the random grid
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
RFR = RandomForestRegressor()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = RFR, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(RFR_X1, RFR_y1)

rf_random.best_params_ #get the best parameters

#recreate random forest model using best parameters
n_estimators = 1800
max_features = 'sqrt'
max_depth = 30
min_samples_split = 10
min_samples_leaf = 1
bootstrap = False

RFR2 = RandomForestRegressor(n_estimators = 1800, max_features = 'sqrt', max_depth = 30, min_samples_split = 10, 
                            min_samples_leaf = 1, bootstrap = False, random_state = 0) 

rf_best = RFR2.fit(RFR_X1, RFR_y1)
RFR_pred2 = RFR2.predict(RFR_X1)

print('Mean Absolute Error:', metrics.mean_absolute_error(RFR_y1, RFR_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(RFR_y1, RFR_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(RFR_y1, RFR_pred2)))



# =============================================================================
# XGBoost
# =============================================================================
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GBR = ensemble.GradientBoostingRegressor(**params)

GBR.fit(RFR_X1, RFR_y1)
mse = mean_squared_error(RFR_y1, GBR.predict(RFR_X1))
print("MSE: %.4f" % mse)


# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(GBR.staged_predict(RFR_X1)):
    test_score[i] = GBR.loss_(RFR_y1, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, GBR.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# Plot feature importance
feature_importance = GBR.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, RFR_X1.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()



RFR_X1 = data[['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

RFR_y1 = data.SalePrice

params = {'n_estimators': 200000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GBR = ensemble.GradientBoostingRegressor(**params)
GBR.fit(RFR_X1, RFR_y1)
GBR_pred1 = GBR.predict(RFR_X1)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(RFR_y1, GBR_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(RFR_y1, GBR_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(RFR_y1, GBR_pred1)))

#Run model on test set

Xtest_RFR = test[['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'Bathrooms', 'Quality_Index', 'Fireplaces', 'd1_KitchenQual', 'd2_KitchenQual',
          'd3_KitchenQual', 'd1_CentralAir', 'd1_ExterQual', 'd2_ExterQual','d3_ExterQual','d1_BsmtQual','d2_BsmtQual']]

Xtest_RFR = Xtest_RFR.fillna(test.mean())

y_testpred5 = GBR.predict(Xtest_RFR)
model5 = pd.DataFrame()
model5['Id'] = test['Id']
model5['SalePrice'] = y_testpred5
model5.to_csv('D:/Northwestern/MSDS 422/Week 4/Ames 2 Assignment 4/Model5.csv', encoding='utf-8', index=False)




















