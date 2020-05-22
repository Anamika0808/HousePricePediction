# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:43:54 2020

@author: 91947
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\91947\Downloads\house-prices-advanced-regression-techniques\train.csv')
df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]
#print(df2)

#print("List of dropped columns:", end=" ")
for c in df.columns:
    if c not in df2.columns:
        print(c, end=", ")
print('\n')
df = df2
df.drop(['Id'], axis = 1, inplace = True)
#df.drop(['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
#print(df.info())

#print(df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
#sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

list(set(df.dtypes.tolist()))

df_num = df.select_dtypes(include = ['float64', 'int64'])
print(df_num.head())

df_num.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8)

df_num_corr = df_num.corr()['SalePrice'][:-1]
#print(abs(df_num_corr))

golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending = False)
#print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
##for i in range(0, len(df_num.columns), 5):
    #sns.pairplot(data=df_num,
                #x_vars=df_num.columns[i:i+5],
                #y_vars=['SalePrice'])
    

import operator

individual_features_df = []
for i in range(0, len(df_num.columns) - 1): # -1 because the last column is SalePrice
    tmpDf = df_num[[df_num.columns[i], 'SalePrice']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
#print(all_correlations)
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))
    
golden_feature_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
corr = df_num.drop('SalePrice', axis=1).corr() # We already examined SalePrice correlations
#plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5)|(corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
                 
quantitative_features_list = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']
df_quantitative_values = df[quantitative_features_list]

features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]
features_to_analyse.append('SalePrice')

df_features_to_analyse = df[features_to_analyse]

x = df_features_to_analyse.drop(['SalePrice'], axis = 1)
y = df_features_to_analyse['SalePrice']

x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
reg = LinearRegression()
reg.fit(x_train, y_train)
accuracy = reg.score(x_test, y_test)
y_pred= reg.predict(x_test)