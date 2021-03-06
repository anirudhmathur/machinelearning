#https://www.kaggle.com/c/home-data-for-ml-course/data?select=train.csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import  r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("train.csv")
print(df.head())

#print all features
features = []
features = list(df.columns)
print()
print('#'*5,'No of features in training data',len(features))
print(features)
print('-'*100)
print()

#number of numeric features 
num_fet = []
num_fet = list(df.select_dtypes(exclude = 'object').columns)
print()
print('#'*5,'No of numerical features in training data',len(num_fet))
print(num_fet)
print('-'*100)
print()

#number of Categorical  features 
cat_fet = []
cat_fet = list(df.select_dtypes(include = 'object').columns)
print()
print('#'*5,'No of Categorical  features in training data',len(cat_fet))
print(cat_fet)
print('-'*100)
print()

# Some statiscs for numerical cols
print(df.describe().T)

# Some statiscs for all cols
print()
print()
print(df.describe(include= 'all').T)

# Printing a specific column
print(df['SalePrice'][:7]) # Printing top 7 rows

print('-'*50)

# Printing some specific rows and columns
print(df.iloc[10:14,-3:]) # The last three features and 10 to 13 rows
# Here iloc stands for Inverted Letter Of Credit a mehod to Select row or col

print('-'*50)

# Printing some rows and cols using specific cols name
print(df.loc[90:95,'SalePrice'])

# Lets see some row by applying condition
print(df.loc[(df.SaleType == 'New') & (df.YrSold == 2007) & (df.MSZoning=='FV')])

#features which have missing values
for col in features:
    if df[col].isnull().sum() > 0:
        print(str(col+' '*2)+str('-'*7)+str('->  ')+ str(df[col].isnull().sum()))


print()
print()


#Now, lets give an another category named Unknown for every categorical features that have missing values
for col in cat_fet:
    if df[col].isnull().sum() > 0:
        print(str(col+' '*2)+str('-'*7)+str('->  ')+ str(df[col].isnull().sum()))

for col in cat_fet:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna('Unknown')
#For handling the numercial missing values we will use the features mean
for col in num_fet:
    if df[col].isnull().sum() > 0:
        feature_mean = df[col].mean()
        df[col].replace(np.nan,feature_mean,inplace = True)

#finally check if there is anymore missing values.
print(df.isnull().values.any())

print(df.tail())

#Handling categorical features
#for col in cat_fet:
 #   df[col] = df[col].astype('category').cat.codes + 1
for col in cat_fet:
   # print('*'*30)
   # print(type(df[col]))
    le = LabelEncoder()    
    print(df[col])
    #df[col] = np.array(le.fit_transform(df[col]))
    df[col] = df[col].astype('category').cat.codes + 1
    #print(df[col])
    #print('*'*30)

print(df.head())


#create training data
#X_train = df.iloc[:,:-1].values
cols = ['Alley', 'Fence', 'FireplaceQu', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
        'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType',
        'GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
X_train = df[cols]
print("creating training data")
print(X_train)
#y_train = df.iloc[:,-1].values
y_train = df['SalePrice']
print(y_train)

regressor = RandomForestRegressor(n_estimators = 700, max_features='log2', max_samples = 0.9, n_jobs = -1)
regressor.fit(X_train, y_train)
print(regressor.score(X_train,y_train))

############################Test Data**************************************
df_test = pd.read_csv("test.csv")
print(df_test)

features_test = []
fetures_test = list(df_test.columns) 

#The numeric features
num_fet_test = []
num_fet_test = list(df_test.select_dtypes(exclude = 'object').columns)

#The Categorical features
cat_fet_test = []
cat_fet_test = list(df_test.select_dtypes(include = 'object').columns)

for col in cat_fet_test:
    if df_test[col].isnull().sum() > 0:
        df_test[col] = df_test[col].fillna('Unknown')


for col in num_fet_test:
    if df_test[col].isnull().sum() > 0:
        feature_mean = df_test[col].mean()
        df_test[col].replace(np.nan,feature_mean,inplace = True)

for col in cat_fet_test:
    df_test[col] = df_test[col].astype('category').cat.codes + 1

print(df_test.head())

X_test = df_test.iloc[:, :].values
cols = ['Alley', 'Fence', 'FireplaceQu', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
        'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType',
        'GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
X_test = df[cols]

#y_test = df_test.iloc[:, -1].values
y_pred = regressor.predict(X_test)
#print(r2_score(y_test, y_pred))
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(y_pred)





