#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split

# Metrics for Evaluation of model Accuracy and F1-score
from sklearn.metrics  import f1_score,accuracy_score

#Importing the KNN from scikit-learn library
from sklearn.neighbors import KNeighborsClassifier
 
#Importing the Decision Tree from scikit-learn library
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

filepath = '/Users/apple/Documents/Kaggle/churn/Churn_Modelling.csv'
df = pd.read_csv(filepath)


# In[2]:


df.shape


# In[3]:


df.head()


# In[4]:


cols = df.columns
cols


# In[5]:


df.dtypes


# In[6]:


df.nunique(axis=0)


# In[7]:


df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))


# In[8]:


df.isnull().sum()


# In[9]:


# removing outliers

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_out


# In[10]:


#Removing null 
df_out = df_out.dropna(axis=0)
df_out


# In[11]:



amount_retained = df[df['Exited'] == 0]['Exited'].count() / df.shape[0] * 100
amount_lost = df[df['Exited'] == 1]['Exited'].count() / df.shape[0] * 100

fig, ax = plt.subplots()
sns.countplot(x='Exited', palette="Set3", data=df)
plt.xticks([0, 1], ['Retained', 'Lost'])
plt.xlabel('Condition', size=15, labelpad=12, color='grey')
plt.ylabel('Amount of customers', size=15, labelpad=12, color='grey')
plt.title("Proportion of customers lost and retained", size=15, pad=20)
plt.ylim(0, 9000)
plt.text(-0.15, 7000, f"{round(amount_retained, 2)}%", fontsize=12)
plt.text(0.85, 1000, f"{round(amount_lost, 2)}%", fontsize=12)
sns.despine()
plt.show()


# In[12]:


categorical_labels = [['Gender', 'Geography'], ['HasCrCard', 'IsActiveMember']]
colors = [['Set1', 'Set2'], ['Set3', 'PuRd']]

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for i in range(2):
    for j in range(2):
        feature = categorical_labels[i][j]
        color = colors[i][j]
        ax1 = sns.countplot(x=feature, hue='Exited', palette=color, data=df, ax=ax[i][j])
        ax1.set_xlabel(feature, labelpad=10)
        ax1.set_ylim(0, 6000)
        ax1.legend(title='Exited', labels= ['No', 'Yes'])
        if i == 1:
            ax1.set_xticklabels(['No', 'Yes'])
sns.despine()


# In[13]:


#correlation

corr = df.corr()# plot the heatmap
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 40, as_cmap=True))


# In[14]:


features = ['Age','EstimatedSalary','Balance', 'IsActiveMember','NumOfProducts','Gender']
X = df
y = df['Exited']

# Split dataset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# In[15]:


# All categorical columns, training and validation
cat_cols = [col for col in X.columns if X[col].dtype == "object"]

# All numeric columns
num_cols = list(set(X.columns)-set(cat_cols))
#num_cols.remove('Id')

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])


# In[16]:


model = DecisionTreeClassifier()


# In[17]:


KNN_model = KNeighborsClassifier()


# In[18]:


# PIPELINE 1 Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)
prediction = my_pipeline.predict(X_valid)
print(accuracy_score(y_valid,prediction))
print(f1_score(y_valid,prediction))


# In[19]:


# PIPELINE 2 Bundle preprocessing and modeling code in a pipeline
my_pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', KNN_model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)
prediction2 = my_pipeline.predict(X_valid)
print(accuracy_score(y_valid,prediction2))
print(f1_score(y_valid,prediction2))


# In[ ]:




