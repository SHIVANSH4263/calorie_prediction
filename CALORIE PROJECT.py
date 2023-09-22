#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
calories = pd.read_csv('D:\CALORIE PROJECT\calorie_dataset\calories.csv')
calories.head()


# In[2]:


exercise_data = pd.read_csv('D:\CALORIE PROJECT\calorie_dataset\exercise.csv')
exercise_data.head()


# In[3]:


calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
calories_data.head()


# In[4]:


calories_data.shape


# In[5]:


calories_data.info


# In[6]:


calories_data.isnull().sum()


# In[7]:


calories_data.describe()


# In[8]:


sns.set()
sns.countplot(calories_data['Gender'])


# In[9]:


sns.distplot(calories_data['Age'])


# In[10]:


sns.distplot(calories_data['Height'])


# In[11]:


sns.distplot(calories_data['Weight'])


# In[12]:


sns.pairplot(calories_data)


# In[13]:


correlation = calories_data.corr()


# In[14]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[15]:


calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[16]:


calories_data.head()


# In[17]:


X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']
print(X)


# In[18]:


print(Y)


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[20]:


model = XGBRegressor()
model.fit(X_train, Y_train)


# In[21]:


test_data_prediction = model.predict(X_test)
print(test_data_prediction)


# In[22]:


mae = metrics.mean_absolute_error(Y_test, test_data_prediction)


# In[23]:


print("Mean Absolute Error = ", mae)


# In[ ]:




