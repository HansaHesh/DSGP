#!/usr/bin/env python
# coding: utf-8

# ### Importing the required Libraries
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ### Importing the dataset
# 
# 

# In[2]:


df = pd.read_csv('vehicle_dataset.csv')


# In[3]:


df.head()


# ### Data Cleaning & Preprocessing

# In[4]:


df.dtypes


# #### Converting Columns into appropiate data types
# 

# In[5]:


def extract_num(row):
    amount = ""
    for w in row:
        if w.isnumeric():
            amount += w
    return int(amount)

df["Price"] = df["Price"].apply(extract_num)
df["Mileage"] = df["Mileage"].apply(extract_num)
df["Capacity"] = df["Capacity"].apply(extract_num)

df = df.rename({"Price":"Price (Rs)", "Mileage":"Mileage (km)", "Capacity": "Capacity (cc)"}, axis=1)


# Checking the Null values

# In[6]:


df.isnull().sum()


# In[7]:


plt.figure(figsize = (10,6))
plt.title("Missing values in Each Column\n", size = 15)
sns.heatmap(df.isnull(), yticklabels=False, cbar=False);


# Imputing missing Values 
# 
# 
# 

# In[8]:


df['Edition'].fillna(df['Edition'].mode()[0], inplace = True)
df['Body'].fillna(df['Body'].mode()[0], inplace = True)


# In[9]:


df.isnull().sum()


# In[10]:


df.head()


# In[11]:


df.dtypes


# In[12]:


df.describe()


# In[13]:


df['Title'] = df['Brand'] + " "+ df['Model']


# In[14]:


df['published_date'] = pd.to_datetime(df['published_date'])


# Seperating Year, Month & day for 3 columns

# In[15]:


# for year
df['Published_year'] = pd.DatetimeIndex(df['published_date']).year

# for month
df['Published_month'] = pd.DatetimeIndex(df['published_date']).month

# for day
df['Published_day'] = pd.DatetimeIndex(df['published_date']).day


# In[16]:


df.columns


# #### Dropping Unnecessary Features

# In[17]:


df.drop(["Sub_title",'Description', 'published_date'], axis = 1, inplace = True)


# In[18]:


df.head()


# In[19]:


df['Current_year'] = 2022


# In[20]:


df['Years_Old'] = df['Current_year'] - df['Published_year']


# In[21]:


df.head()


# In[22]:


df=df.drop(columns=['Title', 'Brand', 'Model', 'Edition', 'Condition', 'Body', 'Location', 'Post_URL', 'Years_Old', 'Seller_name', 'Seller_type', 'Published_year', 'Published_month', 'Published_day', 'Current_year'])


# In[23]:


df.head()


# ### Encoding the categorical variables

# In[24]:


df.select_dtypes(include='object').columns


# In[25]:


# one hot encoding
df = pd.get_dummies(data=df, drop_first=True)


# In[26]:


df.head()


# In[27]:


df.shape


# In[28]:


# matrix of features
x = df.drop(columns='Price (Rs)')


# In[29]:


# target variable
y = df['Price (Rs)']


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[31]:


x_train.shape


# In[32]:


x_test.shape


# ### Building The Model

# In[34]:


from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor()
regressor.fit(x_train, y_train)


# In[35]:


y_pred = regressor.predict(x_test)


# In[36]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# ### Hyperparameter Tuning

# In[37]:


from sklearn.model_selection import RandomizedSearchCV


# In[38]:


parameters = {
    'n_estimators':[100,200,300,400,500,600,700,800,900,1000],
    'criterion':['mse','mae'],
    'max_depth':[10,20,30,40,50],
    'min_samples_split':[2,5,10,20,50],
    'min_samples_leaf':[1,2,5,10],
    'max_features':['auto', 'sqrt', 'log2']
}


# In[39]:


random_cv = RandomizedSearchCV(estimator=regressor, param_distributions=parameters, n_iter=10, 
                               scoring='neg_mean_absolute_error', cv=5, verbose=2, n_jobs=-1)


# In[40]:


random_cv.fit(x_train, y_train)


# In[41]:


random_cv.best_estimator_


# In[42]:


random_cv.best_params_


# ### Predicting the price

# In[4]:


df.head()


# In[1]:


single_pred = [[2015,500,20000,0,1,0,0,0,0,0,1]]


# In[3]:


regressor.predict(single_pred)


# In[ ]:





# In[ ]:




