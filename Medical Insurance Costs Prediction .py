#!/usr/bin/env python
# coding: utf-8

# Below is the code for a predictive modelling project. This model predicts medical insurance costs based on someone's age, location, number of children, smoking status, obesity levels etc., 
# 
# Supervised machine learning project 
# 
# Data set from Kaggle.com 

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score, KFold
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
insurance = pd.read_csv("insurance.csv")
insurance.head()


# In[2]:


insurance=pd.read_csv("insurance.csv")
insurance.info()


# In[3]:


# Replacing string values to numbers
insurance['sex'] = insurance['sex'].apply({'male':0,      'female':1}.get) 
insurance['smoker'] = insurance['smoker'].apply({'yes':1, 'no':0}.get)
insurance['region'] = insurance['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[4]:


insurance.head()


# In[5]:


import seaborn as sns
# Correlation betweeen 'charges' and 'age' 
sns.jointplot(x=insurance['age'],y=insurance['charges'])


# In[6]:


# Code to distinguish obese from non-obese - 0 will be nonsmoker and 1 is smoker
def map_obese(column):
    mapped=[]
    for row in column:
        if row>30:
            mapped.append(1)
        else:
            mapped.append(0)
    return mapped
insurance["obese"]=map_obese(insurance["bmi"])


# In[7]:


# Code to distinguish smoker vs. non-smoker - 0 will be not obese and 1 obese
def map_smoking(column):
    mapped=[]
    for row in column:
        if row=="yes":
            mapped.append(1)
        else:
            mapped.append(0)
    return mapped
insurance["smoker_norm"]=map_smoking(insurance["smoker"])
nonnum_cols=[col for col in insurance.select_dtypes(include=["object"])]


# In[8]:


insurance.head(5)


# In[9]:


# features
X = insurance[['age', 'sex', 'bmi', 'children','smoker','region']]
# predicted variable
y = insurance['charges']


# In[10]:


# Train and test prediction model 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[11]:


len(X_test) # 402
len(X_train) # 936
len(insurance) # 1338


# In[12]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[13]:


predictions = model.predict(X_test)


# In[14]:


import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[15]:


# Prediction of charges for new customer - Named "Random"
data = {'age' : 45,
        'sex' : 0,
        'bmi' : 45.50,
        'children' : 2,
        'smoker' : 0,
        'region' : 3}
index = [1]
random_df = pd.DataFrame(data,index)
random_df


# In[16]:


prediction_random = model.predict(random_df)
print("Medical Insurance cost for Random is : ",prediction_random)


# In[ ]:




