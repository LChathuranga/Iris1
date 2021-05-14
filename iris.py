#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset=pd.read_csv('iris.csv').values
print(dataset)


# In[2]:


data=dataset[:,:4]
target=dataset[:,4]


# In[3]:


from sklearn.model_selection import train_test_split
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2)


# In[4]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)


# In[5]:


model.fit(train_data,train_target)


# In[6]:


predicate_target=model.predict(test_data)


# In[7]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(predicate_target,test_target)
print(acc)


# In[8]:


test_data=[[5.2,1.2,4.2,0.5]]
result=model.predict(test_data)
print(result)


# In[9]:


test_data=[[115.2,1.2,4.2,0.5]]
result=model.predict(test_data)
print(result)


# In[10]:


import joblib
joblib.dump(model,'kn.sav')


# In[ ]:




