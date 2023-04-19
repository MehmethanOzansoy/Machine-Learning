#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[6]:


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64','int64'])
df = df.dropna()
df.head(10)


# In[5]:


df.size


# In[7]:


df_table = df["table"]
df_table.head()


# In[8]:


sns.boxplot(x=df_table);


# In[10]:


Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3 - Q1


# In[11]:


Q1


# In[12]:


Q3


# In[13]:


IQR


# In[14]:


alt_sınır = Q1 - 1.5*IQR
ust_sınır = Q3 + 1.5*IQR


# In[15]:


alt_sınır


# In[16]:


ust_sınır


# In[20]:


((df_table < alt_sınır) | (df_table > ust_sınır))


# In[30]:


aykırı_df = (df_table < alt_sınır) 


# In[31]:


aykırı_df.head()


# In[32]:


df_table[aykırı_df]


# In[33]:


df_table[aykırı_df].index


# ### Silme

# In[35]:


import pandas as pd


# In[36]:


type(df_table)


# In[37]:


df_table = pd.DataFrame(df_table)


# In[38]:


df_table.shape


# In[39]:


t_df = df_table[~((df_table<alt_sınır) | (df_table >ust_sınır)).any(axis = 1)]


# In[40]:


##605 tane aykırı var silme işlemi yaptık
t_df


# ### Ortalama ile Doldurma

# In[41]:


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64','int64'])
df = df.dropna()
df.head(10)


# In[42]:


df_table = df["table"]
df_table.head()


# In[43]:


aykırı_df.head()


# In[44]:


df_table[aykırı_df]


# In[45]:


df_table.mean()


# In[47]:


import warnings
warnings.filterwarnings('ignore')
df_table[aykırı_df] = df_table.mean()


# In[48]:


df_table[aykırı_df]


# ### Baskılama yöntemi

# In[49]:


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64','int64'])
df = df.dropna()
df.head(10)


# In[51]:


df_table = df["table"]


# In[52]:


df_table[aykırı_df]


# In[53]:


df_table[aykırı_df] = alt_sınır


# In[54]:


df_table[aykırı_df]


# ## Cok değişkenli Outlier

# In[56]:


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64','int64'])
df = df.dropna()
df.head(10)


# In[57]:


import numpy as np
from sklearn.neighbors import LocalOutlierFactor


# In[58]:


clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)


# In[59]:


clf.fit_predict(df)


# In[60]:


df_scores = clf.negative_outlier_factor_


# In[63]:


df_scores[0:10]


# In[67]:


np.sort(df_scores)[0:20]


# In[68]:


threshold = np.sort(df_scores)[13] ## Biz seçtik bu değeri


# In[69]:


aykırı_df = df_scores > threshold


# In[70]:


aykırı_df


# In[71]:


yeni_df = df[df_scores > threshold]


# In[72]:


yeni_df


# In[73]:


df[df_scores < threshold]


# ## Baskılama

# In[74]:


df[df_scores == threshold]


# In[75]:


baskı_degeri = df[df_scores == threshold]


# In[78]:


baskı_degeri


# In[76]:


aykırılar = df[~aykırı_df]


# In[77]:


aykırılar


# In[79]:


res = aykırılar.to_records(index=False)


# In[81]:


res ## array e cevirdi indeksleri almadan


# In[82]:


res[:] = baskı_degeri.to_records(index=False)


# In[84]:


res ## aykırların yerine baskılanmış değeri atadı thresholdu yani


# In[85]:


df[~aykırı_df]


# In[86]:


df[~aykırı_df] = pd.DataFrame(res,index=df[~aykırı_df].index)


# In[87]:


df[~aykırı_df]


# In[ ]:




