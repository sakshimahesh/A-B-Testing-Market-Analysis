#!/usr/bin/env python
# coding: utf-8

# In[32]:


#first we find probabilities of the users who are 
#1. willing to convert into new designed page
#2. prob of users who actually converted to new designed page
import pandas as pd
import numpy as np
import random
random.seed(42)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data=pd.read_csv(r'C:\Users\saksh\OneDrive\Desktop\ab_data.csv')#reading .csv file
data.head() #displaying first five rows in the data
#data.shape


# In[33]:


data.shape #displays total rows and columns in the data


# In[34]:


data.nunique() #displays number/type of each column


# In[35]:


data['converted'].value_counts()[1]/(data['converted'].value_counts()[0] 
                                     + data['converted'].value_counts()[1]) 


# In[36]:


data.query("group == 'treatment' and landing_page != 'new_page'").count()[0] + data.query("group != 'treatment' and landing_page == 'new_page'").count()[0] 
#looking here for the group treatment not ending up in new_page
#.query() is knowing number of things in a dataset whcih depends on the 
#operations we given in query command


# In[37]:


data.info()


# In[38]:


data.drop(data.query("group == 'treatment' and landing_page != 'new_page'").index, inplace = True)
data.drop(data.query("group != 'treatment' and landing_page == 'new_page'").index, inplace = True)
a = data
a.shape


# In[39]:


a[((a['group'] == 'treatment') == (a['landing_page'] == 'new_page')) == False].shape[0]


# In[40]:


a.nunique()


# In[41]:


a[a['user_id'].duplicated()] #finding a duplicate row
#a.head()


# In[42]:


dup = a[a['user_id'].duplicated()] #duplicate row assigned to dup
a.drop(dup.index, inplace = True) #to delete/drop the row
a.nunique()


# In[43]:


a[a['converted'] == 1].count()[0] / a['converted'].count() #probability overall
#only 11.9% convert to new page


# In[44]:


pc = a.query("group == 'control' and converted == 1").count()[4]/a[a['group'] == 'control'].count()[4]
#probability of converted in control group
pc


# In[45]:


#probability of converted rows in treatment 
pt= a.query("group == 'treatment' and converted ==1").count()/a[a['group']=='treatment'].count()
pt
#.count()[4] helps display just one probability in one line like in datatype pc


# In[46]:


a.query("landing_page == 'new_page'").count()[0] / a['landing_page'].count()

#Results
#atleast 50% users are willing to land in new_page
#the python statemnt shows the probabilty of users landing on new page


# In[47]:


#A/B testing
#pold-pnew=>0 
#pold-pnew<0
# first we find pnew under 0/null
pnew=a.query("converted==1").count()[4]/a["converted"].count()
pnew


# In[48]:


pold=a.query("converted==1").count()[4]/a["converted"].count()
pold
#difference=pnew-pold
#difference


# In[49]:


nnew=a.query('group=="treatment"').count()[0]
nnew


# In[50]:


nold=a.query('group=="control"').count()[0]
nold


# In[51]:


difference=nnew-nold
difference


# In[52]:


pdiff=pnew-pold
pdiff


# In[53]:


#new page converted
o=np.random.choice([0,1], nnew, p=(pnew, 1-pnew))
o


# In[54]:



q=np.random.choice([0,1], nold, p=(pold, 1-pold))
q#old page converted


# In[55]:


o.mean() - q.mean()


# In[57]:


probdiff = [] #probability of differfence, a new assigned
size = a.shape[0]
for i in range(10000):
    samp = a.sample(size, replace = True )
    o=np.random.choice([0,1], nnew, p=(pnew, 1-pnew))
    q=np.random.choice([0,1], nold, p=(pold, 1-pold))
    probdiff.append(o.mean() - q.mean()) 


# In[ ]:


probdiff = np.array(probdiff)
plt.hist(probdiff, density=True, bins=30);
plt.ylabel('users')
plt.xlabel('prob');
#plt.ylabel("users")
#plt.gca().set(xlabel = "prob")
#plt.gca().set( ylabel="users")


# In[ ]:


obsdiff=pc-pt
#control-treatment#actual treatment and control groups
obsdiff


# In[ ]:


#distribution among null hypo
#h0=consider no changed
nullvalue=np.random.normal(0, probdiff.std(), probdiff.size)
nullvalue


# In[ ]:


plt.hist(nullvalue);
plt.axvline(obsdiff.mean(), color='r')
#plt.axvline(obsdiff, color ='r');


# In[ ]:


#pvalue
#(nullvalue>obsdiff[0]).mean()
v=nullvalue > obsdiff[0]
v.mean()
#pavlue is 9.3% >type 1 erro 5%
#i.e H0 cannot be rejected


# In[ ]:


#with statitics package
import statsmodels.api as sm
from scipy.stats import norm
opc=o.mean() #old and new page converted , there mean
oldpc=q.mean()
nold = nold
nnew = nnew


# In[ ]:


zscore, pvalue = sm.stats.proportions_ztest(np.array([opc, oldpc]),np.array([nnew, nold]), alternative='larger')
zscore, pvalue
#pvalue is around 50% which isnt significant


# In[ ]:


norm.cdf(zscore)


# In[ ]:


norm.pdf(1-(0.05/2))
#zscore<norm.pdf hecne
#H0 cannot be rejected
#ie we keep the old version of the page


# In[ ]:


#logistic regession
a['intercept']=1
a=a.join(pd.get_dummies(a['landing_page']))
a['ab_page']=pd.get_dummies(a['group'])['treatment']
a.head() 


# In[ ]:


log = sm.Logit(a['converted'], a[['intercept', 'ab_page']])
r=log.fit()
r.summary() 


# In[ ]:


#pvalue is 0.19
#ab_page not significant
#H0=no difference bewteen control adn treatment
import pandas as pd
country=pd.read_csv(r'C:\Users\saksh\OneDrive\Desktop\countries.csv')
newdata=country.set_index('user_id').join(a.set_index('user_id'), how ='inner')
newdata.head()

