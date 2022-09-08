#!/usr/bin/env python
# coding: utf-8

# In[1]:


#McD India Menu Analysis


# In[2]:



import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")


# In[3]:


df=pd.read_csv("Downloads\India_Menu.csv (1).xls")


# In[4]:


df


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df["Sodium (mg)"].fillna(362,inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.columns


# In[11]:


df_snacks = df[df['Per Serve Size'].str.contains('g')]
df_snacks['Per Serve Size'] = df_snacks['Per Serve Size'].str.replace(' g', '')
df_snacks['Per Serve Size'] = df_snacks['Per Serve Size'].astype('float64')

df_drinks = df[df['Per Serve Size'].str.contains('ml')]
df_drinks['Per Serve Size'] = df_drinks['Per Serve Size'].str.replace(' ml', '')
df_drinks['Per Serve Size'] = df_drinks['Per Serve Size'].astype('float64')


# In[12]:


#snacks Item


# In[13]:


df_snacks.index = np.arange(1, len(df_snacks) + 1)
df_snacks


# In[14]:


#Drinks/Beverages/Condiments Menu


# In[15]:


df_drinks.index = np.arange(1, len(df_drinks) + 1)
df_drinks


# In[ ]:





# In[ ]:





# In[16]:


df.iloc[0:5,0:5]


# In[17]:


df.loc[0:5,"Menu Category":"Protein (g)"]


# In[18]:


#Comparing Snacks and Drinks Count


# In[19]:


fig = px.bar(x =['snacks', 'drinks'] ,y=[df_snacks.shape[0], df_drinks.shape[0]], color=['snacks', 'drinks'],color_discrete_map={'snacks':'red','drinks':'yellow'})
fig.update_xaxes(title_text=None)
fig.update_yaxes(title_text='Count')


# In[ ]:





# In[20]:


df_snacks['Menu Items'] = df_snacks['Menu Items'].str.lower()
df_snacks_nonveg = df_snacks[(df_snacks['Menu Items'].str.contains('chicken')) | (df_snacks['Menu Items'].str.contains('fish')) | (df_snacks['Menu Items'].str.contains(' egg')) | (df_snacks['Menu Items'].str.contains('sausage'))] 

df_snacks_veg = pd.concat([df_snacks, df_snacks_nonveg]).drop_duplicates(keep=False)


# In[21]:


#veg snacks menu


# In[22]:


df_snacks_veg.index = np.arange(1, len(df_snacks_veg) + 1)
df_snacks_veg


# In[23]:


#non veg menu


# In[24]:


df_snacks_nonveg.index = np.arange(1, len(df_snacks_nonveg) + 1)
df_snacks_nonveg


# In[25]:


#Comparing veg and non veg items count


# In[26]:


fig = px.bar(x =['veg', 'nonveg'] ,y=[df_snacks_veg.shape[0], df_snacks_nonveg.shape[0]], color=['veg', 'nonveg'],color_discrete_map={'veg':'green','nonveg':'red'})
fig.update_xaxes(title_text=None)
fig.update_yaxes(title_text='Count')


# In[27]:


fig = px.scatter_matrix(df_snacks.iloc[:,2:], dimensions=df_snacks.iloc[:,2:].columns.tolist(), height=1000, width=1000)
fig.update_layout(
    font_size=5,
)


# In[ ]:





# In[28]:


#-!pip install cufflinks
import numpy as np
from plotly.offline import init_notebook_mode, iplot
import cufflinks as cf 
cf.go_offline () 
cf.set_config_file (offline=False, world_readable=True)


# In[29]:


df.select_dtypes([np.int, np.float]).nunique().iplot(kind='bar',labels= 'index', values= 'values', title="Unique Count: Numeric Columns", color= 'blue')


# In[ ]:





# In[30]:


import plotly.express as px
px.histogram(df["Protein (g)"], title= 'Distribution of Protein')


# In[31]:


df.select_dtypes(object).nunique().sort_values(ascending= False).iplot(kind='bar', 
                                                     labels= 'index', values= 'values', title="Unique Count: Discrete Variables", color= 'orange')


# In[32]:


#


# In[33]:


pie4 = df["Cholesterols (mg)"].value_counts().head(10)
pie_df4 = pd.DataFrame({'index':pie4.index, 'values': pie4.values})
pie_df4.iplot(kind='pie', labels= 'index', values= 'values', hole= .5, title="Value counts: Cholesterols")


# In[34]:


top_10_prep = df.loc[df["Protein (g)"] >= 1, ['Menu Items', 'Protein (g)']].sort_values(by='Protein (g)', ascending=False).head(10)
px.bar(top_10_prep, y='Menu Items', x='Protein (g)', color='Protein (g)')


# In[35]:


px.scatter(df, x='Total carbohydrate (g)', y='Total Sugars (g)', color= 'Menu Items')


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(),annot=True,cmap='GnBu',linewidths=0.3)
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.show()


# In[36]:


d = df.groupby(['Per Serve Size', 'Menu Items']).mean().reset_index().head(10)
px.bar(d, x='Per Serve Size', y='Sodium (mg)', color='Menu Items', color_discrete_sequence= 
       px.colors.qualitative.Dark2)


# In[37]:


pd.options.plotting.backend = "plotly"
df.groupby('Menu Items').sum().plot.area(color_discrete_sequence=px.colors.qualitative.Safe)


# In[38]:


df.groupby(['Total fat (g)', 'Total carbohydrate (g)']).mean().iplot(kind='line')


# In[39]:


sunb = df.groupby(['Menu Items', 'Protein (g)']).mean().reset_index().head(50)
px.sunburst(sunb,path= ['Menu Items', 'Protein (g)'], values = 'Total fat (g)')


# In[ ]:


df


# In[41]:


import matplotlib.pyplot as plt


# In[42]:


fig = plt.figure(figsize = (50, 20))
plt.bar(df['Menu Items'], df['Protein (g)'], color ='red', 
        width = 1)


# In[43]:


df.pivot_table('Energy (kCal)', 'Menu Category').plot(kind='bar')


# In[44]:


import seaborn as sns


# In[45]:


measures = ['Energy (kCal)', 'Total fat (g)', 'Cholesterols (mg)','Sodium (mg)', 'Total Sugars (g)', 'Total carbohydrate (g)']

for m in measures:   
    plot = sns.swarmplot(x="Menu Category", y=m, data=df)
    plt.setp(plot.get_xticklabels(), rotation=45)
    plt.title(m)
    plt.show()


# In[46]:


px.parallel_categories(df.drop(["Energy (kCal)","Protein (g)","Total fat (g)"],axis=1).head(10), color_continuous_scale=px.colors.sequential.Agsunset)


# In[ ]:




