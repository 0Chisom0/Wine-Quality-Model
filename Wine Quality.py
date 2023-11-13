#!/usr/bin/env python
# coding: utf-8

# # Importing Dataset and Creating DataFrame

# In[65]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler as sc
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from scipy import stats
import plotly
import scipy
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score


# In[5]:


df = pd.read_csv('/Users/chisomokezie/Downloads/winequality-red.csv')
df.head()


# # Hypothesis for testing
# 
# Null Hypothesis (H0): The level of citric acidity and volatility acidity in wine does not significantly affect the quality score. In other words, there is no relationship between these acidity levels and wine quality.
# 
# Alternative Hypothesis (H1): The level of citric acidity and volatility acidity in wine significantly affects the quality score. Specifically, you believe that higher rates of citric acidity and lower rates of volatility acidity are associated with higher-quality wine.
# 
# Assuming the null hypothesis is true, let's find enough evidence to reject the null hypothesis.

# In[6]:


sns.heatmap(data = df.corr())


# \begin{equation}
# \begin{cases}
# H_0: \mu_{\text{citric}} = \mu_{\text{volatility}} \\
# H_a: \mu_{\text{citric}} \neq \mu_{\text{volatility}}
# \end{cases}
# \end{equation}
# 
# $Let\quad\alpha = 5\%$
# $n=1599\\
# $

# In[7]:


df.columns


# In[63]:


df.info()


# In[9]:


df.dtypes


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# # EDA

# In[12]:


#set significance level as 0.05
alpha = 0.05


# In[13]:


#sample mean and standard deviation
ci_mean = df['citric acid'].mean
vo_mean = df['volatile acidity'].mean
q_mean = df['quality'].mean

ci_std = df['citric acid'].std(ddof=1)
vo_std = df['volatile acidity'].std(ddof=1)
q_std = df['quality'].std(ddof=1)

##perform two-sample t-test
from scipy.stats import ttest_ind
#convert the df to nd.array
ca = df['citric acid'].values
va = df['volatile acidity'].values

t_statistic, p_value = ttest_ind(ca,va)
print(p_value)

if p_value < alpha:
    print("reject the null, significant difference")
else:
    print("Fail to reject the null, no significant difference")


# In[14]:


# Perform ANOVA tests
f_stat_citric, p_value_citric = stats.f_oneway(ca, df['quality'])
f_stat_volatile, p_value_volatile = stats.f_oneway(va, df['quality'])

print("ANOVA for 'citric acidity':")
print(f"F-statistic: {f_stat_citric}")
print(f"P-value: {p_value_citric}")

print("\nANOVA for 'volatile acidity':")
print(f"F-statistic: {f_stat_volatile}")
print(f"P-value: {p_value_volatile}")


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt


# Create a box plot for 'volatile acidity,' 'citric acid,' and 'quality'
plt.figure(figsize=(12, 6))  # Set the figure size

# Create a subplot for 'volatile acidity'
plt.subplot(1, 3, 1)
sns.boxplot(data=df, y='volatile acidity')
plt.title('Volatile Acidity')

# Create a subplot for 'citric acid'
plt.subplot(1, 3, 2)
sns.boxplot(data=df, y='citric acid')
plt.title('Citric Acid')

# Adjust spacing between subplots
plt.tight_layout()

# Show the box plots
plt.show()


# # Confirming the distribution 

# In[16]:


#Using shapiro p-valuefor volatile acidity
from scipy.stats import shapiro

test = df['volatile acidity']

stat, p = shapiro(test)

print('stat=%.2f, p=%.30f' % (stat, p))

if p > 0.05:
    print("normal distribution")
    
else:
    print('Not a normal distribution')


# In[17]:


#Using shapiro p-value for citric acidity
from scipy.stats import shapiro

test = df['citric acid']

stat, p = shapiro(test)

print('stat=%.2f, p=%.30f' % (stat, p))

if p > 0.05:
    print("normal distribution")
    
else:
    print('Not a normal distribution')


# In[18]:


#using D’Agostino’s K^2 Test for volatile acidity
from scipy.stats import  normaltest
test = df['volatile acidity']
stat, p = normaltest(test)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
 print('Probably Gaussian')
else:
 print('Probably not Gaussian')


# In[19]:


#using  D’Agostino’s K^2 Test for citric acid
from scipy.stats import  normaltest
test = df['citric acid']
stat, p = normaltest(test)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
 print('Probably Gaussian')
else:
 print('Probably not Gaussian')


# In[20]:


#Creating a distribution plot for each column in the dataframe
plt.figure(figsize=(30,45))
for i, col in enumerate(df.columns):
    if df[col].dtype != 'object':
        ax = plt.subplot(9, 2, i+1)
        sns.kdeplot(df[col], ax = ax)
        plt.xlabel(col)
        
plt.show()


# In[21]:


#using scaler
sc = sc()
scaled_df = sc.fit_transform(df)


# In[89]:


scaled_df


# In[23]:


#elbow method 
inertia = []
range_val = range(1,12)
for i in range_val:
    kmean = KMeans(n_clusters=i)
    kmean.fit_predict(pd.DataFrame(scaled_df))
    inertia.append(kmean.inertia_)
plt.plot(range_val,inertia,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()


# In[24]:


#Silhouette score
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for n_clusters in range_n_clusters:
    #Initializing the clusterer with n_clusters value and a random   generator
    kmean = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmean.fit_predict(df)
    #The silhouette_score gives the average value for all the   samples.
    #Calculating number of clusters
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("For n_clusters =", n_clusters,"The average   silhoutte_score is :", silhouette_avg)


# In[25]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
pca_df


# In[26]:


#Choosing a Kmeans value of 2
kmeans_model=KMeans(2)
kmeans_model.fit_predict(scaled_df)
pca_df_kmeans = pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)


# In[27]:


#Creating 2 clusters
plt.figure(figsize=(8,8))
ax=sns.scatterplot(x="PCA1", y="PCA2", hue="cluster",data=pca_df_kmeans,palette=['green','black'])
plt.title("Clustering using K-Means Algorithm")
plt.show()


# # Correlation(Feature Analysis)

# In[91]:


#creating correlation we choose spearman because we don't have a gausian distribution
corrdf_matrix = df.corr(method = 'spearman')

plt.figure(figsize=(10, 8))

# Create the heatmap using the `heatmap` function of Seaborn to see if theres a correlation between quality and acididty
sns.heatmap(corrdf_matrix, cmap='coolwarm', annot=True)

plt.show()


# # Spearman Rank Correlation Test

# In[30]:


#Spearman Rank Correlation Test
from scipy import stats
correlation_coefficient, p_value = stats.spearmanr(df['volatile acidity'], df['quality'])
print('p_value=%.3f' % (p_value))
if p_value > 0.05:
 print('Probably independent')
else:
 print('Probably dependent')

print(p_value)


# In[31]:


#Spearman Rank Correlation Test, the p-value is 6.15 so it is true
from scipy import stats
correlation_coefficient, p_value = stats.spearmanr(df['citric acid'], df['quality'])
print('p_value=%.3f' % (p_value))
if p_value > 0.05:
 print('Probably independent')
else:
 print('Probably dependent')

print(p_value)


# # ANOVA

# In[32]:


df['volatile_acidity_groups'] = pd.cut(df['volatile acidity'], bins=3, labels=['Group 1', 'Group 2', 'Group 3'])


# In[33]:


df['citric_acidity_groups'] = pd.cut(df['citric acid'], bins=3, labels=['Group 1', 'Group 2', 'Group 3'])


# In[34]:


df


# In[35]:


from scipy import stats
from sklearn.preprocessing import LabelEncoder

#categorized data into groups
grouped_citric_acidity = df['citric_acidity_groups']
grouped_volatile_acidity = df['volatile_acidity_groups']

# Create label encoders
label_encoder = LabelEncoder()

# Encode the categorical group labels
grouped_citric_acidity_encoded = label_encoder.fit_transform(grouped_citric_acidity)
grouped_volatile_acidity_encoded = label_encoder.fit_transform(grouped_volatile_acidity)

# Perform ANOVA tests
f_stat_citric, p_value_citric = stats.f_oneway(grouped_citric_acidity_encoded, df['quality'])
f_stat_volatile, p_value_volatile = stats.f_oneway(grouped_volatile_acidity_encoded, df['quality'])

print("ANOVA for 'citric acidity':")
print(f"F-statistic: {f_stat_citric}")
print(f"P-value: {p_value_citric}")

print("\nANOVA for 'volatile acidity':")
print(f"F-statistic: {f_stat_volatile}")
print(f"P-value: {p_value_volatile}")


# In[36]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df['quality'], groups=df['citric_acidity_groups'], alpha=0.05)

# Plot the results
tukey.plot_simultaneous()

plt.title("Group to quality")
plt.xlabel("Quality")
plt.ylabel("citric_acidity_groups")

plt.show()
#shows that alochol with the higher acidity on average has a better quality 


# In[37]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df['quality'], groups=df['volatile_acidity_groups'], alpha=0.05)

# Plot the results
tukey.plot_simultaneous()

plt.title("Group to quality")
plt.xlabel("Quality")
plt.ylabel("volatile_acidity_groups")

plt.show()
#Here it shows that wine with lower VA has higher quality


# The p-values you've obtained are extremely small, indicating highly significant Spearman rank correlations between the variables. Specifically:
# 
# For the relationship between "citric acidity" and "quality," the p-value is approximately 2.73e-56. This is an extremely small p-value, indicating a highly significant, strong, and negative Spearman rank correlation between "citric acidity" and "quality."
# For the relationship between "volatile acidity" and "quality," the p-value is approximately 6.16e-18. This is also an extremely small p-value, indicating a highly significant, albeit less strong, Spearman rank correlation between "volatile acidity" and "quality."
# Given these results, it is reasonable to conclude that both "citric acidity" and "volatile acidity" have statistically significant relationships with the "quality" of the wine. The negative correlation for "citric acidity" suggests that as "citric acidity" increases, wine quality tends to decrease, while the correlation for "volatile acidity" indicates a similar but weaker effect.

# # Further Testing - Just contine with the above plot to show where values are and then create a barchart or countplot to view the data relative to the groups

# # Cluster Analysis so we can view what features are consistent in a high quality wine

# In[72]:


#Cluster analysis. based on the correlation to visualize the relationship between quality and alcohol content
df.plot(x='quality',y='alcohol', kind='scatter', title= 'Alcohol % vs Quality')


# In[87]:


#/* visualize data in multiple dimensions */
import plotly.graph_objects as go
import plotly.offline as pyo


columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']


fig = go.Figure(
    data=[
        go.Scatterpolar(r=df['fixed acidity'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['volatile acidity'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['citric acid'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['residual sugar'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['chlorides'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['free sulfur dioxide'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['total sulfur dioxide'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['density'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['pH'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['sulphates'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['alcohol'], theta=columns, fill='toself'),
        go.Scatterpolar(r=df['quality'], theta=columns, fill='toself')
    ],
    layout=go.Layout(
        title=go.layout.Title(text='Feature Analysis'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
)

#pyo.plot(fig)
fig.show()

#The output shows, provides a visualiziation of what the output of a decision tree might be, visualize the weight of each feature


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Save the model
