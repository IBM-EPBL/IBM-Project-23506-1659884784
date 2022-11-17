#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# # Importing Dataset

# In[2]:


dataset=pd.read_csv('car performance.csv')
dataset


# # Finding missing data

# In[3]:


dataset.isnull().any()


# There are no null characters in the columns but there is a special character '?' in the 'horsepower' column. So we we replaced '?' with nan and replaced nan values with mean of the column.

# In[4]:


dataset['horsepower']=dataset['horsepower'].replace('?',np.nan)


# In[5]:


dataset['horsepower'].isnull().sum()


# In[6]:


dataset['horsepower']=dataset['horsepower'].astype('float64')


# In[7]:


dataset['horsepower'].fillna((dataset['horsepower'].mean()),inplace=True)


# In[8]:


dataset.isnull().any()


# In[9]:


dataset.info() #Pandas dataframe.info() function is used to get a quick overview of the dataset.


# In[10]:


dataset.describe() #Pandas describe() is used to view some basic statistical details of a data frame or a series of numeric values.


# There is no use with car name attribute so drop it

# In[11]:


dataset=dataset.drop('car name',axis=1) #dropping the unwanted column.


# In[12]:


corr_table=dataset.corr()#Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. 
corr_table


# # Data Visualizations

# Heatmap : which represents correlation between attributes

# In[13]:


sns.heatmap(dataset.corr(),annot=True,linecolor ='black', linewidths = 1)#Heatmap is a way to show some sort of matrix plot,annot is used for correlation.
fig=plt.gcf()
fig.set_size_inches(8,8)


# Visualizations of each attributes w.r.t rest of all attributes

# In[14]:


sns.pairplot(dataset,diag_kind='kde') #pairplot represents pairwise relation across the entire dataframe.
plt.show()


# Regression plots(regplot()) creates a regression line between 2 parameters and helps to visualize their linear relationships.

# In[15]:


sns.regplot(x="cylinders", y="mpg", data=dataset)


# In[16]:


sns.regplot(x="displacement", y="mpg", data=dataset)


# In[17]:


sns.regplot(x="horsepower", y="mpg", data=dataset)


# In[18]:


sns.regplot(x="weight", y="mpg", data=dataset)


# In[19]:


sns.regplot(x="acceleration", y="mpg", data=dataset)


# In[20]:


sns.regplot(x="model year", y="mpg", data=dataset)


# In[21]:


sns.regplot(x="origin", y="mpg", data=dataset)


# In[22]:


sns.set(style="whitegrid")
sns.boxplot(x=dataset["mpg"])


# Finding quartiles for mgp

# # The P-value is the probability value that the correlation between these two variables is statistically significant. 
# Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between 
# the variables is significant.
# 
# By convention, when the
# <ul>
#     <li>p-value is $<$ 0.001: we say there is strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.05: there is moderate evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.1: there is weak evidence that the correlation is significant.</li>
#     <li>the p-value is $>$ 0.1: there is no evidence that the correlation is significant.</li>
# </ul>

# In[23]:


from scipy import stats


# <h3>Cylinders vs mpg</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'Cylinders' and 'mpg'.

# In[24]:


pearson_coef, p_value = stats.pearsonr(dataset['cylinders'], dataset['mpg'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h5>Conclusion:</h5>
# 
# <p>Since the p-value is $<$ 0.001, the correlation between cylinders and mpg is statistically significant, and the coefficient of ~ -0.775 shows that the relationship is negative and moderately strong.

# <h3>Displacement vs mpg</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'Displacement' and 'mpg'.

# In[25]:


pearson_coef, p_value = stats.pearsonr(dataset['displacement'], dataset['mpg'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h5>Conclusion:</h5>
# 
# <p>Since the p-value is $<$ 0.1, the correlation between displacement and mpg is statistically significant, and the linear negative relationship is quite strong (~-0.809, close to -1)</p>

# <h3>Horsepower vs mpg</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'mpg'.

# In[26]:


pearson_coef, p_value = stats.pearsonr(dataset['horsepower'], dataset['mpg'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h5>Conclusion:</h5>
# 
# <p>Since the p-value is $<$ 0.001, the correlation between horsepower and mpg is statistically significant, and the coefficient of ~ -0.771 shows that the relationship is negative and moderately strong.

# <h3>Weght vs mpg</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'weight' and 'mpg'.

# In[27]:


pearson_coef, p_value = stats.pearsonr(dataset['weight'], dataset['mpg'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h5>Conclusion:</h5>
# 
# <p>Since the p-value is $<$ 0.001, the correlation between weight and mpg is statistically significant, and the linear negative relationship is quite strong (~-0.831, close to -1)</p>

# <h3>Acceleration vs mpg</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'Acceleration' and 'mpg'.

# In[28]:


pearson_coef, p_value = stats.pearsonr(dataset['acceleration'], dataset['mpg'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h5>Conclusion:</h5>
# <p>Since the p-value is $>$ 0.1, the correlation between acceleration and mpg is statistically significant, but the linear relationship is weak (~0.420).</p>

# <h3>Model year vs mpg</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'Model year' and 'mpg'.

# In[29]:


pearson_coef, p_value = stats.pearsonr(dataset['model year'], dataset['mpg'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h5>Conclusion:</h5>
# <p>Since the p-value is $<$ 0.001, the correlation between model year and mpg is statistically significant, but the linear relationship is only moderate (~0.579).</p>

# <h3>Origin vs mpg</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'Origin' and 'mpg'.

# In[30]:


pearson_coef, p_value = stats.pearsonr(dataset['origin'], dataset['mpg'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h5>Conclusion:</h5>
# <p>Since the p-value is $<$ 0.001, the correlation between origin and mpg is statistically significant, but the linear relationship is only moderate (~0.563).</p>

# <b>Ordinary Least Squares</b>  Statistics

# In[31]:


test=smf.ols('mpg~cylinders+displacement+horsepower+weight+acceleration+origin',dataset).fit()
test.summary()


# Inference as in the above summary the p value of the accelaration is maximum(i.e 0.972) so we can remove the acc variable from the dataset
# 

# # Seperating into Dependent and Independent variables

# <b>Independent variables</b>

# In[32]:


x=dataset[['cylinders','displacement','horsepower','weight','model year','origin']].values
x


# <b>Dependent variables</b>

# In[33]:


y=dataset.iloc[:,0:1].values
y


# # Splitting into train and test data.

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


# we are splitting as 90% train data and 10% test data

# # decision tree regressor

# In[36]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0,criterion="mae")
dt.fit(x_train,y_train)


# In[ ]:


import pickle
pickle.dump(dt,open('decision_model.pkl','wb'))


# In[37]:


y_pred=dt.predict(x_test)
y_pred


# In[38]:


y_test


# In[40]:


import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[39]:


ax1 = sns.distplot(dataset['mpg'], hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for mpg')
plt.xlabel('mpg')
plt.ylabel('Proportion of Cars')
 
plt.show()
plt.close()


# We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.

# <b>R-squared</b>
# <p>R-squared is a statistical measure of how close the data are to the fitted regression line. 
# It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.</p>
# 
# 
# R-squared = Explained variation / Total variation
# 
# <b>Mean Squared Error (MSE)</b>
# 
# <p>The Mean Squared Error measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).</p>

# In[41]:


from sklearn.metrics import r2_score,mean_squared_error


# In[42]:


r2_score(y_test,y_pred)


# In[43]:


mean_squared_error(y_test,y_pred)


# In[44]:


np.sqrt(mean_squared_error(y_test,y_pred))


# # random forest regressor

# In[45]:


from sklearn.ensemble import RandomForestRegressor


# In[46]:


rf= RandomForestRegressor(n_estimators=10,random_state=0,criterion='mae')
rf.fit(x_train,y_train)


# In[47]:


y_pred2=rf.predict(x_test)
y_pred2


# In[48]:


ax1 = sns.distplot(dataset['mpg'], hist=False, color="r", label="Actual Value")
sns.distplot(y_pred2, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for mpg')
plt.xlabel('mpg')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.

# In[49]:


from sklearn.metrics import r2_score,mean_squared_error


# In[50]:


r2_score(y_test,y_pred2)


# In[51]:


mean_squared_error(y_test,y_pred2)


# In[52]:


np.sqrt(mean_squared_error(y_test,y_pred2))


# # linear regression

# In[53]:


from sklearn.linear_model import LinearRegression
mr=LinearRegression()
mr.fit(x_train,y_train)


# In[54]:


y_pred3=mr.predict(x_test)
y_pred3


# In[55]:


ax1 = sns.distplot(dataset['mpg'], hist=False, color="r", label="Actual Value")
sns.distplot(y_pred3, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for mpg')
plt.xlabel('mpg')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# We can see that the fitted values are not as close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.

# In[56]:


from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_test,y_pred3)


# In[57]:


mean_squared_error(y_test,y_pred3)


# In[58]:


np.sqrt(mean_squared_error(y_test,y_pred3))


# <b>Conclusion:</b>
# <p>When comparing models, the model with the higher R-squared value is a better fit for the data.</p>
# <p>When comparing models, the model with the smallest MSE value is a better fit for the data.</p>
# 
# Comparing these three models, we conclude that the DecisionTree model is the best model to be able to predict mpg from our dataset. 
# 
