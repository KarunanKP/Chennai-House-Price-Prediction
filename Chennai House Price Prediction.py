#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)


# In[2]:


# Reading csv file
df = pd.read_csv('Chennai houseing sale.csv')
df.head(5)


# In[3]:


# Information about the Data
df.info()


# In[4]:


# Changing Date Column to Datetime Datatype
df['DATE_SALE'] = pd.to_datetime(df['DATE_SALE'], format= '%d-%m-%Y')
df['DATE_BUILD'] = pd.to_datetime(df['DATE_BUILD'], format= '%d-%m-%Y')
df['PRT_ID'] = df['PRT_ID'].str[1:].astype('int64')


# In[5]:


df.info()


# In[6]:


df.head(2)


# In[7]:


# Size of the data
df.shape


# In[8]:


# Checking for Null Values
df.isnull().sum()


# In[9]:


# Replacing the Null Values in 'QS_OVERALL' column by the Average of 'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM'.
df['QS_OVERALL'] = df['QS_OVERALL'].fillna(round(df[['QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM']].mean(axis=1), 2))


# In[10]:


# Droping Null Values
df.dropna(inplace=True)


# In[11]:


# Checking if Null Values are gone
df.isnull().sum()


# In[12]:


df.shape


# In[13]:


# Changing datatype to int
df['N_BEDROOM'] = df['N_BEDROOM'].fillna(0).astype('int64')
df['N_BATHROOM'] = df['N_BATHROOM'].fillna(0).astype('int64')


# In[14]:


df.info()


# In[15]:


# lets see the unique values of columns
for cols in df.columns:
    if df[cols].dtype == object:
        print()
        print(cols)
        print(df[cols].unique())


# ### There seems to be some misspells in the Dataset

# In[16]:


# Cleaning the Data 
df.AREA.replace(['Ana Nagar','Ann Nagar'],'Anna Nagar',inplace=True)
df.AREA.replace('Karapakam','Karapakkam',inplace=True)
df.AREA.replace(['Chrompt','Chrmpet','Chormpet','Chrompet'],'Chromepet',inplace=True)
df.AREA.replace('KKNagar','KK Nagar',inplace=True)
df.AREA.replace('TNagar','T Nagar',inplace=True)
df.AREA.replace('Adyr','Adyar',inplace=True)
df.AREA.replace('Velchery','Velachery',inplace=True)
df.BUILDTYPE.replace('Comercial','Commercial',inplace=True)
df.BUILDTYPE.replace('Other','Others',inplace=True)
df.UTILITY_AVAIL.replace('All Pub','AllPub',inplace=True)
df.SALE_COND.replace('Ab Normal','AbNormal',inplace=True)
df.SALE_COND.replace(['PartiaLl','Partiall'],'Partial',inplace=True)
df.SALE_COND.replace('Adj Land','AdjLand',inplace=True)
df.PARK_FACIL.replace('Noo','No',inplace=True)
df.STREET.replace('Pavd','Paved',inplace=True)
df.STREET.replace('NoAccess','No Access',inplace=True)


# In[17]:


# Checking if the changes has been applied
for cols in df.columns:
    if df[cols].dtype == object:
        print()
        print(cols)
        print(df[cols].unique())


# In[18]:


df.head(3)


# ## The Dataset is Cleaned and Ready for Analysis

# In[19]:


# Stastical Description of Data
df.describe()


# In[20]:


df.AREA.value_counts()


# In[21]:


# Distribution of Houses in various Areas
px.pie(df.groupby('AREA',as_index=False)['PRT_ID'].count(), values='PRT_ID', names='AREA',
       title='<b> Number of houses in various areas of Chennai</b>', labels={'PRT_ID':'Count'},
       color_discrete_sequence=px.colors.sequential.Plasma, hole=.5)


# ## Insights :
# - Chrompet has the highest number of houses among all areas in Chennai.
# - T Nagar has a lower number of houses compared to other areas in Chennai.

# In[22]:


px.box(df, x='AREA', y='SALES_PRICE', title= '<b>Sales Price of houses in various areas of Chennai',
    color= 'AREA')


# ## Insights :
# - T Nagar and Anna Nagar areas have the highest selling prices for houses, while KK Nagar comes in Third.
# - Houses in Karapakam area have lower selling prices compared to houses in other areas.

# In[23]:


px.scatter(df,x= 'INT_SQFT', y= 'SALES_PRICE', color= 'AREA', size= 'INT_SQFT',
           title= '<b> SQFT Versus SALES PRICE of Houses in Various Areas', labels= {'INT_SQFT': 'SQFT'})


# ## Insights : 
# - An increase in the square footage of houses is associated with an increase in the sales price of houses.
# - Houses in the same area tend to have similar square footage.
# - Houses in T Nagar and Anna Nagar areas have the highest prices, with square footage typically ranging from 1500 to 2000.
# - KK Nagar area houses have Higher square foot, typically ranging from 1400 to 2500.

# In[24]:


px.scatter(df, x='DIST_MAINROAD', y='SALES_PRICE', color='AREA', size='SALES_PRICE',
           title='<b> MAINROAD DISTANCE Versus SALES PRICE of Houses in Various Areas',
           labels= {'DIST_MAINROAD':'Distance from Mainroad'})


# ## Insights :
# - It appears that the distance to the main road does not significantly affect the sales price of houses. 
# - Houses with both shorter and longer distances to the main road have similar prices, indicating that the main road distance does not have a strong impact on the sales price.

# In[25]:


fig=px.scatter(df, x='DATE_BUILD', y='SALES_PRICE', color='AREA', size='SALES_PRICE',
               title='<b> Build Date Vs Sales Price')
fig.show()
fig=px.scatter(df, x='DATE_SALE', y='SALES_PRICE', color='AREA', size='SALES_PRICE',
               title='<b> Sale Date Vs Sales Price')
fig.show()


# ## Insights :
# - The columns 'DATE_BUILD' and 'DATE_SALE' do not appear to have a significant impact on the 'SALES_PRICE' of properties.
# - This suggests that the dates when the properties were built and sold do not directly influence their selling prices.

# In[26]:


px.box(df, x='N_ROOM', y='SALES_PRICE', color='AREA', labels= {'N_ROOM':'No.of Rooms'},
       title='<b> Total Rooms Versus Sales Price of Houses in Various Areas')


# ## Insights: 
# - There is a positive correlation between the number of rooms ('N_ROOM') and the sales price ('SALES_PRICE'). As the number of rooms increases, the sales price also tends to increase.
# 
# - The majority of houses in the dataset have 4 to 5 rooms, indicating that this range is common among the properties.
# 
# - Only KK Nagar has houses with 6 rooms. This aligns with the previous insight showing that KK Nagar has the highest square footage houses, suggesting that larger houses with more rooms are more common in KK Nagar.
# 
# - Anna Nagar and T Nagar have houses predominantly with 4 to 5 rooms. Despite this, they have higher sales prices, indicating that these areas are desirable and command higher prices despite having fewer properties with more rooms.

# In[27]:


px.box(df, x='N_BEDROOM', y='SALES_PRICE', color='AREA', labels= {'N_BEDROOM':'No.of Bedrooms'},
       title='<b>Total BedRooms Versus Sales Price of Houses in Various Areas')


# ## Insights:
# - There is a positive correlation between the number of bedrooms ('N_BEDROOM') and the sales price ('SALES_PRICE'). As the number of bedrooms increases, the sales price also tends to increase.
# 
# - While an increase in the number of bedrooms tends to increase the sales price, the sales are directly affected by the area where the property is located.
# 
# - The majority of houses in the dataset have 1 to 2 bedrooms, indicating that this range is common among the properties.
# 
# - KK Nagar and Velachery are the only areas with 3-bedroom houses, but KK Nagar also has 4-bedroom houses, indicating a wider range of housing options in KK Nagar.
# 
# - Anna Nagar and T Nagar have houses predominantly with 1 to 2 bedrooms. Despite this, they have higher sales prices, suggesting that the area itself has a direct effect on the sales price, regardless of the number of bedrooms in the properties.

# In[28]:


px.box(df,x='N_BATHROOM', y='SALES_PRICE', color='AREA', labels= {'N_BATHROOM':'No.of Bathrooms'},
       title='<b>Total BathRooms Versus Sales Price of Houses in Various Areas')


# ## Insights:
# - There is a positive correlation between the number of bathrooms ('N_BATHROOM') and the sales price ('SALES_PRICE'). As the number of bathrooms increases, the sales price also tends to increase.
# 
# - More than half of the houses in the dataset have 1 bathroom, indicating that this is the most common number of bathrooms among the properties.
# 
# - Anna Nagar and T Nagar have houses with only a single bathroom. Despite this, they have higher sales prices, suggesting that these areas are desirable and command higher prices despite having fewer bathrooms in the properties.

# In[29]:


px.box(df, x='SALE_COND', y='SALES_PRICE', color='AREA', labels= {'SALE_COND':'Sale Condition'},
       title='<b>Sale Condition Versus Sales Price of Houses in various Areas')


# ## Insights:
# - There doesn't seem to be much difference in sales price based on sale condition. 
# - The sale condition doesn't appear to have a significant impact on the sales price of properties.

# In[30]:


px.box(df, x='PARK_FACIL', y='SALES_PRICE', color='AREA', labels= {'PARK_FACIL':'Parking Facility'},
       title='<b>Parking Facility Versus Sales Price of Houses in Different Areas')


# ## Insight:
# - Houses with a parking facility tend to have slightly higher prices across different areas.
# - This suggests that having a parking facility is a desirable feature that can positively impact the sales price of a house.

# In[31]:


px.box(df, x='BUILDTYPE', y='SALES_PRICE', color='AREA',
       title='<b>Build Type Versus Sales Price of Houses in Different Areas')


# ## Insights:
# - Commercial houses appear to have significantly higher prices compared to residential houses and other property types in different areas.

# In[32]:


px.box(df, x='UTILITY_AVAIL', y='SALES_PRICE', color='AREA',
       title='<b> Utility Available Versus Sales Price of Houses in Different Areas')


# ## Insight:
# - Across different utility availability types there isn't a significant difference in sales prices.
# - This suggests that in this dataset utility availability may not be a major factor affecting property sales prices.

# In[33]:


px.box(df, x='STREET', y='SALES_PRICE', color='AREA',
       title='<b> Street Versus Sales Price of Houses in Different Areas')


# ## Insight:
# - There isn't much difference in sales price between properties located on a Paved street and those on a Gravel street.
# - Properties with no street access have lower sales prices compared to properties on both Paved and Gravel streets.

# In[34]:


px.box(df, x='MZZONE', y='SALES_PRICE', color='AREA',
       title='<b> MZZONE Versus Sales Price of Houses in Different Areas')


# ## Insight:
# - Houses in RM (Residential Medium Density) zones tend to have the highest sales prices among all zones, followed by RL (Residential Low Density) and RH (Residential High Density) zones.
# 
# - Houses in A (Agricultural) zones tend to have lower sales prices compared to all, typically indicates areas designated for agricultural use.
# 
# - The areas 'Anna Nagar', 'Chromepet', 'KK Nagar', and 'T Nagar' only have houses in residential zones, indicating that these areas are primarily residential in nature.

# In[35]:


numeric_columns = ['INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'REG_FEE', 'COMMIS', 'SALES_PRICE']
numeric_df = df[numeric_columns]

corr_matrix = numeric_df.corr()
numeric_df.corr()


# In[36]:


plt.figure(figsize=(12,6))
sns.heatmap(numeric_df.corr(),annot=True,cmap='GnBu')
plt.title('Correlation of Numeric Column')
plt.show()


# In[37]:


df.head(5)


# In[38]:


# Droping Unnecessary Columns based on our insights
df.drop(['DATE_SALE', 'DATE_BUILD', 'DIST_MAINROAD', 'SALE_COND','UTILITY_AVAIL', 'STREET',
         'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'REG_FEE', 'COMMIS'], axis=1, inplace=True)
df


# In[39]:


# Creating a backup file
df_bk=df.copy()


# In[40]:


df_bk


# # Coverting the labels into a numeric form using Label Encoder

# In[41]:


from sklearn.preprocessing import LabelEncoder
encoders = {}
original_categories = {}

# Iterate over each column
for col in df.select_dtypes(include='object').columns:
    # Create a LabelEncoder object
    encoders[col] = LabelEncoder()
    # Fit and transform the data for each column
    df[col] = encoders[col].fit_transform(df[col])
    # Store the original categories
    original_categories[col] = encoders[col].classes_

    # Print original categories and their corresponding encoded values
    print(f"Column: {col}")
    for category, encoded_value in zip(original_categories[col], encoders[col].transform(original_categories[col])):
        print(f"  Original Category: {category}  Encoded Value: {encoded_value}")


# In[42]:


df.head()


# In[43]:


df.info()


# In[44]:


# storing the Dependent Variables in X and Independent Variable in Y
x=df.drop('SALES_PRICE',axis=1)
y=df['SALES_PRICE']


# In[45]:


# Splitting the Data into Training set and Testing Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[46]:


# Scaling the values to convert the int values to Machine Languages
from sklearn.preprocessing import MinMaxScaler
mmscaler=MinMaxScaler(feature_range=(0,1))
x_train=mmscaler.fit_transform(x_train)
x_test=mmscaler.fit_transform(x_test)
x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test)


# In[47]:


# Creating a Dataframe to store the metrics score
a={'Model Name':[], 'Mean_Absolute_Error MAE':[] ,'Mean_Absolute_Percentage_Error MAPE':[] ,'Mean_Squared_Error MSE':[],'Root_Mean_Squared_Error RMSE':[] ,'Root_Mean_Squared_Log_Error RMSLE':[] ,'R2 score':[],'Adj_R_Square':[]}
Results=pd.DataFrame(a)
Results.head()


# In[48]:


# Build the Regression / Regressor models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[49]:


# Create objects of Regression / Regressor models with default hyper-parameters

modelmlg = LinearRegression()
modeldcr = DecisionTreeRegressor()
modelrfr = RandomForestRegressor()
modelSVR = SVR()
modelXGR = xgb.XGBRegressor()
modelKNN = KNeighborsRegressor(n_neighbors=5)
modelETR = ExtraTreesRegressor()
modelGBR = GradientBoostingRegressor()


# In[50]:


MM = [modelmlg, modeldcr, modelrfr, modelKNN, modelETR, modelGBR, modelXGR]

for models in MM:
    
    # Fit the model with train data
    models.fit(x_train, y_train)
    
    # Predict the model with test data
    y_pred = models.predict(x_test)
    
    # Print the model name
    print('Model Name: ', models)
    
    # Evaluation metrics for Regression analysis
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
    from sklearn import metrics

    # Assuming y_true are the actual values and y_pred are the predicted values
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmsle = np.log(rmse)
    r_squared = r2_score(y_test, y_pred)

    # Adjusted R-squared
    n = len(y_test)
    p = x.shape[1]  # Number of features
    adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Root Mean Squared Log Error (RMSLE): {rmsle}")
    print(f"R-squared (R^2): {r_squared}")
    print(f"Adjusted R-squared: {adj_r_squared}")
    print('------------------------------------------------------------------------------------------------------------')
    
#-------------------------------------------------------------------------------------------
    new_row = pd.DataFrame({'Model Name': [str(models)],
                            'Mean_Absolute_Error MAE': [metrics.mean_absolute_error(y_test, y_pred)],
                            'Mean_Absolute_Percentage_Error MAPE': [np.mean(np.abs((y_test - y_pred) / y_test)) * 100],
                            'Mean_Squared_Error MSE': [metrics.mean_squared_error(y_test, y_pred)],
                            'Root_Mean_Squared_Error RMSE': [np.sqrt(metrics.mean_squared_error(y_test, y_pred))],
                            'Root_Mean_Squared_Log_Error RMSLE': [np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))],
                            'R2 score': [metrics.r2_score(y_test, y_pred)],
                            'Adj_R_Square': [adj_r_squared]})

    # Append new_row to Results
    Results = pd.concat([Results, new_row], ignore_index=True)


# In[51]:


Results


# From the above Results, the Top 3 Models by Comparing Adjacent R Square Values are¶
# 
# - XGBRegressor
# - GradientBoostingRegressor
# - ExtraTreesRegressor
# 
# Training and Predicting with XGBRegressor

# In[52]:


# Training the Model
modelXGR.fit(x_train, y_train)
    
# Predict the model with test data
y_pred = modelXGR.predict(x_test)


# In[53]:


out=pd.DataFrame({'Price_actual':y_test,'Price_pred':y_pred})
result=df_bk.merge(out,left_index=True,right_index=True)


# In[54]:


result


# In[55]:


result[['PRT_ID','AREA','Price_actual','Price_pred']].sample(20)


# In[56]:


px.scatter(result, x='Price_actual', y='Price_pred', trendline='ols', color_discrete_sequence=['magenta'],
           template='plotly_dark', title='<b> Actual Price  Vs  Predicted Price ')


# In[ ]:




