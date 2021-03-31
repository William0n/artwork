
"""
Created on Thu Mar  4 14:38:50 2021

@author: William
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 

data = pd.read_csv('art data.csv',encoding='cp1252')
data.describe()

plt.scatter(data['width'],data['height'] )
len(data['artist'].unique())

### seeing which artist has the most artwork 

data['artist'].value_counts()[[0,1,2,3,4]].sort_values().plot(kind = 'barh')

jmw_data = data[data['artist'].str.contains('Turner, Joseph Mallord William')]

plt.hist(jmw_data['year'], bins = 20)

### Creating KDE  plot 
from scipy.stats import gaussian_kde

jmw_data = jmw_data['year'].dropna()

jmw_data = pd.DataFrame(jmw_data)
density = gaussian_kde(jmw_data['year'])
xs = np.linspace(1700,1900, 100)
plt.plot(xs, density(xs))




## Removing unnecessary columns, NA values, and creating new column. Then splitting into train & test sets

list(data)

dim_data = data.iloc[:,[8,11,12]]
dim_data.isna().sum()
dim_data = dim_data.dropna()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dim_data.iloc[:,1:3], dim_data.iloc[:,0], random_state = 20)


### Random Forest modeling 

from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error


model_rf = RandomForestRegressor(n_estimators = 1000, max_leaf_nodes = 18, n_jobs = -1)
model_rf.fit(x_train, y_train)
model_rf.feature_importances_

y_pred = model_rf.predict(x_test)
mean_absolute_error(y_test, y_pred)

