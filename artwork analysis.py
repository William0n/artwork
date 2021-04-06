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
plt.xlabel('Width (mm)')
plt.ylabel('Height (mm)')
plt.show()

len(data['artist'].unique())

### seeing which artist has the most artwork 

data['artist'].value_counts()[[0,1,2,3,4]].sort_values().plot(kind = 'barh')

jmw_data = data[data['artist'].str.contains('Turner, Joseph Mallord William')]

plt.hist(jmw_data['year'], bins = 20)
plt.xlabel('Year Created')
plt.show()

### Creating KDE  plot 
from scipy.stats import gaussian_kde

jmw_data = jmw_data['year'].dropna()

jmw_data = pd.DataFrame(jmw_data)
density = gaussian_kde(jmw_data['year'])
xs = np.linspace(1700,1900, 100)

plt.plot(xs, density(xs))
plt.xlabel('Year Created')
plt.show()

data['medium'].value_counts()
len(data['medium'].unique())
data['medium'].value_counts()[[0,1,2,3,4,5,6,7,8,9]].sort_values().plot(kind = 'barh')
 
aspect = data['width']/ data['height']



plt.scatter(data['year'], aspect)
plt.xlabel('Year Created')
plt.ylabel('Aspect Ratio')
plt.ylim(0, 120)
plt.show()

### closer look at the scatter plot 
plt.scatter(data['year'], aspect)
plt.xlabel('Year Created')
plt.ylabel('Aspect Ratio')
plt.ylim(0, 20)
plt.show()



### Removing unnecessary columns, NA values, and creating new column. Then splitting into train & test sets

list(data)

top_10_name = data['medium'].value_counts()[[0,1,2,3,4,5,6,7,8,9]].index.tolist()

### This function returns a integer from 0-10 depending on the name of the medium used in the
### artwork. Note: Only the top 10 most frequently used mediums are considered. 

def categorizer(x):
    if x['medium'] == top_10_name[0]:
        return 0 
    elif x['medium'] == top_10_name[1]:
        return 1 
    elif x['medium'] == top_10_name[2]:
        return 2 
    elif x['medium'] == top_10_name[3]:
        return 3 
    elif x['medium'] == top_10_name[4]:
        return 4 
    elif x['medium'] == top_10_name[5]:
        return 5 
    elif x['medium'] == top_10_name[6]:
        return 6 
    elif x['medium'] == top_10_name[7]:
        return 7 
    elif x['medium'] == top_10_name[8]:
        return 8 
    elif x['medium'] == top_10_name[9]:
        return 9 
    else: 
        return 10   
data['material'] = data.apply(categorizer, axis =1)


dim_data = data.iloc[:,[8,11,12, 15]]
dim_data.isna().sum()
dim_data = dim_data.dropna()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dim_data.iloc[:,1:4], dim_data.iloc[:,0], random_state = 20)


### Random Forest modeling 

from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


model_rf = RandomForestRegressor(n_estimators = 600, max_leaf_nodes = 16, max_depth = 20, n_jobs = -1)
model_rf.fit(x_train, y_train)

y_pred = model_rf.predict(x_test)



mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)

for name, score in zip(list(dim_data)[1:], model_rf.feature_importances_):
    print(name,score)


### Improving the model 

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 4, 8]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
               }

model_rf_t = RandomForestRegressor()

rf_search =RandomizedSearchCV(estimator = model_rf_t, param_distributions = random_grid,
                              n_iter = 5, cv = 5, n_jobs = -1, random_state= 20)

rf_search.fit(x_train, y_train)
rf_search.best_params_

new_model = rf_search.best_estimator_
y_new_pred = new_model.predict(x_test)

mean_absolute_error(y_test, y_new_pred)
mean_squared_error(y_test, y_new_pred)

for name, score in zip(list(dim_data)[1:], new_model.feature_importances_):
    print(name,score)

