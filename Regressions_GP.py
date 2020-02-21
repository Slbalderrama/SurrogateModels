#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:03:33 2019

@author: balderrama
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import pydot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
#%%
# Data manipulation
data = pd.read_excel('Data_Base_1.xls', index_col=0, Header=None)   
#data = data.loc[data['Gap']< 1]

y = pd.DataFrame()
target='NPC'
y[target] = data[target]

y=y.astype('float')

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data['Renewable Unitary Invesment Cost']   
X['Battery Unitary Invesment Cost'] = data['Battery Unitary Invesment Cost']
X['Deep of Discharge'] = data['Deep of Discharge']
X['Battery Cycles'] = data['Battery Cycles']
X['GenSet Unitary Invesment Cost'] = data['GenSet Unitary Invesment Cost']
X['Generator Efficiency'] = data['Generator Efficiency']
X['Low Heating Value'] = data['Low Heating Value']
X['Fuel Cost'] = data['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X['HouseHolds'] = data['HouseHolds']
X['Renewable Energy Unit Total'] = data['Renewable Energy Unit Total']
#X['Max Demand'] = data['Max Demand']
#X['Y'] = data['Y']


feature_list = list(X.columns)
y, X = shuffle(y, X, random_state=10)
#%%
y=np.array(y)
y = y.ravel() 
#%%

from sklearn.preprocessing import MinMaxScaler



#y = np.array(y)
#y = y.reshape(1, -1) 


y = MinMaxScaler().fit_transform(y)


X = MinMaxScaler().fit_transform(X)

#y = y.transpose()
X = pd.DataFrame(X, columns=feature_list)
y = pd.DataFrame(y, columns=[target])


#%%

# Cross Validation results
l = [1,1,1,1,1,1,1,1,1,1]
#l = [8.39355389e+02, 3.08601304e+02, 2.76926443e-01, 1.11221067e+04,
#       1.48025407e+03, 8.80945445e-02, 3.32911702e+00, 4.34149233e-01,
#        1.49795551e+02, 1.57835288e+02]
#kernel = (C()**2)*RBF(l)
kernel =  RBF(l)
gp = GaussianProcessRegressor(kernel=kernel,optimizer = 'fmin_l_bfgs_b', 
                              n_restarts_optimizer=3000)

Results = cross_validate(gp, X, y, cv=5,return_train_score=True,n_jobs=-1
                         , scoring = 'r2' #'neg_mean_absolute_error'
                         )
scores = Results['test_score']
scores.mean()

Results = pd.DataFrame(Results)


#%%

# Cross Validation 
l = [1,1,1,1,1,1,1,1,1,1]
#l = [8.39355389e+02, 3.08601304e+02, 2.76926443e-01, 1.11221067e+04,
#       1.48025407e+03, 8.80945445e-02, 3.32911702e+00, 4.34149233e-01,
#        1.49795551e+02, 1.57835288e+02]
#kernel = (C()**2)*RBF(l)
kernel =  RBF(l)
gp = GaussianProcessRegressor(kernel=kernel,optimizer = 'fmin_l_bfgs_b', 
                              n_restarts_optimizer=3000)

Results = cross_validate(gp, X, y, cv=5,return_train_score=True,n_jobs=-1
                         , scoring = 'r2' #'neg_mean_absolute_error'
                         )
scores = Results['test_score']
scores.mean()


#%%
from sklearn.model_selection import train_test_split
#l = [1,1,1,1,1,1,1,1,1,1]
l = [8.39355389e+02, 3.08601304e+02, 2.76926443e-01, 1.11221067e+04,
        1.48025407e+03, 8.80945445e-02, 3.32911702e+00, 4.34149233e-01,
        1.49795551e+02, 1.57835288e+02]
#kernel =  (C()**2)*RBF(l)
kernel =  RBF(l)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1,
                                                    random_state=500)

gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=12,
                              optimizer = 'fmin_l_bfgs_b'
#                              , normalize_y=True
                              
                              )

gp = gp.fit(X_train, y_train)

R_2_train = round(gp.score(X_train,y_train),4)

print('R^2 for the gaussian process with the train data set is ' + str(R_2_train))

R_2_test = gp.score(X_test, y_test) 

print('R^2 for the gaussian process with the test data set is ' + str(R_2_test))

y_gp = gp.predict(X_test)
MAE_Random =  round(mean_absolute_error(y_test,y_gp),2)

print('MAE for the gaussian process is ' + str(MAE_Random))

#%%
# Predic vs computed plot

import matplotlib.pyplot as plt


l = [8.39355389e+02, 3.08601304e+02, 2.76926443e-01, 1.11221067e+04,
        1.48025407e+03, 8.80945445e-02, 3.32911702e+00, 4.34149233e-01,
        1.49795551e+02, 1.57835288e+02]
kernel =  RBF(l)

gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=12,
                              optimizer = 'fmin_l_bfgs_b'
#                              , normalize_y=True
    )

y_gp = cross_val_predict(gp, X, y, cv=5,n_jobs=-1)


fig, ax = plt.subplots()
ax.scatter(y/1000, y_gp/1000,s = 5)
ax.plot([y.min()/1000, y.max()/1000], [y.min()/1000, y.max()/1000], 'k-', lw=2)
ax.set_xlabel('Measured (thousand USD)')
ax.set_ylabel('Predicted (thousand USD)')
plt.show()
















