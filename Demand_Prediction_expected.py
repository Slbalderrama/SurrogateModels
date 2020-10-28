#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:57:19 2020

@author: balderrama
"""
import pandas as pd
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from math import sqrt as sq
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump,load
#%%
Demand = pd. DataFrame()

for i in range(50, 570,50):
    
    Village = 'village_' + str(i)
    Energy_Demand = pd.read_excel('Example/Demand.xls',sheet_name=Village
                                  ,index_col=0,Header=None)
    
    
    Demand[i] = Energy_Demand[1]/1000
    

e = 0.12
y=20

a = e*(1+e)**y
b = (1+e)**y -1
    
CFR  = a/b

Demand_discounted = Demand.sum()/CFR



#%%
y = Demand_discounted
X = list(Demand_discounted.index)
#%%

y, X = shuffle(y, X, random_state=10)
#%%
X = np.array(X)
X = X.reshape(-1, 1)
#%%

scoring = 'r2' #'r2' 'neg_mean_absolute_error' # 'neg_mean_squared_error'

lm = linear_model.LinearRegression(fit_intercept=True)

Results = cross_validate(lm, X, y, cv=2,return_train_score=True,n_jobs=-1
                         , scoring = scoring       )

scores = Results['test_score']
score = scores.mean()
if scoring == 'neg_mean_squared_error':
    score = sq(-score)    
    print(scoring + ' for the linear regression with the test data set is ' + str(score))
else:    
    print(scoring + ' for the linear regression with the test data set is ' + str(score))
#%%

lm1 = linear_model.LinearRegression(fit_intercept=True)
lm1 = lm1.fit(X, y)    
dump(lm1, 'Demand_Chaco.joblib')  
#%%

NPC = load('NPC.joblib')
LCOE = load('LCOE.joblib')

data = pd.read_excel('Data_Base.xls', index_col=0, Header=None)  

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data['Renewable Unitary Invesment Cost']   
X['Battery Unitary Invesment Cost'] = data['Battery Unitary Invesment Cost']
X['Deep of Discharge'] = data['Deep of Discharge']
X['Battery Cycles'] = data['Battery Cycles']
X['GenSet Unitary Invesment Cost'] = data['GenSet Unitary Invesment Cost']
X['Generator Efficiency'] = data['Generator Efficiency']
X['Low Heating Value'] = data['Low Heating Value']
X['Fuel Cost'] = data['Fuel Cost']
X['HouseHolds'] = data['HouseHolds']
X['Renewable Energy Unit Total'] = data['Renewable Energy Unit Total']
X.index =range(len(X))

Results = pd.DataFrame()


x = np.array(X['HouseHolds'])
x = x.reshape(-1, 1)
    
Results['Demand_Discounted'] = pd.DataFrame(lm1.predict(x))[0] 
Results['NPC'] = pd.DataFrame(NPC.predict(X))[0]
Results['LCOE'] = pd.DataFrame(LCOE.predict(X))[0]
Results['LCOE 1'] = Results['NPC']/Results['Demand_Discounted']

print(r2_score(Results['LCOE'], Results['LCOE 1']))

#%%

Results1 = pd.DataFrame()


Results1['Demand_Discounted'] = data['Total Demand'].tolist()
Results1['Demand_Discounted'] = Results1['Demand_Discounted']/CFR
Results1['Demand_Discounted 1'] = pd.DataFrame(lm1.predict(x))[0] 
Results1['NPC'] = data['NPC'].tolist()
Results1['LCOE'] = data['LCOE'].tolist()
Results1['LCOE 1'] = Results['NPC']/Results1['Demand_Discounted']
Results1['LCOE 2'] = Results['NPC']/Results1['Demand_Discounted 1']
Results1['Dif'] = abs(Results1['LCOE'] - Results1['LCOE 1'])
Results1['Dif 1'] = abs(Results1['LCOE 1'] - Results1['LCOE 2'])

print(r2_score(Results1['LCOE'], Results1['LCOE 2']))




