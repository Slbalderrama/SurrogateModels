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

demand = load('Demand_Chaco.joblib')
NPC = load('NPC.joblib')
LCOE = load('LCOE.joblib')

X = pd.read_csv('independant_variables.csv', index_col=0)  

X['Renewable Energy Unit Total'] = np.random.uniform(low=430, high=460, size=(len(X.index),))
X = round(X,2)
X1 = X.loc[X['HouseHolds']>60]
X1 = X1.loc[X['HouseHolds']<500]


Results = pd.DataFrame()


x = np.array(X1['HouseHolds'])
x = x.reshape(-1, 1)

e = 0.12
y=20

a = e*(1+e)**y
b = (1+e)**y -1
    
CFR  = a/b

    
Results['Demand_Discounted'] = pd.DataFrame(demand.predict(x))[0]/CFR 
Results['NPC'] = pd.DataFrame(NPC.predict(X1))[0]
Results['LCOE'] = pd.DataFrame(LCOE.predict(X1))[0]
Results['LCOE 1'] = Results['NPC']/Results['Demand_Discounted']
Results['Dif'] = Results['LCOE'] - Results['LCOE 1']

print(mean_absolute_error(Results['LCOE'], Results['LCOE 1']))

