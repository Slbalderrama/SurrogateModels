#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:59:00 2020

@author: balderrama
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor,  RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, ExpSineSquared, RationalQuadratic 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, neighbors
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import sqrt as sq

#%% 
data = pd.read_excel('Data_Base.xls', index_col=0, Header=None)  
target = 'Renewable Penetration'

#%%

data_1 = data.copy()
data_2 = pd.DataFrame()

for i in range(50,570,50):
    foo = data_1.loc[data_1['HouseHolds']==i]
    data_2 = foo.sort_values(target, ascending=True)
    
    data_2.index = range(len(data_2))


    plt.scatter(data_2.index, data_2[target])

data_2['line'] = 0.85
plt.plot(data_2.index, data_2['line'])
plt.show()

#%%

y = pd.Series()
for i in data.index:
    
    if data['Renewable Penetration'][i] >= 0.85:
        y.loc[i] = 1 
    else:
        y.loc[i] = 0
        

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




y, X = shuffle(y, X, random_state=100)
#%%
n_estimators = [100, 300, 500]
max_depth = [ 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [ 5, 10,15] 
max_feature = [0.2,0.3,0.4]

hyperF = dict(n_estimators = n_estimators, 
              max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf,
            max_features =max_feature
                  )

forest = RandomForestClassifier(random_state=10)
gridF = GridSearchCV(forest, hyperF, cv = 5, verbose = 1, 
                      n_jobs = -1,scoring='neg_mean_absolute_error')

bestF = gridF.fit(X, y)
Best_Par = bestF.best_params_
Best_index = bestF.best_index_
Best_Score = bestF.best_score_
Results = bestF.cv_results_

print(Best_Par)
print(Best_index)
print(Best_Score)



#%%
start = time.time()
l1 = [1,1,1,1,1,1,1,1,1,1]
#l2 = [1,1,1,1,1,1,1,1,1,1]




kernel =  RBF(l1) #+ RBF(l2) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


gp = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=1000
                               , optimizer = 'fmin_l_bfgs_b')

gp = gp.fit(X_train, y_train)

R_2_train = round(gp.score(X_train,y_train),4)

print('R^2 for the gaussian process with the train data set is ' + str(R_2_train))

R_2_test = gp.score(X_test, y_test) 

print('R^2 for the gaussian process with the test data set is ' + str(R_2_test))

y_gp = gp.predict(X_test)
MAE_Random =  round(mean_absolute_error(y_test,y_gp),2)

print('MAE for the gaussian process is ' + str(MAE_Random))

end = time.time()
print('The Regression took ' + str(round(end - start,0)) + ' segundos')    

# gp.kernel_.get_params()
start = time.time()

#%%

# Cross Validation results
    
scoring =  ['neg_mean_absolute_error','r2'] #'r2' 'neg_mean_absolute_error' # 'neg_mean_squared_error'
for i in scoring:    
    l1 = [1,1,1,1,1,1,1,1,1,1]
    l2 = [1,1,1,1,1,1,1,1,1,1]
    
    kernel =   RBF(l1) + RBF(l2) 
    gp = GaussianProcessClassifier(kernel=kernel,optimizer = 'fmin_l_bfgs_b', 
                                  n_restarts_optimizer=3000)
    
    Results = cross_validate(gp, X, y, cv=5,return_train_score=True,n_jobs=-1
                             , scoring = i       )
    
    scores = Results['test_score']
    score = round(scores.mean(),4)
    
    if i == 'neg_mean_squared_error':
        score = sq(-score)    
        print(i + ' for the gaussian process with the test data set is ' + str(score))
    else:    
        print(i + ' for the gaussian process with the test data set is ' + str(score))

    Results = pd.DataFrame(Results)
    
    path = 'Results_Regressions/Kcross_valiadation_GP_classifier' + '_' +  i + '.csv'
    
    Results.to_csv(path)
