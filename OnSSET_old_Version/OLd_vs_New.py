#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:06:19 2020

@author: balderrama
"""

import pandas as pd



data_new = pd.read_csv('OnSSET_New/Result_OnSSET_New.csv')   
data_old = pd.read_csv('OnSSET_Old/Base_case/Result_OnSSET_Old.csv')   

data_new['Pop2025'].sum() == data_old['Pop2025'].sum()  


#%%
################### GRID
# lcoe
lcoe = []
for i in data_new.index:
    
    if data_new['MinimumOverallCode2025'][i] == 1 and data_old['MinimumOverallCode2025'][i] == 1:
        if data_new['MinimumOverallLCOE2025'][i] != data_old['MinimumOverallLCOE2025'][i]:
            lcoe.append(i)
            print(i)
            
# invesment    
            
invesment = []
for i in data_new.index:
    
    if data_new['MinimumOverallCode2025'][i] == 1 and data_old['MinimumOverallCode2025'][i] == 1:
        if data_new['InvestmentCost2025'][i] != data_old['InvestmentCost2025'][i]:
            invesment.append(i)
            print(i)
# New Capacity

new_capacity = []
for i in data_new.index:
    if data_new['MinimumOverallCode2025'][i] == 1 and data_old['MinimumOverallCode2025'][i] == 1:
        if data_new['NewCapacity2025'][i] != data_old['NewCapacity2025'][i]:
            new_capacity.append(i)
            print(i)            
#%%            
################### SA diesel
# lcoe
lcoe_sa_diesel = []
for i in data_new.index:
    
    if data_new['MinimumOverallCode2025'][i] == 2 and data_old['MinimumOverallCode2025'][i] == 2:
        if data_new['MinimumOverallLCOE2025'][i] != data_old['MinimumOverallLCOE2025'][i]:
            lcoe_sa_diesel.append(i)
            print(i)
            
# invesment    
            
invesment_sa_diesel = []
for i in data_new.index:
    
    if data_new['MinimumOverallCode2025'][i] == 2 and data_old['MinimumOverallCode2025'][i] == 2:
        if data_new['InvestmentCost2025'][i] != data_old['InvestmentCost2025'][i]:
            invesment_sa_diesel.append(i)
            print(i)            

# New Capacity

new_capacity_diesel = []
for i in data_new.index:
    if data_new['MinimumOverallCode2025'][i] == 2 and data_old['MinimumOverallCode2025'][i] == 2:
        if data_new['NewCapacity2025'][i] != data_old['NewCapacity2025'][i]:
            new_capacity_diesel.append(i)
            print(i)    

#%%            
################### SA PV
# lcoe
lcoe_sa_PV = []
for i in data_new.index:
    
    if data_new['MinimumOverallCode2025'][i] == 3 and data_old['MinimumOverallCode2025'][i] == 3:
        if data_new['MinimumOverallLCOE2025'][i] != data_old['MinimumOverallLCOE2025'][i]:
            lcoe_sa_PV.append(i)
            print(i)
            
# invesment    
            
invesment_sa_PV = []
for i in data_new.index:
    
    if data_new['MinimumOverallCode2025'][i] == 3 and data_old['MinimumOverallCode2025'][i] == 3:
        if data_new['InvestmentCost2025'][i] != data_old['InvestmentCost2025'][i]:
            invesment_sa_PV.append(i)
            print(i)            
            
new_capacity_PV = []
for i in data_new.index:
    if data_new['MinimumOverallCode2025'][i] == 3 and data_old['MinimumOverallCode2025'][i] == 3:
        if data_new['NewCapacity2025'][i] != data_old['NewCapacity2025'][i]:
            new_capacity_PV.append(i)
            print(i)               
            
#%%
### grid comparison 
grid_old = data_old.loc[data_old['MinimumOverallCode2025']==1]

grid_new = data_new.loc[data_new['MinimumOverallCode2025']==1]            
            
Comunities_grid_old = len(grid_old)            
Comunities_grid_new = len(grid_new)            
            
print('OnSSET old algorith has ' + str(Comunities_grid_old) + ' communities electrified by the grid')            
print('OnSSET new algorith has ' + str(Comunities_grid_new) + ' communities electrified by the grid')                 
            
Invesment_grid_old = round(grid_old['InvestmentCost2025'].sum()/1000000,0)            
Invesment_grid_new = round(grid_new['InvestmentCost2025'].sum()/1000000,0)                 
            
print('OnSSET old algorith has an investment of ' + str(Invesment_grid_old) + ' millions USD in grid')            
print('OnSSET new algorith has an investment of ' + str(Invesment_grid_new) + ' millions USD in grid')                        


Pop_grid_old = round(grid_old['Pop2025'].sum()/1000000,2)            
Pop_grid_new = round(grid_new['Pop2025'].sum()/1000000,2)                 
            
print('OnSSET old algorith, persons served by the grid are ' + str(Pop_grid_old) + ' millions')            
print('OnSSET new algorith, persons served by the grid are  ' + str(Pop_grid_new) + ' millions')   


Capacity_grid_old = round(grid_old['NewCapacity2025'].sum()/1000)            
Capacity_grid_new = round(grid_new['NewCapacity2025'].sum()/1000)                 
            
print('OnSSET old algorith has an install capacity of ' + str(Capacity_grid_old) + ' MW')            
print('OnSSET new algorith has an install capacity of ' + str(Capacity_grid_old) + ' MW')  

#%%

# SA diesel

diesel_old = data_old.loc[data_old['MinimumOverallCode2025']==2]

diesel_new = data_new.loc[data_new['MinimumOverallCode2025']==2]            
            
Comunities_diesel_old = len(diesel_old)            
Comunities_diesel_new = len(diesel_new)            
            
print('OnSSET old algorith has ' + str(Comunities_diesel_old) + ' communities electrified by Stand alone diesel system')            
print('OnSSET new algorith has ' + str(Comunities_diesel_new) + ' communities electrified by Stand alone diesel system')                 
            
Invesment_diesel_old = round(diesel_old['InvestmentCost2025'].sum()/1000000,0)            
Invesment_diesel_new = round(diesel_new['InvestmentCost2025'].sum()/1000000,0)                 
            
print('OnSSET old algorith has an investment of ' + str(Invesment_diesel_old) + ' millions USD in stand alone systems')            
print('OnSSET new algorith has an investment of ' + str(Invesment_diesel_new) + ' millions USD in stand alone systems')                        


Pop_diesel_old = round(diesel_old['Pop2025'].sum()/1000000,2)            
Pop_diesel_new = round(diesel_new['Pop2025'].sum()/1000000,2)                 
            
print('OnSSET old algorith, persons served by stand alone diesel systems are ' + str(Pop_diesel_old) + ' millions')            
print('OnSSET new algorith, persons served by stand alone diesel systems are ' + str(Pop_diesel_new) + ' millions')   


Capacity_diesel_old = round(diesel_old['NewCapacity2025'].sum()/1000)            
Capacity_diesel_new = round(diesel_new['NewCapacity2025'].sum()/1000)                 
            
print('OnSSET old algorith has an install capacity of ' + str(Capacity_diesel_old) + ' MW of gensets')            
print('OnSSET new algorith has an install capacity of ' + str(Capacity_diesel_new) + ' MW of gensets')  



#%%

# SA PV

PV_old = data_old.loc[data_old['MinimumOverallCode2025']==3]
PV_new = data_new.loc[data_new['MinimumOverallCode2025']==3]            
            
Comunities_PV_old = len(PV_old)            
Comunities_PV_new = len(PV_new)            
            
print('OnSSET old algorith has ' + str(Comunities_PV_old) + ' communities electrified by Stand alone PV systems')            
print('OnSSET new algorith has ' + str(Comunities_PV_new) + ' communities electrified by Stand alone PV systems')                 
            
Invesment_PV_old = round(PV_old['InvestmentCost2025'].sum()/1000000,0)            
Invesment_PV_new = round(PV_new['InvestmentCost2025'].sum()/1000000,0)                 
            
print('OnSSET old algorith has an investment of ' + str(Invesment_PV_old) + ' millions USD in stand alone PV systems')            
print('OnSSET new algorith has an investment of ' + str(Invesment_PV_new) + ' millions USD in stand alone PV systems')                        


Pop_PV_old = round(PV_old['Pop2025'].sum()/1000000,2)            
Pop_PV_new = round(PV_new['Pop2025'].sum()/1000000,2)                 
            
print('OnSSET old algorith, persons served by stand alone PV systems are ' + str(Pop_PV_old) + ' millions')            
print('OnSSET new algorith, persons served by stand alone PV systems are ' + str(Pop_PV_new) + ' millions')   


Capacity_PV_old = round(PV_old['NewCapacity2025'].sum()/1000)            
Capacity_PV_new = round(PV_new['NewCapacity2025'].sum()/1000)                 
            
print('OnSSET old algorith has an install capacity of ' + str(Capacity_PV_old) + ' MW of PV')            
print('OnSSET new algorith has an install capacity of ' + str(Capacity_PV_new) + ' MW of PV')  


#%%

# Hybrid

Hybrid_new = data_new.loc[data_new['MinimumOverallCode2025']==4]            
            
Comunities_Hybrid_new = len(Hybrid_new)            
            
print('OnSSET new algorith has ' + str(Comunities_Hybrid_new) + ' communities electrified by Hybrid systems')                 
            
Invesment_Hybrid_new = round(Hybrid_new['InvestmentCost2025'].sum()/1000000,0)                 
            
print('OnSSET new algorith has an investment of ' + str(Invesment_Hybrid_new) + ' millions USD in Hybrid systems')                        


Pop_Hybrid_new = round(Hybrid_new['Pop2025'].sum()/1000000,2)                 
            
print('OnSSET new algorith, persons served by Hybrid systems are ' + str(Pop_Hybrid_new) + ' millions')   


Capacity_Hybrid_new = round(Hybrid_new['NewCapacity2025'].sum()/1000)                 
            
print('OnSSET new algorith has an install capacity of ' + str(Capacity_Hybrid_new) + ' MW of PV')  

#%%

Comunities_Hybrid_new + Comunities_PV_new + Comunities_diesel_new + Comunities_grid_new == len(data_new)
Invesment_Hybrid_new + Invesment_PV_new + Invesment_diesel_new + Invesment_grid_new == round(data_new['InvestmentCost2025'].sum()/1000000,0)
Hybrid_new['Pop2025'].sum() + PV_new['Pop2025'].sum() + diesel_new['Pop2025'].sum() +grid_new['Pop2025'].sum() == data_new['Pop2025'].sum()
Capacity_Hybrid_new + Capacity_PV_new + Capacity_diesel_new + Capacity_grid_new == round(data_new['NewCapacity2025'].sum()/1000) 















