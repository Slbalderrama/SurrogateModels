#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:07:51 2020

@author: balderrama
"""

import pandas as pd


data_lowlands = {}
grid = {}
microgrids = {}
sa_pv = {}
sa_diesel = {}

prices = ['018', '58', '1']

for i in prices:
    path = 'Diesel_' + i + '.csv'
    data = pd.read_csv(path)
    
    data_lowlands = data.loc[data['Elevation']<800]  
    
    grid[i] = data_lowlands.loc[data_lowlands['FinalElecCode2025']==1]
    
    microgrids[i] = data_lowlands.loc[data_lowlands['FinalElecCode2025'] ==4] 
    
    sa_pv[i] = data_lowlands.loc[data_lowlands['FinalElecCode2025'] ==3]
    
    sa_diesel[i] = data_lowlands.loc[data_lowlands['FinalElecCode2025'] ==2]

#%% 5190
foo = 0    
name = ['grid ',' microgrids ', 'sa PV ', 'sa diesel ']

for j in prices:
    
    Population = 0
    Invesment = 0
    Capacity = 0
    
    for i in [grid, microgrids, sa_pv, sa_diesel]:
    
        
        tech = i[j]
        population = round(tech['Pop2025'].sum(),0)
        invesment = round(tech['InvestmentCost2025'].sum()/1000000,0)
        capacity = round(tech['NewCapacity2025'].sum()/1000,0)
        
        Population += population
        Invesment  += invesment
        Capacity   += capacity
        

        print('Population reached by ' + name[foo] + ' is ' + str(population) + ' in scenario ' + j + '.')
        print('Invesment in ' + name[foo] + ' is ' + str(invesment) + ' in scenario ' + j + '.')
        print('Installed capacity in ' + name[foo] + ' is ' + str(capacity) + ' in scenario ' + j + '.' )
        
    foo += 1
    print('Total population is ' + str(Population) + ' in scenario ' + j + '.')
    print('Total Invesment is ' + str(Invesment) + ' in scenario ' + j + '.')
    print('Total Installed capacity is ' + str(Capacity) + ' in scenario ' + j + '.' )