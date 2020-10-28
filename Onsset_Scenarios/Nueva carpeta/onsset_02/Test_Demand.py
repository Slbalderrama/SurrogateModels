#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:49:46 2020

@author: balderrama
"""

import pandas as pd

#%%

'''Test with an technology, it is  also check that the demand is not change with amazonia data'''

#%%

data = pd.read_csv('Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  
data1 = pd.read_csv('Bolivia/Multy_Tech_Test_4.csv', index_col=0)  


#%%
df = pd.DataFrame()

df['EnergyPerSettlement2020'] = data['EnergyPerSettlement2020']
df['EnergyPerSettlement2020 1'] = data1['EnergyPerSettlement2020']
df['EnergyPerSettlement2030'] = data['EnergyPerSettlement2030']
df['EnergyPerSettlement2030 1'] = data1['EnergyPerSettlement2030']
df['TotalEnergyPerCell'] = data['TotalEnergyPerCell']
df['TotalEnergyPerCell 1'] = data1['TotalEnergyPerCell']
df['Demand_Name'] = data['Demand_Name']
df['Elevation'] = data['Elevation']
df['Y_deg'] = data['Y_deg']
df['Pop2012'] = data['Pop2012']