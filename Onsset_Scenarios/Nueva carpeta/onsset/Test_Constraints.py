#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:49:46 2020

@author: balderrama
"""

import pandas as pd


#%%

data = pd.read_csv('Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  

#%%
df = pd.DataFrame()


df['MG_Diesel_Highlands2030'] = data['MG_Diesel_Highlands2030']
df['MG_Diesel_Highlands2020'] = data['MG_Diesel_Highlands2020']
df['MG_Diesel_Amazonia2030']  = data['MG_Diesel_Amazonia2030']
df['MG_Diesel_Amazonia2020']  = data['MG_Diesel_Amazonia2020']
df['MG_Diesel_Chaco2030']     = data['MG_Diesel_Chaco2030']
df['MG_Diesel_Chaco2020']     = data['MG_Diesel_Chaco2020']
df['Elevation'] = data['Elevation']
df['Y_deg'] = data['Y_deg']


#%%
df1 = pd.DataFrame()
df1['SA_PV_Highlands2030'] = data['SA_PV_Highlands2030']
df1['SA_PV_Highlands2020'] = data['SA_PV_Highlands2020']
df1['SA_PV_Amazonia2030']  = data['SA_PV_Amazonia2030']
df1['SA_PV_Amazonia2020']  = data['SA_PV_Amazonia2020']
df1['SA_PV_Chaco2030']     = data['SA_PV_Chaco2030']
df1['SA_PV_Chaco2020']     = data['SA_PV_Chaco2020']
df1['Pop2012'] = data['Pop2012']

#%%

df2 = pd.DataFrame()
df2['SA_Diesel_Highlands2030'] = data['SA_Diesel_Highlands2030']
df2['SA_Diesel_Highlands2020'] = data['SA_Diesel_Highlands2020']
df2['SA_Diesel_Amazonia2030']  = data['SA_Diesel_Amazonia2030']
df2['SA_Diesel_Amazonia2020']  = data['SA_Diesel_Amazonia2020']
df2['SA_Diesel_Chaco2030']     = data['SA_Diesel_Chaco2030']
df2['SA_Diesel_Chaco2020']     = data['SA_Diesel_Chaco2020']
df2['PopStartYear'] = data['PopStartYear']

#%%

df3 = pd.DataFrame()
df3['MG_PV_Highlands2020'] = data['MG_PV_Highlands2020']
df3['MG_PV_Highlands2030'] = data['MG_PV_Highlands2030']
df3['MG_PV_Amazonia2020'] = data['MG_PV_Amazonia2020']
df3['MG_PV_Amazonia2030'] = data['MG_PV_Amazonia2030']
df3['MG_PV_Chaco2020'] = data['MG_PV_Chaco2020']
df3['MG_PV_Chaco2030'] = data['MG_PV_Chaco2030']
df3['Elevation'] = data['Elevation']
df3['PopStartYear'] = data['PopStartYear']
df3['Y_deg'] = data['Y_deg']