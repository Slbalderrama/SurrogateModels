# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:20:41 2020

@author: Dell
"""

import pandas as pd

data   = pd.read_csv('onsset_classic/Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  
data1  = pd.read_csv('onsset_surrogate/Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  



data   = data.loc[data['Demand_Name2025'] == 'LowLands']
data1  = data1.loc[data1['Demand_Name2025'] == 'LowLands']


data.to_csv('Plot_Data_Classic.csv')
data1.to_csv('Plot_Data_surrogate.csv')

df_hybrid = data1.loc[data1['MinimumOverall2025'] == 'MG_Hybrid_LowLands_2025']
new_connections_hybrid = round(df_hybrid['NewConnections2025'].sum(),0)
print('The number of new connections for the hybrids is ' + str(new_connections_hybrid))

df_grid = data.loc[data['MinimumOverall2025'] == 'Grid2025']
new_connections_grid = round(df_grid['NewConnections2025'].sum(),0)
print('The number of new connections for the Grid is ' + str(new_connections_grid))