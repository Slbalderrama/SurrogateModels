# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:20:41 2020

@author: Dell
"""

import pandas as pd

data   = pd.read_csv('onsset_1/Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  
data1  = pd.read_csv('onsset_1/Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  
data06 = pd.read_csv('onsset_06/Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)
data02 = pd.read_csv('onsset_02/Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  


data   = data.loc[data['Demand_Name2025'] == 'LowLands']
data1  = data1.loc[data1['MinimumOverall2025'] == 'MG_Hybrid_LowLands2025']
data06 = data06.loc[data06['MinimumOverall2025'] == 'MG_Hybrid_LowLands2025']
data02 = data02.loc[data02['MinimumOverall2025'] == 'MG_Hybrid_LowLands2025']
data_PV = data.loc[data['MinimumOverall2025'] == 'SA_PV_2025']


for i in data1.index:
    
    print(i in data06.index)
    
for j in data06.index:
    
    print(j in data02.index)    
    
for n in data02.index:
    
    print(j in data.index)        
    
    
for i in data.index:
    
    if i  in data1.index:
        
        data.loc[i,'Scenario'] = '1 USD/l'
    
    elif i in data06.index:
        data.loc[i,'Scenario'] = '0.6 USD/l'

    elif i in data02.index:

        data.loc[i,'Scenario'] = '0.2 USD/l'

    else:
    
        data.loc[i,'Scenario'] = 'Other technologies'

    if i in data_PV.index:
        
        data.loc[i,'SA'] = True
    else:
        
        data.loc[i,'SA'] = False

print(data_PV['NewConnections2025'].sum())


test1 = data.loc[data['Scenario']=='1 USD/l']
test06 = data.loc[data['Scenario']=='0.6 USD/l']
test02 = data.loc[data['Scenario']=='0.2 USD/l']

comp = test1.index == data1.index
print(comp.all())

test_06 = pd.DataFrame()

test_06 = test_06.append(pd.DataFrame(test1.index))
test_06 = test_06.append(pd.DataFrame(test06.index))
test_06.index = range(len(test_06))

for i in test_06.index:
    
    print(test_06[0][i] in data06.index)



test_02 = pd.DataFrame()

test_02 = test_02.append(test_06[0])
test_02 = test_02.append(pd.DataFrame(test02.index))
test_02.index = range(len(test_02))


for i in test_02.index:
    
    print(test_02[0][i] in data02.index)



data.to_csv('Plot_Data.csv')


