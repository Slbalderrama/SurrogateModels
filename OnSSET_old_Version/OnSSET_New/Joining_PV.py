# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:35:11 2020

@author: Dell
"""

import pandas as pd

pv = pd.DataFrame(columns=['Energy'])

data = pd.read_csv('PV_Energy_Total.csv', index_col=0)  
data1 = pd.read_csv('PV_Energy_Total_1.csv', index_col=0)

#%%
pv = pv.append(data) 

#%%
data1 = data1.drop([1034]) 
pv = pv.append(data1)

#%%
data2= pd.read_csv('PV_Energy_Total_2.csv', index_col=0)

data2 = data2.drop([5110]) 
pv = pv.append(data2)

#%%
data3= pd.read_csv('PV_Energy_Total_3.csv', index_col=0)

#data3 = data3.drop([5110]) 
pv = pv.append(data3)

#%%

data4= pd.read_csv('PV_Energy_Total_4.csv', index_col=0)

data4 = data4.drop([5786]) 
pv = pv.append(data4)

#%%

data5= pd.read_csv('PV_Energy_Total_5.csv', index_col=0)

#data5 = data5.drop([5786]) 
pv = pv.append(data5)

#%%

data6= pd.read_csv('PV_Energy_Total_6.csv', index_col=0)

#data5 = data5.drop([5786]) 
pv = pv.append(data6)

#%%

data7 = pd.read_csv('PV_Energy_Total_7.csv', index_col=0)

#data7 = data7.drop([5786]) 
pv = pv.append(data7)

#%%

data8 = pd.read_csv('PV_Energy_Total_8.csv', index_col=0)
data8 = data8.drop([12019]) 
pv = pv.append(data8)

#%%

data9 = pd.read_csv('PV_Energy_Total_9.csv', index_col=0)
#data9 = data9.drop([12019]) 
pv = pv.append(data9)
#%%

data10 = pd.read_csv('PV_Energy_Total_10.csv', index_col=0)
pv = pv.append(data10)
#%%
data11 = pd.read_csv('PV_Energy_Total_11.csv', index_col=0)
pv = pv.append(data11)

#%%
data12 = pd.read_csv('PV_Energy_Total_12.csv', index_col=0)
pv = pv.append(data12)
#%%
data13 = pd.read_csv('PV_Energy_Total_13.csv', index_col=0)
pv = pv.append(data13)


#%%

Data = pd.read_csv('Database_new.csv', index_col=None)  



print( len(Data) == len(pv))
comparation = Data.index == pv.index
print(comparation.all())


pv.to_csv('PV_2012.csv')


















