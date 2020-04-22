#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:16:49 2019

@author: balderrama
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pylab as pylab
#%%
# Data load
data = pd.read_excel('Data_Base_Fix.xls', index_col=0, Header=None)   

#%%
# Data results

mean = data.mean()
Datos = pd.DataFrame()

Datos.loc['NPC (thousands USD)', 'Mean'] = mean['NPC'] 
Datos.loc['LCOE (USD/kWh)', 'Mean'] = mean['LCOE'] 
Datos.loc['PV nominal capacity (kW)', 'Mean'] = mean['Renewable Capacity'] 
Datos.loc['Battery nominal capacity (kWh)', 'Mean'] = mean['Battery Capacity'] 
Datos.loc['Renewable energy penetration (%)', 'Mean'] = mean['Renewable Penetration']*100 
Datos.loc['Battery usage (%)', 'Mean'] = mean['Battery Usage Percentage'] 
Datos.loc['Energy curtail (%)', 'Mean'] = mean['Curtailment Percentage'] 

Max = data.max()

Datos.loc['NPC (thousands USD)', 'Max'] = Max['NPC'] 
Datos.loc['LCOE (USD/kWh)', 'Max'] = Max['LCOE'] 
Datos.loc['PV nominal capacity (kW)', 'Max'] = Max['Renewable Capacity'] 
Datos.loc['Battery nominal capacity (kWh)', 'Max'] = Max['Battery Capacity'] 
Datos.loc['Renewable energy penetration (%)', 'Max'] = Max['Renewable Penetration']*100 
Datos.loc['Battery usage (%)', 'Max'] = Max['Battery Usage Percentage'] 
Datos.loc['Energy curtail (%)', 'Max'] = Max['Curtailment Percentage'] 

Min = data.min()

Datos.loc['NPC (thousands USD)', 'Min'] = Min['NPC'] 
Datos.loc['LCOE (USD/kWh)', 'Min'] = Min['LCOE'] 
Datos.loc['PV nominal capacity (kW)', 'Min'] = Min['Renewable Capacity'] 
Datos.loc['Battery nominal capacity (kWh)', 'Min'] = Min['Battery Capacity'] 
Datos.loc['Renewable energy penetration (%)', 'Min'] = Min['Renewable Penetration']*100 
Datos.loc['Battery usage (%)', 'Min'] = Min['Battery Usage Percentage'] 
Datos.loc['Energy curtail (%)', 'Min'] = Min['Curtailment Percentage'] 


std = data.std()

Datos.loc['NPC (thousands USD)', 'Std'] = std['NPC'] 
Datos.loc['LCOE (USD/kWh)', 'Std'] = std['LCOE'] 
Datos.loc['PV nominal capacity (kW)', 'Std'] = std['Renewable Capacity'] 
Datos.loc['Battery nominal capacity (kWh)', 'Std'] = std['Battery Capacity'] 
Datos.loc['Renewable energy penetration (%)', 'Std'] = std['Renewable Penetration']*100 
Datos.loc['Battery usage (%)', 'Std'] = std['Battery Usage Percentage'] 
Datos.loc['Energy curtail (%)', 'Std'] = std['Curtailment Percentage'] 

Datos.to_latex('table')


#%%


data_1 = MinMaxScaler().fit_transform(data)
data_1 = pd.DataFrame(data_1, columns=data.columns)

data_2 = pd.DataFrame()
for i in data.columns:
    foo = data_1[i]
    foo = foo.sort_values( ascending=False)
    data_2[i] = foo.values
    
index_LDC = []
for i in range(len(data_2)):
    index_LDC.append((i+1)/float(len(data_2))*100)
    
data_2.index = index_LDC    


size = [20,15]
label_size = 25
tick_size = 25 

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

#ax.plot(index_LDC, data_2['NPC'], c='b')
#ax.plot(index_LDC, data_2['LCOE'], c='k')
ax.plot(index_LDC, data_2['Renewable Capacity'], c='y')
ax.plot(index_LDC, data_2['Battery Capacity'], c='g')
ax.plot(index_LDC, data_2['Renewable Penetration'], c='r')
ax.plot(index_LDC, data_2['Battery Usage Percentage'], c='c')
ax.plot(index_LDC, data_2['Curtailment Percentage'], c='m')

# limits
ax.set_xlim([0,100])
ax.set_ylim([0,1])
# labels
ax.set_xlabel('%',size=label_size) 
#ax.set_ylabel('HouseHolds',size=label_size) 
#NPC = mlines.Line2D([], [], color='b',label='NPC')
#LCOE = mlines.Line2D([], [], color='k',label='LCOE')
Battery_Capacity = mlines.Line2D([], [], color='g',label='Battery nominal capacity')
PV_Capacity = mlines.Line2D([], [], color='y',label='PV nominal capacity')
Renewable_Penetration = mlines.Line2D([], [], color='r',label='Renewable energy penetration')
Battery_Usage = mlines.Line2D([], [], color='c',label='Battery usage')
Energy_Curtailment = mlines.Line2D([], [], color='m',label='Energy curtail')

plt.legend(handles=[
#        NPC, 
#        LCOE, 
        PV_Capacity,
        Battery_Capacity, 
        Renewable_Penetration,
                   Battery_Usage, Energy_Curtailment
 ], bbox_to_anchor=(1, 1),fontsize = 20)

plt.savefig('Duration_Curve_Results.png', bbox_inches='tight')    
plt.show()   

#%%
# duration curve of NPC and LCOE
data_3 = pd.DataFrame()
for i in ['NPC', 'LCOE']:
    foo = data[i]
    foo = foo.sort_values( ascending=False)
    data_3[i] = foo.values

index_LDC = []
for i in range(len(data_3)):
    index_LDC.append((i+1)/float(len(data_3))*100)
    
data_3.index = index_LDC    


size = [20,15]
label_size = 25
tick_size = 25 

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax.plot(index_LDC, data_3['NPC']/1000, c='b')
ax2.plot(index_LDC, data_3['LCOE'], c='k')

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right') 
# limits
ax.set_xlim([0,100])
ax.set_ylim([0,1500])
ax2.set_xlim([0,100])
ax2.set_ylim([0,0.8])
# labels
ax.set_xlabel('%',size=label_size) 
ax.set_ylabel('NPC (Thousand USD)',size=label_size) 
ax2.set_ylabel('LCOE (kWh/USD)',size=label_size) 

NPC = mlines.Line2D([], [], color='b',label='NPC')
LCOE = mlines.Line2D([], [], color='k',label='LCOE')


plt.legend(handles=[ NPC, LCOE ], bbox_to_anchor=(1, 1),fontsize = 20)

plt.savefig('Duration_Curve_Costos.png', bbox_inches='tight')    
plt.show()   

#%%
# Box plots NPC and LCOE
BoxPlot_NPC = pd.DataFrame()
BoxPlot_LCOE = pd.DataFrame()

for i in range(50,570,50):
    df = data.loc[data['HouseHolds']==i]
    df.index = range(150)
    BoxPlot_NPC[i] = df['NPC']/1000
    BoxPlot_LCOE[i] = df['LCOE']
        
size = [20,15]
label_size = 25
tick_size = 25 
mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 
ax = BoxPlot_LCOE.boxplot(figsize=size)
ax.set_xlabel('HouseHolds',size=label_size) 
ax.set_ylabel('LCOE (USD/kWh)',size=label_size) 



ax =BoxPlot_NPC.boxplot(figsize=size)
ax.set_xlabel('HouseHolds',size=label_size) 
ax.set_ylabel('NPC (Thousand USD)',size=label_size) 

#%%


data_1 = pd.DataFrame()
for i in data.columns:
    foo = data[i]
    foo = foo.sort_values( ascending=False)
    data_1[i] = foo.values
    
index_LDC = []
for i in range(len(data_1)):
    index_LDC.append((i+1)/float(len(data_1))*100)
    
data_1.index = index_LDC    


size = [20,15]
label_size = 25
tick_size = 25 

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax.plot(index_LDC, data_1['Renewable Capacity'], c='y')
ax.plot(index_LDC, data_1['Battery Capacity'], c='g')

ax2.plot(index_LDC, data_1['Renewable Penetration']*100, c='r')
ax2.plot(index_LDC, data_1['Battery Usage Percentage'], c='c')
ax2.plot(index_LDC, data_1['Curtailment Percentage'], c='m')


ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right') 
# limits
ax.set_xlim([0,100])
ax.set_ylim([0,1000])


ax2.set_xlim([0,100])
ax2.set_ylim([0,100])


# labels
ax.set_xlabel('%',size=label_size) 
ax.set_ylabel('Nominal Capacities',size=label_size) 
ax2.set_ylabel('%',size=label_size) 

#NPC = mlines.Line2D([], [], color='b',label='NPC')
#LCOE = mlines.Line2D([], [], color='k',label='LCOE')
Battery_Capacity = mlines.Line2D([], [], color='g',label='Battery nominal capacity (kWh)')
PV_Capacity = mlines.Line2D([], [], color='y',label='PV nominal capacity (kW)')
Renewable_Penetration = mlines.Line2D([], [], color='r',label='Renewable energy penetration')
Battery_Usage = mlines.Line2D([], [], color='c',label='Battery usage')
Energy_Curtailment = mlines.Line2D([], [], color='m',label='Energy curtail')

plt.legend(handles=[
#        NPC, 
#        LCOE, 
        PV_Capacity,
        Battery_Capacity, 
        Renewable_Penetration,
                   Battery_Usage, Energy_Curtailment
 ], bbox_to_anchor=(1, 1),fontsize = 20)

plt.savefig('Duration_Curve_Capacities.png', bbox_inches='tight')    
plt.show()   

#%%


name = 'Renewable Penetration'
data_1 = data.copy()
data_1 = data_1.sort_values(name, ascending=False)

    
index_LDC = []
for i in range(len(data_1)):
    index_LDC.append((i+1)/float(len(data_1))*100)
    
data_1.index = index_LDC    


size = [20,15]
label_size = 25
tick_size = 25 

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax.plot(index_LDC, data_1['Renewable Capacity'], c='y')
ax.plot(index_LDC, data_1['Battery Capacity'], c='g')

ax2.plot(index_LDC, data_1['Renewable Penetration']*100, c='r')
ax2.plot(index_LDC, data_1['Battery Usage Percentage'], c='k')
#ax2.plot(index_LDC, data_1['Curtailment Percentage'], c='m')


ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right') 
# limits
ax.set_xlim([0,100])
ax.set_ylim([0,1000])


ax2.set_xlim([0,100])
ax2.set_ylim([0,100])


# labels
ax.set_xlabel('%',size=label_size) 
ax.set_ylabel('Nominal Capacities',size=label_size) 
ax2.set_ylabel('%',size=label_size) 


ax.set_xlabel('%',size=label_size) 
ax.set_ylabel('Nominal Capacities',size=label_size) 
ax2.set_ylabel('%',size=label_size) 

#NPC = mlines.Line2D([], [], color='b',label='NPC')
#LCOE = mlines.Line2D([], [], color='k',label='LCOE')
Battery_Capacity = mlines.Line2D([], [], color='g',label='Battery nominal capacity (kWh)')
PV_Capacity = mlines.Line2D([], [], color='y',label='PV nominal capacity (kW)')
Renewable_Penetration = mlines.Line2D([], [], color='r',label='Renewable energy penetration')
Battery_Usage = mlines.Line2D([], [], color='k',label='Battery usage')
#Energy_Curtailment = mlines.Line2D([], [], color='m',label='Energy curtail')

plt.legend(handles=[
#        NPC, 
#        LCOE, 
        PV_Capacity,
        Battery_Capacity, 
        Renewable_Penetration,
                   Battery_Usage
#                   , Energy_Curtailment
 ], bbox_to_anchor=(1, 1),fontsize = 20)

plt.savefig('Duration_Curve_Capacities.png', bbox_inches='tight')    
plt.show()   



#%%
# Capacities
name = 'Fuel Cost'
data_1 = data.copy()
data_1 = data_1.sort_values(name, ascending=True)

size = [20,15]
tick_size = 25 

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
#ax2=fig.add_subplot(111, label="2", frame_on=False)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax.plot(data_1['Fuel Cost'], data_1['Renewable Capacity'], c='y')
ax.plot(data_1['Fuel Cost'], data_1['Battery Capacity'], c='g')
ax.set_xlabel('Fuel Price') 
ax.set_ylabel('Nominal Capacities') 
#ax2.plot(data_1['Fuel Cost'], data_1['Gap'], c='k')

#ax2.yaxis.tick_right()
#ax2.yaxis.set_label_position('right') 

plt.legend()
ax.set_ylim([0,150])

ax2.set_ylim([0,1.5])


#%%
# MOneys
name = 'Fuel Cost'
data_1 = data.copy()
data_1 = data_1.sort_values(name, ascending=True)
data_1['Invesment Cost'] =  data_1['Renewable Invesment Cost'] + data_1['Battery Invesment Cost'] + data_1['Generator Invesment Cost']  

size = [20,15]
tick_size = 25 

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
#ax2=fig.add_subplot(111, label="2", frame_on=False)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax.plot(data_1['Fuel Cost'], data_1['NPC']/1000, c='r', marker='o')
ax.plot(data_1['Fuel Cost'], data_1['Operation Cost']/1000, c='y', marker='o')
ax.plot(data_1['Fuel Cost'], data_1['Invesment Cost']/1000, c='g', marker='o')


#ax.plot(data_1['Fuel Cost'], -data_1['Lower Bound']/1000, c='y')
#ax.plot(data_1['Fuel Cost'], -data_1['Upper Bound']/1000, c='k')

#ax2.plot(data_1['Fuel Cost'], data_1['LCOE'], c='g')
#ax2.plot(data_1['Fuel Cost'], data_1['Gap'], c='k', marker='o')
#ax2.plot(data_1['Fuel Cost'], data_1['Curtailment Percentage']/100, c='k')

#ax2.yaxis.tick_right()
#ax2.yaxis.set_label_position('right') 

plt.legend()
ax.set_ylim([0,550])

ax2.set_ylim([0,1.5])

#%%

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern,DotProduct
from sklearn.utils import shuffle
data_1 = pd.read_excel('Data_Base.xls', index_col=0, Header=None)  

y_NPC = pd.DataFrame()
target= 'NPC' #  'Renewable Capacity' 'Renewable Penetration'
y_NPC[target] = data_1[target]

X_NPC = pd.DataFrame()
X_NPC['Renewable Invesment Cost'] = data_1['Renewable Unitary Invesment Cost']   
X_NPC['Battery Unitary Invesment Cost'] = data_1['Battery Unitary Invesment Cost']
X_NPC['Deep of Discharge'] = data_1['Deep of Discharge']
X_NPC['Battery Cycles'] = data_1['Battery Cycles']
X_NPC['GenSet Unitary Invesment Cost'] = data_1['GenSet Unitary Invesment Cost']
X_NPC['Generator Efficiency'] = data_1['Generator Efficiency']
X_NPC['Low Heating Value'] = data_1['Low Heating Value']
X_NPC['Fuel Cost'] = data_1['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X_NPC['HouseHolds'] = data_1['HouseHolds']
X_NPC['Renewable Energy Unit Total'] = data_1['Renewable Energy Unit Total']
#X['Max Demand'] = data['Max Demand']
#X['Y'] = data['Y']

name = 'Fuel Cost'
data_2 = data.copy()
data_2 = data_2.sort_values(name, ascending=True)

X_NPC_1 = pd.DataFrame()
X_NPC_1['Renewable Invesment Cost'] = data_2['Renewable Unitary Invesment Cost']   
X_NPC_1['Battery Unitary Invesment Cost'] = data_2['Battery Unitary Invesment Cost']
X_NPC_1['Deep of Discharge'] = data_2['Deep of Discharge']
X_NPC_1['Battery Cycles'] = data_2['Battery Cycles']
X_NPC_1['GenSet Unitary Invesment Cost'] = data_2['GenSet Unitary Invesment Cost']
X_NPC_1['Generator Efficiency'] = data_2['Generator Efficiency']
X_NPC_1['Low Heating Value'] = data_2['Low Heating Value']
X_NPC_1['Fuel Cost'] = data_2['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X_NPC_1['HouseHolds'] = data_2['HouseHolds']
X_NPC_1['Renewable Energy Unit Total'] = data_2['Renewable Energy Unit Total']


# gp

l = [6.92062438e+02, 3.31543469e+02, 3.28884416e-01, 8.23503004e+03,
        1.35915431e+03, 8.95198703e-02, 3.47678961e+00, 4.12690057e-01,
        1.57775284e+02, 1.47131806e+02] # Not normalize

kernel =  RBF(l)

gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0, optimizer = 'fmin_l_bfgs_b')
gp_NPC = gp.fit(X_NPC,y_NPC)
print(gp_NPC.score(X_NPC,y_NPC))

y_gp_NPC, std_NPC = gp.predict(X_NPC_1, return_std=True)

# lm 

lm = linear_model.LinearRegression(fit_intercept=True)

lm_NPC = lm.fit(X_NPC,y_NPC)
print(lm_NPC.score(X_NPC,y_NPC))

lm_NPC = lm.fit(X_NPC,y_NPC)
y_lm_NPC = lm_NPC.predict(X_NPC_1)


# plot NPC
fig=plt.figure(figsize=size)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax1=plt.plot(data_2['Fuel Cost'], y_gp_NPC/1000, c='r', label='GPR')
ax2=plt.plot(data_2['Fuel Cost'], y_lm_NPC/1000, c='g', label='MVLR')
ax3_NPC = plt.scatter(data_2['Fuel Cost'], data_2['NPC']/1000, c='y', marker='o', label='Computed NPC')

#ax1.fill_between(data_2['Fuel Cost'], y_gp_NPC[:, 0]/1000 - std_NPC[:]/1000, 
#                y_gp_NPC[:, 0]/1000 + std_NPC[:]/1000, color='darkorange',
#                 alpha=0.2)
pylab.ylim([0,550])
GPR_NPC_line = mlines.Line2D([], [], color='r', label='NPC with GPR')
lm_NPC_line = mlines.Line2D([], [], color='g', label='NPC with MVLR')

plt.legend(handles=[GPR_NPC_line,lm_NPC_line,(ax3_NPC)])




#%%
#LCOE

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern,DotProduct
from sklearn.utils import shuffle


data_1 = pd.read_excel('Data_Base.xls', index_col=0, Header=None)  

y = pd.DataFrame()
target= 'LCOE' #  'Renewable Capacity' 'Renewable Penetration'
y[target] = data_1[target]

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data_1['Renewable Unitary Invesment Cost']   
X['Battery Unitary Invesment Cost'] = data_1['Battery Unitary Invesment Cost']
X['Deep of Discharge'] = data_1['Deep of Discharge']
X['Battery Cycles'] = data_1['Battery Cycles']
X['GenSet Unitary Invesment Cost'] = data_1['GenSet Unitary Invesment Cost']
X['Generator Efficiency'] = data_1['Generator Efficiency']
X['Low Heating Value'] = data_1['Low Heating Value']
X['Fuel Cost'] = data_1['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X['HouseHolds'] = data_1['HouseHolds']
X['Renewable Energy Unit Total'] = data_1['Renewable Energy Unit Total']
#X['Max Demand'] = data['Max Demand']
#X['Y'] = data['Y']

name = 'Fuel Cost'
data_2 = data.copy()
data_2 = data_2.sort_values(name, ascending=True)

X_1 = pd.DataFrame()
X_1['Renewable Invesment Cost'] = data_2['Renewable Unitary Invesment Cost']   
X_1['Battery Unitary Invesment Cost'] = data_2['Battery Unitary Invesment Cost']
X_1['Deep of Discharge'] = data_2['Deep of Discharge']
X_1['Battery Cycles'] = data_2['Battery Cycles']
X_1['GenSet Unitary Invesment Cost'] = data_2['GenSet Unitary Invesment Cost']
X_1['Generator Efficiency'] = data_2['Generator Efficiency']
X_1['Low Heating Value'] = data_2['Low Heating Value']
X_1['Fuel Cost'] = data_2['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X_1['HouseHolds'] = data_2['HouseHolds']
X_1['Renewable Energy Unit Total'] = data_2['Renewable Energy Unit Total']


# gp

l = [5.31988464e+03, 4.52553288e+02, 7.99862755e-01, 1.59213927e+04,
        1.56852094e+04, 1.07986737e-01, 4.15088047e+00, 7.11928485e-01,
        3.77518124e+03, 5.67702975e+02]# Not normalize




kernel =  RBF(l)

gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0, optimizer = 'fmin_l_bfgs_b')
gp_LCOE = gp.fit(X,y)
print(gp_LCOE.score(X,y))

y_gp_LCOE, std_LCOE = gp.predict(X_1, return_std=True)

# lm 

lm = linear_model.LinearRegression(fit_intercept=True)

lm_LCOE = lm.fit(X, y)
print(lm_LCOE.score(X, y))

lm_LCOE = lm.fit(X,y)
y_lm_LCOE = lm_LCOE.predict(X_1)


# plot LCOE
fig=plt.figure(figsize=size)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax1=plt.plot(data_2['Fuel Cost'], y_gp_LCOE, c='r', label='LCOE with GPR')
ax2=plt.plot(data_2['Fuel Cost'], y_lm_LCOE, c='g', label='LCOE with MVLR')
ax3_LCOE=plt.scatter(data_2['Fuel Cost'], data_2['LCOE'], c='y', marker='o', label='LCOE Computed')

#ax1.fill_between(data_2['Fuel Cost'], y_gp_NPC[:, 0]/1000 - std_NPC[:]/1000, 
#                y_gp_NPC[:, 0]/1000 + std_NPC[:]/1000, color='darkorange',
#                 alpha=0.2)
pylab.ylim([0,0.8])
GPR_LCOE_line = mlines.Line2D([], [], color='r', label='LCOE with GPR')
lm_LCOE_line = mlines.Line2D([], [], color='g', label='LCOE with MVLR')

plt.legend(handles=[GPR_LCOE_line,lm_LCOE_line,(ax3_LCOE)])




#%%
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern,DotProduct
from sklearn.utils import shuffle


data_1 = pd.read_excel('Data_Base_Fix.xls', index_col=0, Header=None)  

y = pd.DataFrame()
target= 'Renewable Capacity' #  'Renewable Capacity' 'Renewable Penetration'
y[target] = data_1[target]

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data_1['Renewable Unitary Invesment Cost']   
X['Battery Unitary Invesment Cost'] = data_1['Battery Unitary Invesment Cost']
X['Deep of Discharge'] = data_1['Deep of Discharge']
X['Battery Cycles'] = data_1['Battery Cycles']
X['GenSet Unitary Invesment Cost'] = data_1['GenSet Unitary Invesment Cost']
X['Generator Efficiency'] = data_1['Generator Efficiency']
X['Low Heating Value'] = data_1['Low Heating Value']
X['Fuel Cost'] = data_1['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X['HouseHolds'] = data_1['HouseHolds']
X['Renewable Energy Unit Total'] = data_1['Renewable Energy Unit Total']
#X['Max Demand'] = data['Max Demand']
#X['Y'] = data['Y']

name = 'Fuel Cost'
data_2 = data.copy()
data_2 = data_2.sort_values(name, ascending=True)

X_1 = pd.DataFrame()
X_1['Renewable Invesment Cost'] = data_2['Renewable Unitary Invesment Cost']   
X_1['Battery Unitary Invesment Cost'] = data_2['Battery Unitary Invesment Cost']
X_1['Deep of Discharge'] = data_2['Deep of Discharge']
X_1['Battery Cycles'] = data_2['Battery Cycles']
X_1['GenSet Unitary Invesment Cost'] = data_2['GenSet Unitary Invesment Cost']
X_1['Generator Efficiency'] = data_2['Generator Efficiency']
X_1['Low Heating Value'] = data_2['Low Heating Value']
X_1['Fuel Cost'] = data_2['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X_1['HouseHolds'] = data_2['HouseHolds']
X_1['Renewable Energy Unit Total'] = data_2['Renewable Energy Unit Total']


# gp


l1 =  [1.15e+03, 910, 0.0439, 1.51e+03, 1.56e+04, 4.55, 0.0021, 588, 0.000378, 7.33e+04]
l2 =  [1.29e+03, 329, 0.567, 9.11e+03, 1e+05, 0.116, 4.58, 0.406, 278, 133]

kernel =  RBF(l1) + RBF(l2)

gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0, optimizer = 'fmin_l_bfgs_b')
gp_PV = gp.fit(X,y)
print(gp_PV.score(X,y))

y_gp_PV, std_PV = gp_PV.predict(X_1, return_std=True)


l3 = [7.33831816e+02, 1.54608986e+02, 2.84720913e-01, 5.72084983e+03,
        1.11819934e+03, 6.55008283e-02, 1.85465473e+00, 2.45924395e-01,
        1.24936218e+02, 7.51695193e+01]

kernel1 =  RBF(l3)
gp1 = GaussianProcessRegressor(kernel=kernel1,n_restarts_optimizer=0, optimizer = 'fmin_l_bfgs_b')
gp_PV1 = gp1.fit(X,y)

y_gp_PV_1, std_PV_1 = gp_PV1.predict(X_1, return_std=True)


l4 = [1,1,1,1,1,1,1,1,1,1]
l5 = [1,1,1,1,1,1,1,1,1,1]

kernel2 =  RBF(l4) + RBF(l5) 
gp2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=0, optimizer = 'fmin_l_bfgs_b')
gp_PV2 = gp2.fit(X_1, data_2['Renewable Capacity'])

y_gp_PV_2, std_PV_2 = gp_PV2.predict(X_1, return_std=True)



# lm 

lm = linear_model.LinearRegression(fit_intercept=True)

lm_PV = lm.fit(X, y)
print(lm_PV.score(X, y))

lm_PV = lm.fit(X,y)
y_lm_PV = lm_PV.predict(X_1)


# plot NPC
fig=plt.figure(figsize=size)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax1=plt.plot(data_2['Fuel Cost'], y_gp_PV, c='r', label='PV with GPR')
ax2=plt.plot(data_2['Fuel Cost'], y_lm_PV, c='g', label='PV with MVLR')
ax3_PV=plt.scatter(data_2['Fuel Cost'], data_2['Renewable Capacity'], c='y', marker='o', label='Computed PV Capacity')
#ax4=plt.plot(data_2['Fuel Cost'], y_gp_PV_1, c='b', label='PV with GPR with two SQ kernels')
#ax5=plt.plot(data_2['Fuel Cost'], y_gp_PV_2, c='b', label='PV with GPR with varaiable fuel cost')

#ax1.fill_between(data_2['Fuel Cost'], y_gp_NPC[:, 0]/1000 - std_NPC[:]/1000, 
#                y_gp_NPC[:, 0]/1000 + std_NPC[:]/1000, color='darkorange',
#                 alpha=0.2)
pylab.ylim([0,100])
GPR_PV_line = mlines.Line2D([], [], color='r', label='PV capacity with GPR')
lm_PV_line = mlines.Line2D([], [], color='g', label='PV capacity with MVLR')

plt.legend(handles=[GPR_PV_line,lm_PV_line,(ax3_PV)])



#%%
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern,DotProduct
from sklearn.utils import shuffle


data_1 = pd.read_excel('Data_Base_Fix.xls', index_col=0, Header=None)  

y = pd.DataFrame()
target= 'Battery Capacity' #  'Renewable Capacity' 'Renewable Penetration'
y[target] = data_1[target]

X = pd.DataFrame()
X['Renewable Invesment Cost'] = data_1['Renewable Unitary Invesment Cost']   
X['Battery Unitary Invesment Cost'] = data_1['Battery Unitary Invesment Cost']
X['Deep of Discharge'] = data_1['Deep of Discharge']
X['Battery Cycles'] = data_1['Battery Cycles']
X['GenSet Unitary Invesment Cost'] = data_1['GenSet Unitary Invesment Cost']
X['Generator Efficiency'] = data_1['Generator Efficiency']
X['Low Heating Value'] = data_1['Low Heating Value']
X['Fuel Cost'] = data_1['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X['HouseHolds'] = data_1['HouseHolds']
X['Renewable Energy Unit Total'] = data_1['Renewable Energy Unit Total']
#X['Max Demand'] = data['Max Demand']
#X['Y'] = data['Y']

name = 'Fuel Cost'
data_2 = data.copy()
data_2 = data_2.sort_values(name, ascending=True)

X_1 = pd.DataFrame()
X_1['Renewable Invesment Cost'] = data_2['Renewable Unitary Invesment Cost']   
X_1['Battery Unitary Invesment Cost'] = data_2['Battery Unitary Invesment Cost']
X_1['Deep of Discharge'] = data_2['Deep of Discharge']
X_1['Battery Cycles'] = data_2['Battery Cycles']
X_1['GenSet Unitary Invesment Cost'] = data_2['GenSet Unitary Invesment Cost']
X_1['Generator Efficiency'] = data_2['Generator Efficiency']
X_1['Low Heating Value'] = data_2['Low Heating Value']
X_1['Fuel Cost'] = data_2['Fuel Cost']
#X['Generator Nominal capacity'] = data['Generator Nominal capacity'] 
X_1['HouseHolds'] = data_2['HouseHolds']
X_1['Renewable Energy Unit Total'] = data_2['Renewable Energy Unit Total']


# gp


l1 = [2.41e+04, 0.000502, 5.94, 38.3, 2.5e+04, 0.105, 0.00155, 0.000217, 2.69e+03, 7.48e+03]
l2 = [1.37e+03, 197, 0.389, 7.8e+03, 1e+05, 0.0842, 3.67, 0.352, 261, 73.5]

kernel =  RBF(l1) + RBF(l2)

gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0, optimizer = 'fmin_l_bfgs_b')
gp_bat = gp.fit(X,y)
print(gp_bat.score(X,y))

y_gp_bat, std_bat = gp_bat.predict(X_1, return_std=True)


l3 = [7.56318564e+02, 1.46651732e+02, 2.55953469e-01, 5.26397193e+03,
        1.18120445e+03, 6.25914897e-02, 2.11440252e+00, 1.83576909e-01,
        1.57487327e+02, 4.11697835e+01]

kernel1 =  RBF(l3)
gp1 = GaussianProcessRegressor(kernel=kernel1,n_restarts_optimizer=0, optimizer = 'fmin_l_bfgs_b')
gp_bat1 = gp1.fit(X,y)

y_gp_bat_1, std_PV_1 = gp_bat1.predict(X_1, return_std=True)


l4 = [1,1,1,1,1,1,1,1,1,1]
l5 = [1,1,1,1,1,1,1,1,1,1]

kernel2 =  RBF(l4) + RBF(l5) 
gp2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=100, optimizer = 'fmin_l_bfgs_b')
gp_bat2 = gp2.fit(X_1, data_2['Battery Capacity'])

y_gp_bat_2, std_PV_2 = gp_bat2.predict(X_1, return_std=True)



# lm 

lm = linear_model.LinearRegression(fit_intercept=True)

lm_bat = lm.fit(X, y)
print(lm_bat.score(X, y))

lm_bat = lm.fit(X,y)
y_lm_bat = lm_bat.predict(X_1)


# plot NPC
fig=plt.figure(figsize=size)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax1=plt.plot(data_2['Fuel Cost'], y_gp_bat, c='r', label='PV with GPR')
ax2=plt.plot(data_2['Fuel Cost'], y_lm_bat, c='g', label='PV with MVLR')
ax3_bat=plt.scatter(data_2['Fuel Cost'], data_2['Battery Capacity'], c='y', marker='o', label='Computed Batterry Capacity')
#ax4=plt.plot(data_2['Fuel Cost'], y_gp_bat_1, c='b', label='PV with GPR with two SQ kernels')
#ax5=plt.plot(data_2['Fuel Cost'], y_gp_bat_2, c='b', label='PV with GPR with varaiable fuel cost')

#ax1.fill_between(data_2['Fuel Cost'], y_gp_NPC[:, 0]/1000 - std_NPC[:]/1000, 
#                y_gp_NPC[:, 0]/1000 + std_NPC[:]/1000, color='darkorange',
#                 alpha=0.2)
pylab.ylim([0,380])
GPR_bat_line = mlines.Line2D([], [], color='r', label='Battery with GPR')
lm_bat_line = mlines.Line2D([], [], color='g', label='Battery with MVLR')

plt.legend(handles=[GPR_NPC_line,lm_NPC_line,(ax3_bat)])


#%%

fig = plt.figure(figsize=(40,30))
size = [40,40]
label_size = 25
tick_size = 25 

ax_NPC = fig.add_subplot(221)

# NPC plot
ax_NPC.plot(data_2['Fuel Cost'], y_gp_NPC/1000, c='r', label='NPC with GPR')
ax_NPC.plot(data_2['Fuel Cost'], y_lm_NPC/1000, c='g', label='NPC with MVLR')
ax_NPC.scatter(data_2['Fuel Cost'], data_2['NPC']/1000, c='y', marker='o', label='Computed NPC')
ax_NPC.set_ylim([100,550])
ax_NPC.set_xlim([0.1,2])
ax_NPC.set_xlabel('Fuel Cost (USD/l)', size=label_size)
ax_NPC.set_ylabel('NPC (thousand USD)', size=label_size)
plt.legend(handles=[GPR_NPC_line,lm_NPC_line,(ax3_NPC)])

# LCOE plot
ax2 = fig.add_subplot(222)
ax2.plot(data_2['Fuel Cost'], y_gp_LCOE, c='r', label='LCOE with GPR')
ax2.plot(data_2['Fuel Cost'], y_lm_LCOE, c='g', label='LCOE with MVLR')
ax2.scatter(data_2['Fuel Cost'], data_2['LCOE'], c='y', marker='o', label='LCOE Computed')
ax2.set_ylim([0.1,0.6])
ax_NPC.set_xlim([0.1,2])
ax2.set_xlabel('Fuel Cost (USD/l)', size=label_size)
ax2.set_ylabel('LCOE (USD/kWh)', size=label_size)
plt.legend(handles=[GPR_LCOE_line,lm_LCOE_line,(ax3_LCOE)])

# PV plot

ax3 = fig.add_subplot(223)
ax3.plot(data_2['Fuel Cost'], y_gp_PV, c='r', label='PV with GPR')
ax3.plot(data_2['Fuel Cost'], y_lm_PV, c='g', label='PV with MVLR')
ax3.scatter(data_2['Fuel Cost'], data_2['Renewable Capacity'], c='y', marker='o', label='Computed PV Capacity')
ax3.set_ylim([0,100])
ax_NPC.set_xlim([0.1,2])
ax3.set_xlabel('Fuel Cost (USD/l)', size=label_size)
ax3.set_ylabel('PV (kW)', size=label_size)
plt.legend(handles=[GPR_PV_line,lm_PV_line,(ax3_PV)])

# Bat plot

ax4 = fig.add_subplot(224)
ax4.plot(data_2['Fuel Cost'], y_gp_bat, c='r', label='PV with GPR')
ax4.plot(data_2['Fuel Cost'], y_lm_bat, c='g', label='PV with MVLR')
ax4.scatter(data_2['Fuel Cost'], data_2['Battery Capacity'], c='y', marker='o', label='Computed Batterry Capacity')
ax4.set_xlim([0.1,2])
ax4.set_ylim([0,380])
ax4.set_xlabel('Fuel Cost (USD/l)', size=label_size)
ax4.set_ylabel('Battery Capacity (kWh)', size=label_size)
plt.legend(handles=[GPR_NPC_line,lm_NPC_line,(ax3_bat)])

plt.subplots_adjust(hspace= 0.3)