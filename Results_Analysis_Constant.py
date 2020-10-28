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
import matplotlib.patches as mpatches
#%%
# Data load
data = pd.read_excel('Databases/Database_Fix.xls', index_col=0, Header=None)   

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

for i in [300]:
    df = data.loc[data['HouseHolds']==i]
    df.index = range(100)
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
# Gap

name = 'Fuel Cost'
data_1 = data.copy()
data_1 = data_1.sort_values(name, ascending=True)

size = [20,15]
tick_size = 25 
label_size = 25

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax.scatter(data_1['Fuel Cost'], data_1['Gap'], c='k', s=100)
ax.set_xlabel('Fuel Price',size=label_size) 
ax.set_ylabel('%',size=label_size) 
ax2.scatter(data_1['Fuel Cost'], data_1['Renewable Capacity'], c='y',s = 100)
ax2.set_ylabel('Nominal Capacity (kW)',size=label_size) 
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right') 
ax.set_xlim([1.3,1.7])
ax.set_ylim([0.2,1.05])
ax2.set_xlim([1.3,1.7])
ax2.set_ylim([30,60])

NPC_patch = mpatches.Patch(color='k', label= 'Gap')
PV_patch = mpatches.Patch(color='y', label= 'Renewable Capacity')
plt.legend(handles=[NPC_patch, PV_patch],fontsize=20, loc='upper left')

#%%
# Gap

name = 'Fuel Cost'
data_1 = data.copy()
data_1 = data_1.sort_values(name, ascending=True)

size = [20,15]
tick_size = 25 
label_size = 25

fig=plt.figure(figsize=size)
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

mpl.rcParams['xtick.labelsize'] = tick_size 
mpl.rcParams['ytick.labelsize'] = tick_size 

ax.scatter(data_1['Fuel Cost'], data_1['NPC']/1000, c='k', s=100, label='NPC')
ax.set_xlabel('Fuel Price',size=label_size) 
ax.set_ylabel('USD',size=label_size) 

ax2.scatter(data_1['Fuel Cost'], data_1['Renewable Capacity'], c='y',s = 100,
            label='PV Capacity')
ax2.set_ylabel('Nominal Capacity (kW)',size=label_size) 
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right') 
ax.set_xlim([1.3,1.7])
ax2.set_xlim([1.3,1.7])
ax2.set_ylim([30,70]) 


NPC_patch = mpatches.Patch(color='k', label= 'NPC')
PV_patch = mpatches.Patch(color='y', label= 'Renewable Capacity')
plt.legend(handles=[NPC_patch, PV_patch],fontsize=20, loc='upper left')



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