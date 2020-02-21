#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:16:49 2019

@author: balderrama
"""

import pandas as pd

#%%
# Data manipulation
data = pd.read_excel('Data_Base_1.xls', index_col=0, Header=None)   

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















