#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:57:19 2020

@author: balderrama
"""
import pandas as pd


Demand = pd. DataFrame()

for i in range(50, 570,20):
    
    Village = 'village_' + str(i)
    Energy_Demand = pd.read_excel('Example/Demand.xls',sheet_name=Village
                                  ,index_col=0,Header=None)
    
    
    Village[i] = Energy_Demand[1]
    
    
#%%
    

from joblib import load

#path = 'Regressions/NPC.joblib'
#NPC = load(path)
#
#X = pd.DataFrame(index = MicroGrids.index)
#
#
#X['Renewable Invesment Cost'] = 1300
#X['Battery Invesment Cost'] =   400
#X['Deep of Discharge'] = 0.5
#X['Battery Cycles'] = 5500
#X['Genset Invesment Cost'] = 1480 
#X['Generator Efficiency'] = 0.31
#X['Low Heating Value'] = 9.9
#X['Fuel Cost'] = MicroGrids['Diesel Price']
#X['HouseHolds'] = MicroGrids['HouseHolds']  
#X['Renewable Energy Unit Total'] = 450 
#
#
#Net = pd.DataFrame(NPC.predict(X), columns=['NPC 1'], index=MicroGrids.index)
#
#Net['NPC 2'] = MicroGrids['NPC_Hybrid_Power']
#
##858