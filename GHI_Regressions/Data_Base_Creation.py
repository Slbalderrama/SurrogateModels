#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:30:09 2019

@author: balderrama
"""

import pandas as pd

data = pd.read_excel('Data_Base.xls', index_col=0)

ghi = pd.read_csv('GHI_LowLands.csv', index_col=0)

for i in range(50,520,50):
    for n in range(150):
        name = str(i) + '_'+ str(n)
        print(name)
        data['GHI'] = 



    
    data.to_excel('Data_Base.xls')        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        