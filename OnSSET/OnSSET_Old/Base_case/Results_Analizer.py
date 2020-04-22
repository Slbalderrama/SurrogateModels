#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:05:03 2020

@author: balderrama
"""

import pandas as pd
#%%
data = pd.read_csv('bo-1_0_0_0_0_0_0_0_0_summary.csv')  


Total_Population = data[:7]['2025'].sum()
new_conections = data[7:13]['2025'].sum()