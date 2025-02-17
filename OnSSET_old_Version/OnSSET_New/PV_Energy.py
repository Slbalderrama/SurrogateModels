#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:20:00 2020

@author: balderrama
"""
import pandas as pd
import requests
import json
import math as mt
import pvlib
import time


data = pd.read_csv('Database_new.csv', index_col=None)  


def solar_power(args,solar_energy,start,lat):
    
            token = '8694d4573138f2f38c686d75d2a44c4b5d33c443 '
            api_base = 'https://www.renewables.ninja/api/'
            s = requests.session()
            # Send token header with each request
            s.headers = {'Authorization': 'Token ' + token}
            
            url = api_base + 'data/pv'
            r = s.get(url, params=args)
    
            # Parse JSON to get a pandas.DataFrame of data and dict of metadata
            parsed_response = json.loads(r.text)
    
            data = pd.read_json(json.dumps(parsed_response['data']), 
                            orient='index')
            index = data.index.tz_localize('utc').tz_convert('America/La_Paz')
            data.index = index
            if solar_energy == 'Variable':
                start_1 = str(start) + '-01-01  00:00:00'
                end_1 = str(start) + '-12-31 23:00:00'
            else:
                start_1 = str(2012) + '-01-01  00:00:00'
                end_1 = str(2012) + '-12-31 23:00:00'
            
            data = data[start_1:end_1]
            data = data[~((data.index.month == 2) & (data.index.day == 29))]
            data.index = range(1,8761)
           

            Noct = 44.8
            a = (Noct-20)/800
            
            cec_modules = pvlib.pvsystem.retrieve_sam('cecmod')
            cecmodule = cec_modules.Yingli_Energy__China__YL250P_29b #select module
            
            data['diffuse'] = data['irradiance_diffuse']
            data['direct'] = data['irradiance_direct']
            
            data['Radiation'] = data['direct'] + data['diffuse']
            data['Radiation'] = data['Radiation']*1000
            
            data['PV temperature'] = a*data['Radiation'] + data['temperature']
            
            photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
                pvlib.pvsystem.calcparams_desoto(data['Radiation'],
                                     temp_cell=data['PV temperature'],
                                     alpha_sc=cecmodule['alpha_sc'],
                                     a_ref = cecmodule['a_ref'],
                                     I_L_ref = cecmodule['I_L_ref'],
                                     I_o_ref = cecmodule['I_o_ref'],
                                     R_sh_ref = cecmodule['R_sh_ref'],
                                     R_s = cecmodule['R_s'],
                                     EgRef=1.121,
                                     dEgdT=-0.0002677))
            single_diode_out = pvlib.pvsystem.singlediode(photocurrent, saturation_current,
                                      resistance_series, resistance_shunt, nNsVth)
            
            return list(single_diode_out['p_mp']) 


Power_PV = pd.DataFrame()
#%%
for j in data.index:
        print(j)
    
        lat = data['Y_deg'][j]
        lat = round(lat,5)
        lon = data['X_deg'][j]
        lon = round(lon,5)    

        
        start = 2012
        Data = pd.DataFrame()
        Number_Scenarios = 1
        solar_energy = 'Fix'
        for i in range(Number_Scenarios):
            
            
            if solar_energy == 'Variable':
                end = start + 1
                date_from = str(start) + '-01-01'
                date_to = str(end) + '-01-01'
            else:
                date_from = str(2012) + '-01-01'
                date_to = str(2013) + '-01-01'
        
                
            args = {
            'lat': lat,
            'lon': lon,
            'date_from': date_from,
            'date_to': date_to,
            'dataset': 'merra2',
            'capacity': 1.0,
            'system_loss': 0.1,
            'tracking': 0,
            'tilt': 0,
            'azim': 180,
            'format': 'json',
            'raw':"True"}
        
            if solar_energy=='Variable':
                Data[i+1]= solar_power(args,solar_energy,start,lat)
                start = end
            
            if solar_energy == 'Fix' and i == 0:
                Data[i+1]= solar_power(args,solar_energy,start,lat)
        
            else:
                Data[i+1] = Data[i]
                
                
        Data.index = range(1,8761)

        Power_PV.loc[j,'Energy'] = Data.sum()[1]/1000

        Power_PV.to_csv('GHI.csv')
        time.sleep(20)


#%%
        
        

