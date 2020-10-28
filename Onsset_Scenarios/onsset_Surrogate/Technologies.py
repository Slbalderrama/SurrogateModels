#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:22:12 2020

@author: balderrama
"""
from onsset import Technology
import pandas as pd

def technology_creation(start_year, end_year, grid_price, specs_data, diesel_price, pv_capital_cost_adjust):
    
    
    Base_To_Peak_Grid  = pd.read_csv('Bolivia/Base_to_Peak_Grid.csv', index_col=0)  
    Base_To_Peak_SA_PV =  pd.read_csv('Bolivia/Base_to_Peak_SA_PV.csv', index_col=0)   
    Base_To_Peak_Hybrid_MicroGrid =  pd.read_csv('Bolivia/Base_to_Peak_SA_Diesel.csv', index_col=0)  
    Base_To_Peak_MG_PV =  pd.read_csv('Bolivia/Base_to_Peak_MG_PV.csv', index_col=0)   
    Base_To_Peak_MG_Diesel =  pd.read_csv('Bolivia/Base_to_Peak_MG_Diesel.csv', index_col=0)   
    technologies = []

    Technology.set_default_values(base_year=start_year, # 2012.0
                                      start_year=start_year, # 2012.0
                                      end_year=end_year, # 2025
                                      discount_rate=0.12)


    grid_calc = Technology(om_of_td_lines=0.02,
                               distribution_losses=float(specs_data.iloc[0]['GridLosses']), # 0.183
                               connection_cost_per_hh=125,
                               base_to_peak_load_ratio= Base_To_Peak_Grid['Base_To_Peak_Ratio'],
                               capacity_factor=1,
                               tech_life=30,
                               grid_capacity_investment=float(specs_data.iloc[0]['GridCapacityInvestmentCost']), # 1722
                               grid_penalty_ratio=1,
                               grid_price=grid_price, # 0.1223
                               name = 'Grid',
                               code = 1)
    
    technologies.append(grid_calc)
    

    
    sa_pv_calc = Technology(base_to_peak_load_ratio = Base_To_Peak_SA_PV['Base_To_Peak_Ratio'],
                                tech_life = 15,
                                om_costs = 0.02,
                                capital_cost={float("inf"): 5070 * pv_capital_cost_adjust, # 5070
                                              0.200: 5780 * pv_capital_cost_adjust,        # 5780
                                              0.100: 7660 * pv_capital_cost_adjust,        # 7660
                                              0.050: 11050 * pv_capital_cost_adjust,       # 11050
                                              0.020: 20000 * pv_capital_cost_adjust        # 20000
                                              },
                                standalone=True,
                                name = 'SA_PV_',
                                code = 3)
    
    technologies.append(sa_pv_calc)
 
    mg_pv_calc = Technology(om_of_td_lines=0.03,
                            distribution_losses=0,
                            connection_cost_per_hh=125,
                            base_to_peak_load_ratio=Base_To_Peak_MG_PV['Base_To_Peak_Ratio'],
                            tech_life=20,
                            om_costs=0.02,
                            capital_cost={float("inf"): 3500},
                            mini_grid=True,
                            name = 'MG_PV_',
                            code = 5)
    
    technologies.append(mg_pv_calc)

    mg_diesel_calc = Technology(om_of_td_lines=0.02,
                                distribution_losses=0,
                                connection_cost_per_hh=125,
                                base_to_peak_load_ratio=Base_To_Peak_MG_Diesel['Base_To_Peak_Ratio'],
                                capacity_factor=0.7,
                                tech_life=20,
                                om_costs=0.02,
                                capital_cost={float("inf"): 1480},
                                mini_grid=True,
                                name = 'MG_Diesel_',
                                code = 4)

    technologies.append(mg_diesel_calc)

    surrogate_model_1 = {'name': 'MG_Hybrid_LowLands_',
                         'path_LCOE': 'Bolivia/Surrogate_Models/LCOE_LowLands.joblib' ,
                         'path_NPC': 'Bolivia/Surrogate_Models/NPC_LowLands.joblib', 
                         'path_Investment' : 'Bolivia/Surrogate_Models/Investment_LowLands.joblib',
                         'path_PV_Capacity': 'Bolivia/Surrogate_Models/PV_LowLands.joblib' ,
                         'path_Genset_Capacity': 'Bolivia/Surrogate_Models/Genset_LowLands.joblib' ,
                         'path_Battery_Capacity': 'Bolivia/Surrogate_Models/Battery_LowLands.joblib',
                         'Variables' : 10,
                         'var_1' : 'Renewable Invesment Cost',
                         'value_1': 1500,
                         'var_2' : 'Battery Unitary Invesment Cost',
                         'value_2': 550,
                         'var_3' : 'Deep of Discharge',
                         'value_3': 0.2,
                         'var_4' : 'Battery Cycles',
                         'value_4': 5500,
                         'var_5':'GenSet Unitary Invesment Cost',
                         'value_5': 1480,
                         'var_6' : 'Generator Efficiency',
                         'value_6': 0.314,
                         'var_7' : 'Low Heating Value',
                         'value_7': 9.89,
                         'var_8' : 'Fuel Cost',
                         'var_9' : 'HouseHolds',
                         'var_10' : 'Renewable Energy Unit Total',
                         'value_10': 'PV total output'}                                   
    
    
    mg_hybrid_calc = Technology(om_of_td_lines=0.02,
                                 distribution_losses=0,
                                 connection_cost_per_hh=125,
                                 base_to_peak_load_ratio=Base_To_Peak_Hybrid_MicroGrid['Base_To_Peak_Ratio'],
                                 tech_life=20,
                                 mini_grid=True,
                                 name = 'MG_Hybrid_LowLands_',
                                 code = 8,
                                 surrogate_model = True,
                                 surrogate_model_data = surrogate_model_1,
                                 tech_life_surrogate = 20)
    
    technologies.append(mg_hybrid_calc)
    
    transportation_cost = []
    

    transportation_cost.append({'diesel_price': diesel_price,
                                'fuel_price': diesel_price,
                                'tech_name' : 'MG_Diesel_',
                                'diesel_truck_consumption': 33.7,
                                'diesel_truck_volume':  15000,
                                'Surrogate_Models': True})       
    
    transportation_cost.append({'diesel_price': diesel_price,
                                'fuel_price': diesel_price,
                                'tech_name' : 'MG_Hybrid_LowLands_',
                                'diesel_truck_consumption': 33.7,
                                'diesel_truck_volume':  15000,
                                'Surrogate_Models': True})    
    
    # Constraints
    
    tech_constraints = []
    
    
    tech_constraints.append({'Type': 'minor',
                             'name': 'MG_Hybrid_LowLands',
                             'Column_name': 'HouseHolds', 
                             'bound'      : 550, 
                             'Years'      :  [2025]                          
                                })
    
    tech_constraints.append({'Type': 'mayor',
                             'name':'MG_Hybrid_LowLands',
                             'Column_name':'HouseHolds', 
                             'bound'      : 50, 
                             'Years'      :  [2025]                          
                                })

    tech_constraints.append({'Type'       : 'minor',
                             'name'       : 'MG_Hybrid_LowLands',
                             'Column_name': 'Elevation', 
                             'bound'      : 800, 
                             'Years'      :  [2025]
                                })
    


    # Constraints
    
    demand_constraints = []
    
    demand_constraints.append({'name': 'LowLands',
                               'path': 'Bolivia/Surrogate_Models/Demand_LowLands.joblib',
                               'constraints': 3,
                               'Type_1': 'minor',
                               'Column_name_1': 'Elevation',
                               'bound_1'      : 800, 
                               'Type_2': 'mayor',
                               'Column_name_2': 'HouseHolds mayor',
                               'bound_2'      : 50, 
                               'Type_3': 'minor',
                               'Column_name_3': 'HouseHolds minor',
                               'bound_3'      : 550, 
                               'Variables' : 1,
                               'Var_1' : 'HouseHolds'
                                })
    

    
    return technologies, transportation_cost, tech_constraints, demand_constraints
    