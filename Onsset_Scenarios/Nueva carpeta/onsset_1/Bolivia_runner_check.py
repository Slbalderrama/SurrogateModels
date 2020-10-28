
from runner import scenario
import pandas as pd
import os
from joblib import load


    # 'Database_lower_ElecPopCalib.csv'
    # 'Database_new_1.csv'
specs_path = os.path.join('Bolivia', 'specs_paper_new.xlsx')
calibrated_csv_path = os.path.join('Bolivia', 'Database_new_1.csv')
results_folder = os.path.join('Bolivia')
summary_folder	= os.path.join('Bolivia')
  
scenario(specs_path, calibrated_csv_path, results_folder, summary_folder)


#%%    
data = pd.read_csv('Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  
summary =  pd.read_csv('Bolivia/bo-1-0_0_0_0_0_0_summary.csv', index_col=0) 
indpendant_variables = pd.read_csv('Bolivia/Independent_Variables2025.csv', index_col=0) 


df = pd.DataFrame()
df['Demand_Name2025'] = data['Demand_Name2025']
df['Elevation'] = data['Elevation']
#df['SA_DieselFuelCost2025'] = data['SA_DieselFuelCost2025']
df['MG_Hybrid_LowLandsFuelCost2025'] = data['MG_Hybrid_LowLandsFuelCost2025']
df['SA_PV_2025'] = data['SA_PV_2025']
#df['SA_Diesel2025'] = data['SA_Diesel2025']
df['MG_Hybrid_LowLands2025'] = data['MG_Hybrid_LowLands2025']
df['MG_Hybrid_LowLands2025'] = data['MG_Hybrid_LowLands2025']
df['Minimum_Tech_Off_grid2025'] = data['Minimum_Tech_Off_grid2025']
df['Grid2025'] = data['Grid2025']
df['MinimumOverall2025'] = data['MinimumOverall2025']
df['HouseHolds'] = indpendant_variables['HouseHolds']
df['Pop2025'] = data['Pop2025']
df['EnergyPerSettlement2025'] = data['EnergyPerSettlement2025']
df['TotalEnergyPerCell'] = data['TotalEnergyPerCell']



df1 = df.loc[df['Demand_Name2025'] == 'LowLands']

df2 = df.loc[df['Elevation']<800]
df2 = df2.loc[df['HouseHolds']<550]
df2 = df2.loc[df['HouseHolds']>=50]

comp = df1.index == df2.index
print(comp.all())

#%%

path =  'Bolivia/Surrogate_Models/Demand_LowLands.joblib'
demand = load(path)

X1 = pd.DataFrame()
X2 = pd.DataFrame()

X1['HouseHoldsEnergyPerCell'] = data['NewConnections'  + "{}".format(2025)]/data['NumPeoplePerHH']
X2['HouseHoldsTotalEnergy2025'] = data['Pop'  + "{}".format(2025)]/data['NumPeoplePerHH']

demand_new_2025 = pd.DataFrame(demand.predict(X1))
demand_total_2025 = pd.DataFrame(demand.predict(X2))

comp1 = pd.DataFrame()
comp1['Demand Comparation new 2020'] = abs(df2['EnergyPerSettlement2025'] - demand_new_2025[0][df1.index]) < 1
comp1['TotalEnergyPerCell2020 2020'] = abs(df2['TotalEnergyPerCell'] - demand_total_2025[0][df1.index]) < 2

print(comp1['Demand Comparation new 2020'].all())
print(comp1['TotalEnergyPerCell2020 2020'].all())

#%%

lcoe_1 = pd.read_csv('Bolivia/lcoe_microgrid_grid.csv', index_col=0)  
path_npc =  'Bolivia/Surrogate_Models/NPC_LowLands.joblib'

NPC = load(path_npc)

X = pd.DataFrame(index=range(len(lcoe_1)))
X['Renewable Invesment Cost'] = 1500
X['Battery Unitary Invesment Cost'] = 550
X['Deep of Discharge'] = 0.2
X['Battery Cycles'] = 5500
X['GenSet Unitary Invesment Cost'] = 1480
X['Generator Efficiency'] =  0.314
X['Low Heating Value'] =  9.89
X['Fuel Cost'] =  data['MG_Hybrid_LowLandsFuelCost2025']
X['HouseHolds'] = X1['HouseHoldsEnergyPerCell'] 
X['Renewable Energy Unit Total'] = data['PV total output']

df3 = pd.DataFrame(NPC.predict(X))

e = 0.12
y=20

a = e*(1+e)**y
b = (1+e)**y -1
    
CFR  = a/b

df3['Expected Demand'] = demand_new_2025[0]/CFR
df3['LCOE 1'] = df3[0]/df3['Expected Demand']
df3['LCOE 2'] = lcoe_1['0']
df3 = df3.loc[df['Demand_Name2025'] == 'LowLands']

from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(df3['LCOE 1'],df3['LCOE 2']))
print(mean_squared_error(df3['LCOE 1'], df3['LCOE 2']))


#%%

Base_To_Peak_Grid  = pd.read_csv('Bolivia/Base_to_Peak_Grid.csv', index_col=0)  
Base_To_Peak_SA_PV =  pd.read_csv('Bolivia/Base_to_Peak_SA_PV.csv', index_col=0)  
#Base_To_Peak_SA_Diesel =  pd.read_csv('Bolivia/Base_to_Peak_SA_Diesel.csv', index_col=0)  
Base_To_Peak_Hybrid_MicroGrid =  pd.read_csv('Bolivia/Base_to_Peak_Hybrid_MicroGrid.csv', index_col=0)  


Base_To_Peak_Grid_1 = Base_To_Peak_Grid.loc[Base_To_Peak_Grid['Base_To_Peak_Ratio'] != 0.52958]
Base_To_Peak_SA_PV_1 = Base_To_Peak_SA_PV.loc[Base_To_Peak_SA_PV['Base_To_Peak_Ratio'] != 0.9]
Base_To_Peak_Hybrid_MicroGrid_1 = Base_To_Peak_Hybrid_MicroGrid.loc[Base_To_Peak_Hybrid_MicroGrid['Base_To_Peak_Ratio'] != 0.000001]
#Base_To_Peak_SA_Diesel_1 = Base_To_Peak_SA_Diesel.loc[Base_To_Peak_SA_Diesel['Base_To_Peak_Ratio'] != 0.000001]

comparation1 = df1.index == Base_To_Peak_Grid_1.index
print(comparation1.all())


comparation2 = df1.index == Base_To_Peak_SA_PV_1.index
print(comparation2.all())

comparation3 = df1.index == Base_To_Peak_Hybrid_MicroGrid_1.index
print(comparation3.all())













