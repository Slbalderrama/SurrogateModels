import os
from runner import scenario
import pandas as pd
from joblib import load
import numpy as np

    # 'Database_lower_ElecPopCalib.csv'
    # 'Database_new_1.csv'
    
'test only one constraint for the amazonia and highlands'
    
specs_path = os.path.join('Bolivia', 'specs_paper_new.xlsx')
calibrated_csv_path = os.path.join('Bolivia', 'Database_new_3.csv')
results_folder = os.path.join('Bolivia')
summary_folder	= os.path.join('Bolivia')
  
scenario(specs_path, calibrated_csv_path, results_folder, summary_folder)
    
data = pd.read_csv('Bolivia/bo-1-0_0_0_0_0_0.csv', index_col=0)  
data1 = pd.read_csv('Bolivia/Results_Compare.csv', index_col=0)  


#%%
df = pd.DataFrame()

df['EnergyPerSettlement2020'] = data['EnergyPerSettlement2020']
df['EnergyPerSettlement2020 1'] = data1['EnergyPerSettlement2020']
df['EnergyPerSettlement2030'] = data['EnergyPerSettlement2030']
df['EnergyPerSettlement2030 1'] = data1['EnergyPerSettlement2030']
#df['TotalEnergyPerCell2020'] = data['TotalEnergyPerCell2020']
#df['TotalEnergyPerCell2020 1'] = data1['TotalEnergyPerCell2020']
df['TotalEnergyPerCell'] = data['TotalEnergyPerCell']
df['TotalEnergyPerCell 1'] = data1['TotalEnergyPerCell']
df['Demand_Name'] = data['Demand_Name']
df['Elevation'] = data['Elevation']
df['Y_deg'] = data['Y_deg']
df['Pop2012'] = data['Pop2012']

#%%
booleans = pd.DataFrame()

df1 = df.loc[df['Demand_Name'] == 'Amazonia']

booleans['Elevation'] = list(data1['Elevation'] < 800)
booleans['Y_deg'] = list(data1['Y_deg'] >= -17)
booleans['clasification'] = booleans.all(axis=1)  

df11 = pd.DataFrame()
df11 = booleans.loc[booleans['clasification']==True]

comparation = df11.index == df1.index
print(comparation.all())

#%%
df3 = df.loc[df['Demand_Name'] == 'Highlands']
booleans2 = pd.DataFrame()
booleans2['clasification'] = list(data1['Elevation'] >= 800)


df31 = pd.DataFrame()
df31 = booleans2.loc[booleans2['clasification']==True]

comparation3 = df31.index == df3.index
print(comparation3.all())

#%%

df4 = df.loc[df['Demand_Name'] == 'Chaco']
booleans4 = pd.DataFrame()
booleans4['Elevation'] = list(data1['Elevation'] < 800)
booleans4['Y_deg'] = list(data1['Y_deg'] < -17)
booleans4['clasification'] = booleans4.all(axis=1)  

df41 = pd.DataFrame()
df41 = booleans.loc[booleans4['clasification']==True]

comparation4 = df41.index == df4.index
print(comparation4.all())


#%%

path =  'Bolivia/Surrogate_Models/Demand/demand_regression_Amazonia.joblib'
demand = load(path)

X1 = pd.DataFrame()
X2 = pd.DataFrame()
X3 = pd.DataFrame()
X4 = pd.DataFrame()


X1['HouseHoldsEnergyPerCell'] = data1['NewConnections'  + "{}".format(2020)]/data1['NumPeoplePerHH']
X2['HouseHoldsTotalEnergy2020'] = data1['Pop'  + "{}".format(2020)]/data1['NumPeoplePerHH']
X3['HouseHoldsEnergyPerCell'] = data1['NewConnections'  + "{}".format(2030)]/data1['NumPeoplePerHH']
X4['HouseHoldsTotalEnergy2030'] = data1['Pop'  + "{}".format(2030)]/data1['NumPeoplePerHH']

demand_new_2020 = pd.DataFrame(demand.predict(X1))
demand_total_2020 = pd.DataFrame(demand.predict(X2))
demand_new_2030 = pd.DataFrame(demand.predict(X3))
demand_total_2030 = pd.DataFrame(demand.predict(X4))

#17233
# small slacks are needed bacouse we are using the result data frame with weird rounding after the change of times 
# at the end of the onsset script

comp1 = pd.DataFrame()

comp1['Demand Comparation new 2020'] = abs(df1['EnergyPerSettlement2020'] - demand_new_2020[0][df11.index]) < 1
#comp1['TotalEnergyPerCell2020 2020'] = abs(df1['TotalEnergyPerCell2020'] - demand_total_2020[0][df11.index]) < 2
comp1['EnergyPerSettlement2030 2030'] = abs(df1['EnergyPerSettlement2030'] - demand_new_2030[0][df11.index]) < 1
comp1['TotalEnergyPerCell 2030'] = abs(df1['TotalEnergyPerCell'] - demand_total_2030[0][df11.index]) < 1


print(comp1['Demand Comparation new 2020'].all())
#print(comp1['TotalEnergyPerCell2020 2020'].all())
print(comp1['EnergyPerSettlement2030 2030'].all())
print(comp1['TotalEnergyPerCell 2030'].all())


#%%

df2 = df.loc[df['Demand_Name'] == np.nan]
comp2 = pd.DataFrame()
comp2['Demand Comparation new 2020']   = df2['EnergyPerSettlement2020'] == df2['EnergyPerSettlement2020 1']
#comp2['Demand Comparation total 2020'] = df2['TotalEnergyPerCell2020']  == df2['TotalEnergyPerCell2020 1']
comp2['Demand Comparation new 2030']   = df2['EnergyPerSettlement2030'] == df2['EnergyPerSettlement2030 1']
comp2['Demand Comparation total 2030'] = df2['TotalEnergyPerCell']      == df2['TotalEnergyPerCell 1']

print(comp2['Demand Comparation new 2020'].all())
#print(comp2['Demand Comparation total 2020'].all())
print(comp2['Demand Comparation new 2030'].all())
print(comp2['Demand Comparation total 2030'].all())

#%%

path =  'Bolivia/Surrogate_Models/Demand/demand_regression_HighLands.joblib'
demand3 = load(path)

X11 = pd.DataFrame()
X22 = pd.DataFrame()
X33 = pd.DataFrame()
X44 = pd.DataFrame()


X11['HouseHoldsEnergyPerCell'] = data1['NewConnections'  + "{}".format(2020)]/data1['NumPeoplePerHH']
X22['HouseHoldsTotalEnergy2020'] = data1['Pop'  + "{}".format(2020)]/data1['NumPeoplePerHH']
X33['HouseHoldsEnergyPerCell'] = data1['NewConnections'  + "{}".format(2030)]/data1['NumPeoplePerHH']
X44['HouseHoldsTotalEnergy2030'] = data1['Pop'  + "{}".format(2030)]/data1['NumPeoplePerHH']

demand_new_2020_3 = pd.DataFrame(demand3.predict(X11))
demand_total_2020_3 = pd.DataFrame(demand3.predict(X22))
demand_new_2030_3 = pd.DataFrame(demand3.predict(X33))
demand_total_2030_3 = pd.DataFrame(demand3.predict(X44))

#17233
# small slacks are needed bacouse we are using the result data frame with weird rounding after the change of times 
# at the end of the onsset script

comp3 = pd.DataFrame()

comp3['Demand Comparation new 2020'] = abs(df3['EnergyPerSettlement2020'] - demand_new_2020_3[0][df31.index]) < 12
#comp3['TotalEnergyPerCell2020 2020'] = abs(df3['TotalEnergyPerCell2020'] - demand_total_2020_3[0][df31.index]) < 37
comp3['EnergyPerSettlement2030 2030'] = abs(df3['EnergyPerSettlement2030'] - demand_new_2030_3[0][df31.index]) < 12
comp3['TotalEnergyPerCell 2030'] = abs(df3['TotalEnergyPerCell'] - demand_total_2030_3[0][df31.index]) < 37

print(comp3['Demand Comparation new 2020'].all())
#print(comp3['TotalEnergyPerCell2020 2020'].all())
print(comp3['EnergyPerSettlement2030 2030'].all())
print(comp3['TotalEnergyPerCell 2030'].all())


#%%

path =  'Bolivia/Surrogate_Models/Demand/demand_regression_Chaco.joblib'
demand4 = load(path)

X41 = pd.DataFrame()
X42 = pd.DataFrame()
X43 = pd.DataFrame()
X444 = pd.DataFrame()


X41['HouseHoldsEnergyPerCell'] = data1['NewConnections'  + "{}".format(2020)]/data1['NumPeoplePerHH']
X42['HouseHoldsTotalEnergy2020'] = data1['Pop'  + "{}".format(2020)]/data1['NumPeoplePerHH']
X43['HouseHoldsEnergyPerCell'] = data1['NewConnections'  + "{}".format(2030)]/data1['NumPeoplePerHH']
X444['HouseHoldsTotalEnergy2030'] = data1['Pop'  + "{}".format(2030)]/data1['NumPeoplePerHH']

demand_new_2020_4 = pd.DataFrame(demand4.predict(X41))
demand_total_2020_4 = pd.DataFrame(demand4.predict(X42))
demand_new_2030_4 = pd.DataFrame(demand4.predict(X43))
demand_total_2030_4 = pd.DataFrame(demand4.predict(X444))

#17233
# small slacks are needed bacouse we are using the result data frame with weird rounding after the change of times 
# at the end of the onsset script

comp4 = pd.DataFrame()

comp4['Demand Comparation new 2020'] = abs(df4['EnergyPerSettlement2020'] - demand_new_2020_4[0][df41.index]) < 12
#comp4['TotalEnergyPerCell2020 2020'] = abs(df4['TotalEnergyPerCell2020'] - demand_total_2020_4[0][df41.index]) < 37
comp4['EnergyPerSettlement2030 2030'] = abs(df4['EnergyPerSettlement2030'] - demand_new_2030_4[0][df41.index]) < 12
comp4['TotalEnergyPerCell 2030'] = abs(df4['TotalEnergyPerCell'] - demand_total_2030_4[0][df41.index]) < 25

print(comp4['Demand Comparation new 2020'].all())
#print(comp4['TotalEnergyPerCell2020 2020'].all())
print(comp4['EnergyPerSettlement2030 2030'].all())
print(comp4['TotalEnergyPerCell 2030'].all())








