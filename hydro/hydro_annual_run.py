# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:46:38 2021

@author: Luca01
"""

import pypsa
import pandas as pd
import pyomo.environ as pe

#%%
path_52 = 'D:\\Python\\PyPSA\\Luca\\zonal_nodal_networks\\2018\\reduced_hydro\\zonal_reduced.nc'
n52 = pypsa.Network(path_52)

#%%
# dictionary that contains all hydro storages ordered by zone
hydro_dict = dict()
for zone in n52.storage_units.bus.unique():
    hydro_dict[zone]=n52.storage_units[n52.storage_units.bus==zone].index

def hydro_sum_zone(network,snapshots):
    model = network.model
    def constr_sum_zone(model,zone,snapshot):
        #this is the capacity sum parameter
        capacity_sum = sum(network.storage_units.max_hours[storage] * network.storage_units.p_nom[storage] for storage in hydro_dict[zone])
        # this is the model variable
        soc_sum = sum(model.state_of_charge[storage,snapshot] for storage in hydro_dict[zone])
        return pe.inequality(0.3*capacity_sum,soc_sum,1*capacity_sum)
    model.new_constraint = pe.Constraint(list(hydro_dict),list(snapshots),rule=constr_sum_zone)

#%% Run model for full year
n52.lopf(solver_name='gurobi',extra_functionality=hydro_sum_zone)

#%% Export relevant result

df_zone_soc = pd.DataFrame()
for zone in hydro_dict:
    df_zone_soc[zone] = n52.storage_units_t.state_of_charge[hydro_dict[zone]].sum(axis=1)
df_zone_soc.to_csv('D:\Python\PyPSA\Luca\data\hydro\hydro_weekly_sum_zone.csv',index=True)
