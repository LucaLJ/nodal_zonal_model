# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:46:38 2021

@author: Luca01
"""

# see also jupyter file: run_hydro_year_corridor_03.ipynb

import pypsa
import pandas as pd
import pyomo.environ as pe

# %% Load  network
path_52 = 'D:\\Python\\PyPSA\\Luca\\zonal_nodal_networks\\2018\\reduced_hydro\\zonal_reduced_03_2018.nc'
n52 = pypsa.Network(path_52)

# dictionary that contains all hydro storages ordered by zone
hydro_dict = dict()
for zone in n52.storage_units.bus.unique():
    hydro_dict[zone] = n52.storage_units[n52.storage_units.bus == zone].index


# %% this is to test if changing the efficiency to 1 will make weekly run feasible
# n52.storage_units.efficiency_dispatch=1
# n52.storage_units.efficiency_store=1

# %%
# %% define constraint functions to limit sum of SOC in zones

# constraint:
def hydro_sum_zone(network, snapshots):
    model = network.model

    def constr_sum_zone(model, zone, snapshot):
        # this is the capacity sum parameter
        capacity_sum = sum(
            network.storage_units.max_hours[storage] * network.storage_units.p_nom[storage] for storage in
            hydro_dict[zone])
        # this is the model variable
        soc_sum = sum(model.state_of_charge[storage, snapshot] for storage in hydro_dict[zone])
        return pe.inequality(0.3 * capacity_sum, soc_sum, 1 * capacity_sum)

    model.new_constraint = pe.Constraint(list(hydro_dict), list(snapshots), rule=constr_sum_zone)


# %% alternatively
# %% define constraint function that limits individual storage unit SOC between 30 and 100%

def hydro_soc_individual(network, snapshots):
    model = network.model

    def constr_su(model, storage, snapshot):
        # parameter: individual max capacity of storage units
        cap = network.storage_units.max_hours[storage] * network.storage_units.p_nom[storage]
        # variable: SOC of individual storage units
        soc = model.state_of_charge[storage, snapshot]
        return pe.inequality(0.3 * cap, soc, 1 * cap)

    model.new_constraint = pe.Constraint(list(network.storage_units.index), list(snapshots), rule=constr_su)


# %% Run model for full year
# with sum of soc constraint
#n52.lopf(solver_name='gurobi', extra_functionality=hydro_sum_zone)
# with individual soc constraint
n52.lopf(solver_name='gurobi', extra_functionality=hydro_soc_individual)


# %% Export relevant result i.e. sum of soc in zones

df_zone_soc = pd.DataFrame()
for zone in hydro_dict:
    df_zone_soc[zone] = n52.storage_units_t.state_of_charge[hydro_dict[zone]].sum(axis=1)
# need to add an extra line because I need the initial soc at t=0 which are the same as the final soc
df_zone_soc.loc[0] = df_zone_soc.loc[52]
df_zone_soc.sort_index(inplace=True)
df_zone_soc.to_csv("D:\\Python\\PyPSA\\Luca\\data\\hydro\\2018\\hydro_weekly_sum_zone_04_2018.csv", index=True)
# %% Export results in detail for all individual storage units

df = pd.DataFrame()
df = n52.storage_units_t.state_of_charge

# need to add an extra line because I need the initial soc at t=0 which are the same as the final soc
df.loc[0] = df.loc[52]
df.sort_index(inplace=True)
df.to_csv('D:\\Python\\PyPSA\\Luca\\data\\hydro\\2018\\hydro_weekly_su_individual_const_2018.csv', index=True)

# %% export also network to
# path_save = 'D:\\Python\\PyPSA\\Luca\\jupyter\\hydro\\hydro_model\\annual_network_hydro_solved_03.nc'
# path_save = 'D:\\Python\\PyPSA\\Luca\\jupyter\\hydro\\hydro_model\\annual_network_hydro_solved_eff1_02.nc'
# n52.export_to_netcdf(path_save)

# %% Export results for phs and hydro separately
# dictionary that contains all 'hydro' hydro storages ordered by zone
# hydro_dict_only = dict()
# for zone in n52.storage_units.bus.unique():
#     liste = []
#     for su in n52.storage_units.index:
#         if not su.__contains__('PHS'):
#             if n52.storage_units.bus[su] == zone:
#                 liste.append(su)
#     hydro_dict_only[zone] = liste
#
# # dictionary that contains all "PHS" storages ordered by zone
# phs_dict = dict()
# for zone in n52.storage_units.bus.unique():
#     liste = []
#     for su in n52.storage_units.index:
#         if su.__contains__('PHS'):
#             if n52.storage_units.bus[su] == zone:
#                 liste.append(su)
#     phs_dict[zone] = liste
#
# # export hydro hydro results in detail
# df_zone_soc = pd.DataFrame()
# for zone in hydro_dict_only:
#     for hydro in hydro_dict_only[zone]:
#         df_zone_soc[hydro] = n52.storage_units_t.state_of_charge[hydro]
# # need to add an extra line because I need the initial soc at t=0 which are the same as the final soc
# df_zone_soc.loc[0] = df_zone_soc.loc[52]
# df_zone_soc.sort_index(inplace=True)
# df_zone_soc.to_csv('D:\Python\PyPSA\Luca\data\hydro\hydro_weekly_sum_zone_onlyhydro_su.csv', index=True)

# # export results for hydro hydro
# df_zone_soc = pd.DataFrame()
# for zone in hydro_dict_only:
#     df_zone_soc[zone] = n52.storage_units_t.state_of_charge[hydro_dict_only[zone]].sum(axis=1)
# # need to add an extra line because I need the initial soc at t=0 which are the same as the final soc
# df_zone_soc.loc[0]=df_zone_soc.loc[52]
# df_zone_soc.sort_index(inplace=True)
# df_zone_soc.to_csv('D:\Python\PyPSA\Luca\data\hydro\hydro_weekly_sum_zone_onlyhydro.csv',index=True)

# # export results for phs
# # esport results for hydro hydro
# df_zone_soc = pd.DataFrame()
# for zone in phs_dict:
#     df_zone_soc[zone] = n52.storage_units_t.state_of_charge[phs_dict[zone]].sum(axis=1)
# # need to add an extra line because I need the initial soc at t=0 which are the same as the final soc
# df_zone_soc.loc[0]=df_zone_soc.loc[52]
# df_zone_soc.sort_index(inplace=True)
# df_zone_soc.to_csv('D:\Python\PyPSA\Luca\data\hydro\hydro_weekly_sum_zone_phs.csv',index=True)
