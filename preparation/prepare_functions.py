# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:20:44 2021

@author: Luca01
"""

import pypsa
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import statistics as stat 
import json
from shapely.geometry import shape, Point
import powerplantmatching as pm



def missing_RES_heuristic(network,file_path="D:/Python/PyPSA/Luca/data/national_generation_capacity_stacked.csv"):
    """
    heuristic to assign RES capacities from country wise data 
    taken and changed from add_electricty script of pypsa-eur (https://github.com/PyPSA/pypsa-eur/blob/master/scripts/add_electricity.py)
    ("missing" because there is another source for RES, which is used in the first PyPSA-Eur workflow)
    
    Parameters
    ----------
    network : TYPE
        DESCRIPTION.
    file_path : TYPE, optional
        DESCRIPTION. The default is "D:/Python/PyPSA/Luca/data/national_generation_capacity_stacked.csv".

    Returns
    -------
    network : TYPE
        DESCRIPTION.

    """
    # check in what countries there is RES capacity already assigned from pypsa-eur more complete data
    c_cap = [gen[0]+gen[1] for gen,p in zip(network.generators.index, network.generators.p_nom) if 'solar' in gen or 'wind' in gen if p!=0]
    c_cap=list(set(c_cap))
    countries = network.buses.country.unique()
    countries = (list(set(countries)-set(c_cap)))
    #to replace 
    #     capacities = (pm.data.Capacity_stats().powerplant.convert_country_to_alpha2()
    #                   [lambda df: df.Energy_Source_Level_2]
    #                   .set_index(['Fueltype', 'Country']).sort_index())
    # in pypsa-eur script add_electricity.py
    capacities = pd.read_csv(file_path)
    capacities = (capacities.query('source == "entsoe SO&AF" & year == 2016')
            .rename(columns={'technology': 'Fueltype'}).rename(columns=str.title)
            .replace(dict(Fueltype={
                  'Bioenergy and other renewable fuels': 'Bioenergy',
                  'Bioenergy and renewable waste': 'Waste',
                  'Coal derivatives': 'Hard Coal',
                  'Differently categorized fossil fuels': 'Other',
                  'Differently categorized renewable energy sources':
                  'Other',
                  'Hard coal': 'Hard Coal',
                  'Mixed fossil fuels': 'Other',
                  'Natural gas': 'Natural Gas',
                  'Other or unspecified energy sources': 'Other',
                  'Tide, wave, and ocean': 'Other'}))
            .dropna(subset=['Energy_Source_Level_2'])
            .pipe(pm.utils.set_column_name,'Entsoe So&Af'))
    #capacities.columns.name = 'Entsoe So&Af' #maybe not right!
    capacities = (capacities[lambda df: df.Energy_Source_Level_2].set_index(['Fueltype', 'Country']).sort_index())
    tech_map = {'Wind': ['onwind', 'offwind-ac', 'offwind-dc'], 'Solar': ['solar']}
    def normed(x): return (x/x.sum()).fillna(0.)
    for ppm_fueltype, techs in tech_map.items():
        #tech_capacities = capacities.loc[ppm_fueltype, 'Capacity']#.reindex(cow, fill_value=0.)
        tech_capacities = capacities.loc[ppm_fueltype,'Capacity'].groupby('Country').agg(Capacity='sum').reindex(countries,fill_value=0.)
    #    tech_i = network.generators.query('carrier in @techs').index
        tech_i = (network.generators.query('carrier in @techs')
                      [network.generators.query('carrier in @techs')
                       .bus.map(network.buses.country).isin(countries)].index)
        network.generators.loc[tech_i, 'p_nom'] = (
                (network.generators_t.p_max_pu[tech_i].mean() * 
                 network.generators.loc[tech_i, 'p_nom_max']) # maximal yearly generation
                 .groupby(network.generators.bus.map(network.buses.country))
                 .transform(lambda s: normed(s) * tech_capacities.at[s.name,'Capacity'])
                 .where(lambda s: s>0.1, 0.))  # only capacities above 100kW
    return network


## 
# rewuired input: network
def add_load_shedding(network,marginal_cost=10000):
    """
    add load shedding to all buses

    Parameters
    ----------
    network : TYPE
    marginal_cost : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    network : TYPE

    """
    for bus in network.buses.index:
        network.add("Generator","{} load_shed".format(bus),
                   bus = bus,
                   p_nom=10000,
                   marginal_cost=marginal_cost)
    return network

def remove_extendability(network):
    """
    remove option to have components extendable

    Parameters
    ----------
    network : TYPE
        DESCRIPTION.

    Returns
    -------
    network : TYPE
        DESCRIPTION.

    """
    network.generators.p_nom_extendable=False
    network.storage_units.p_nom_extendable = False
    network.links.p_nom_extendable=False
    network.lines.s_nom_extendable=False    
    return network

def assign_zone_to_bus(network,path_zonefiles='D:/Python/PyPSA/Luca/zone_files'):
    """
    assigns property to zone to buses based on the location of buses

    Parameters
    ----------
    network : TYPE
        DESCRIPTION.
    path_zonefiles : path to zone files, contains individual geojson files for all zones, that are not single countries; optional
        DESCRIPTION. The default is 'D:/Python/PyPSA/Luca/zone_files'.

    Returns
    -------
    network : TYPE
        DESCRIPTION.

    """
    network.buses['zone'] = np.nan
        # load the geojson files from folder
    for subdir, dirs, files in os.walk(path_zonefiles):
        for file in files:
            path = os.path.join(subdir, file)
            zone_name = os.path.splitext(file)[0]
            with open(path) as f: # open the json files in the folder
                data = json.load(f)
            for feature in data['features']:
                polygon = shape(feature['geometry']) # assign the coordinates to a shape
                for ind, x,y in zip(network.buses.index,network.buses.x, network.buses.y):
                    point = Point(x,y) #make coordinates of buses a point
                    if polygon.contains(point): # check if points are in zone polygon
                        network.buses.zone.loc[ind]=zone_name # if yes assign name to zone column
            
    # for all remaining nodes assign zone as in index name
    for ind,zone in zip(network.buses.index,network.buses.zone):
        if not isinstance(zone, str): # True if zone is not a string i.e. nan, i.e. no zone has been assigned yet
            network.buses.zone.loc[ind] = ind[:2]
    
    #Problem with Denmark:
        # all buses that are assigned to 'DK' should instead be part of 'DK2"
    # same problem with Norway:
        #all NO buses should be assigned to NOs
    for ind,zone in zip(network.buses.index,network.buses.zone):
        if zone=='DK':
            network.buses.zone.loc[ind]='DKe'
        elif zone == 'NO':
            network.buses.zone.loc[ind]='NOs'
    return network

def assign_components_to_zones(network):
    """
    assign loads, generators and storage_units to new zone buses
    replace bus column with former bus name and now with zone name
    loads
    
    Parameters
    ----------
    network : 
   
    Returns
    -------
    network : 
        
    """
    ## assign loads, generators and storage_units to new zone buses
    # replace bus column with former bus name and now with zone name
    #loads
    for ind, load_bus in zip(network.loads.index,network.loads.bus):
        network.loads.bus.loc[ind] = network.buses.zone[load_bus]

    #generators
    for ind, gen_bus in zip(network.generators.index,network.generators.bus):
        network.generators.bus.loc[ind] = network.buses.zone[gen_bus]
    
    #storage units
    for ind, stor_bus in zip(network.storage_units.index,network.storage_units.bus):
        network.storage_units.bus.loc[ind] = network.buses.zone[stor_bus]
    return network



def load_ntc(network,ntc_path='D:/Python/PyPSA/Luca/NTC/NTC_2020_LJ.csv'):
    """
    loads ntc data and clean for those zones not included in the network
    
    Parameters
    ----------
    network : PyPSA-network
        DESCRIPTION.
    ntc_path : path to csv file with NTC data, optional
        DESCRIPTION. The default is 'D:/Python/PyPSA/Luca/NTC/NTC_2020_LJ.csv'.

    Returns
    -------
    ntc : TYPE
        DESCRIPTION.

    """
    ntc = pd.read_csv(ntc_path)
    # split border in from and to
    ntc[['bus0','bus1']] = ntc['borders'].str.split('-',expand=True)
    #ntc.borders
    ntc = ntc.set_index('borders')
    # remove from NTC df all that are not included in PyPSA 
    #remove from ntc all lines connecting zones that dont exist in pypsa-eur network
    for ind,bus0, bus1 in zip(ntc.index,ntc.bus0,ntc.bus1):
        if bus0 not in set(network.buses.zone) or bus1 not in set(network.buses.zone):
            ntc = ntc.drop([ind])
    return ntc


def remove_country_components(network,remove_countries=['AL','BA','ME','RS']):
    """
    remove certain countries' components because the data available is not complete
    
    Parameters
    ----------
    network : pypsa network
        DESCRIPTION.
    remove_countries : list of countries to be removes, optional
        DESCRIPTION. The default is ['AL','BA','ME','RS'].

    Returns
    -------
    network : TYPE
        DESCRIPTION.

    """
    remove_gen = [gen for gen in network.generators.index for country in remove_countries if country in gen]
    remove_gen
    network.mremove("Generator", remove_gen)
    remove_load = [gen for gen in network.loads.index for country in remove_countries if country in gen]
    network.mremove("Load", remove_load)
    remove_storage = [sto for sto in network.storage_units.index for country in remove_countries if country in sto]
    network.mremove("StorageUnit", remove_storage)    
    return network

def display_components(network):
    """
    prints the components of a network
    """
    for c in network.iterate_components(list(network.components.keys())[2:]):
        print("Component '{}' has {} entries".format(c.name,len(c.df)))