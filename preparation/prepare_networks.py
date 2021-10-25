# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:37:34 2021

@author: Luca01
"""

import pypsa
import numpy as np
import pandas as pd
from prepare_functions import *
import os
os.chdir('D:\\Python\\PyPSA\\Luca\\spyder\\network_preperation')
def prepare_both(network):
    '''
    Function that performs first network preparation steps which are the same for both zonal and nodal network
    Parameters
    ----------
    network : TYPE
        DESCRIPTION.

    Returns
    -------
    network : TYPE
        DESCRIPTION.

    '''
    # assign missing RES capacities through heuristic
    network = missing_RES_heuristic(network)
    # add load shedding generators to all buses
    network = add_load_shedding(network)
    # remove extendability functionality 
    network = remove_extendability(network)
    # assign zone property
    network = assign_zone_to_bus(network)
    # remove components of countries for which no complete data is available
    network = remove_country_components(network)
    return network

def prepare_nodal(network):
    """
    performs steps to generate nodal network from initial 1024 node pypsa-eur network
    uses functions defined in prepare_functions.py 

    Parameters
    ----------
    network : TYPE
        DESCRIPTION.

    Returns
    -------
    network : TYPE
        DESCRIPTION.

    """
    ### add new node ITCO (for comparability to zonal network)
    co_row = network.buses.iloc[-1,:]
    co_row.name="ITCO"
    co_row.x =9 
    co_row.y = 42
    co_row.country = 'IT'
    co_row.zone = 'ITCO'
    network.buses = network.buses.append(co_row)
    
    ### Replace 1 link ITsar-ITcn with 2 links ITcn-ITCO and ITsar-ITCO (with values from NTC)
    # identify link to be replaced
    for ind0, (bus0,bus1) in enumerate(zip(network.links.bus0,network.links.bus1)):
        if 'ITsar' in network.buses.zone[bus0] and "ITcn" in network.buses.zone[bus1]:
            new_link = network.links.iloc[ind0,:]
            # ITsar-ITCO (i.e. bus0-ITCO)
            new_link.bus1='ITCO'
            new_link.p_nom = 350
            new_link.p_min_pu = -300/350
            network.links.iloc[ind0,:]=new_link
            # ITcn-ITCO
            new_link = network.links.iloc[ind0,:]
            new_link.name= '5631+5'
            new_link.bus0 = bus1
            new_link.bus1='ITCO'
            new_link.p_nom = 300
            new_link.p_min_pu = -1
            network.links = network.links.append(new_link)
            new_link = network.links.iloc[ind0,:]
    return network
    

def prepare_zonal(network):
    """
    performs steps to generate zonal network from initial 1024 node pypsa-eur network
    uses functions defined in prepare_functions.py    

    Parameters
    ----------
    network : TYPE
        DESCRIPTION.

    Returns
    -------
    network : TYPE
        DESCRIPTION.

    """
    # assign components (load, generators, storage_units) to zones
    network = assign_components_to_zones(network)

    # delete internal lines and rename to and from buses to zone buses
    # first rename from and to buses to zones
    for ind, frombus in zip(network.lines.index,network.lines.bus0):
        network.lines.bus0.loc[ind] = network.buses.zone[frombus]
    for ind, tobus in zip(network.lines.index,network.lines.bus1):
        network.lines.bus1.loc[ind] = network.buses.zone[tobus]
    # delete lines that are not connecting different zones
    index_drop = network.lines[network.lines.bus0==network.lines.bus1].index
    network.lines.drop(index_drop,inplace=True)
    
    # do the same for links
    for ind, frombus in zip(network.links.index,network.links.bus0):
        network.links.bus0.loc[ind] = network.buses.zone[frombus]
    for ind, tobus in zip(network.links.index,network.links.bus1):
        network.links.bus1.loc[ind] = network.buses.zone[tobus]
    # delete links that are not connecting zones
    index_drop = network.links[network.links.bus0==network.links.bus1].index
    network.links.drop(index_drop,inplace=True)
    
    ###create new bus dataframe 
    # get unique list of zones
    zone_list = pd.unique(network.buses.zone)
    # create new empty dataframe for new buses
    # assign column names as same as the old one
    bus_df = pd.DataFrame(columns=network.buses.columns)
    
    for zone in zone_list:
        x = []
        y = []
        for ind,bus_zone in zip(network.buses.index,network.buses.zone):
            if bus_zone==zone:
                x.append(network.buses.x.loc[ind])
                y.append(network.buses.y.loc[ind])
                new_zone_row = network.buses.loc[ind,:]# get last row for zone and copy properties to new bus for zone
                # okay, because properties are all the same except we need one slack bus, i.e. AL
        # get mean x and y for the respective zone (only fails for Croatia)
        x_new = stat.mean(x)
        y_new = stat.mean(y)
        new_zone_row.x = x_new
        new_zone_row.y = y_new
        bus_df = bus_df.append(new_zone_row) # append to new bus dataframe
    
    # Add ITCO to buses
    zone_list = np.append(zone_list,'ITCO')
    new_zone_row = network.buses.iloc[-1,:]
    new_zone_row.x = 9
    new_zone_row.y = 42
    new_zone_row.country = 'IT'
    new_zone_row.zone = 'ITCO'
    bus_df = bus_df.append(new_zone_row)
    # change the index of new bus data frame to zone list
    bus_df.index = zone_list
    bus_df.index.names = ['name']# change also index name to 'name'
    network.buses = bus_df # reassign new bus index to network dataframe
    
    # load ntc data
    ntc = load_ntc(network)
    
    # drop all lines and links
    for link in network.links.index:
        network.remove("Link",link)  
    for line in network.lines.index: 
        network.remove("Line",line)
    
    ### Add links for all lines in NTC
    for ind in ntc.index:
        if ntc.NTC_2020[ind]==0:
            network.add("Link", ind,
                             bus0 = ntc.bus0.loc[ind],
                             bus1 = ntc.bus1.loc[ind],
                             p_nom = ntc.NTC_2020[ind],                     
                             p_min_pu=0                      
                             )
        else:
            network.add("Link", ind,
                             bus0 = ntc.bus0.loc[ind],
                             bus1 = ntc.bus1.loc[ind],
                             p_nom = ntc.NTC_2020[ind],                                        
                             p_min_pu = - ntc.back[ind]/ntc.NTC_2020[ind]
                             )
    return network


def main():
    path = "D:/Python/PyPSA/networks/LB/2018_new/elec_s_1024.nc"
    
    network = pypsa.Network(path)
    print('preparing nodal network')    
    network_node_prepare = prepare_both(network)
    network_node = prepare_nodal(network_node_prepare)
    
    path_save_nodal = "D:/Python/PyPSA/Luca/zonal_nodal_networks/2018/nodal_1024.nc"
    
    print('saving nodal networks in {}'.format(path_save_nodal))    
    network_node.export_to_netcdf(path_save_nodal)
    
    network = pypsa.Network(path)
    print('preparing zonal network')
    network_zone_prepare = prepare_both(network)
    network_zone = prepare_zonal(network_zone_prepare)
    
    path_save_zonal = "D:/Python/PyPSA/Luca/zonal_nodal_networks/2018/zonal_1024.nc"
    
    print('saving zonal networks in {}'.format(path_save_zonal))  
    network_zone.export_to_netcdf(path_save_zonal)

if __name__ == "__main__":
    main()    
    