# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:23:28 2021

@author: Luca01
"""
import pandas as pd
import pypsa

def simple_function(x):
    print(2*x)

## load ntc data and clean for those zones not included in the network
def load_ntc(ntc_path,network):
    ntc = pd.read_csv(ntc_path)
    # split border in from and to
    ntc[['bus0','bus1']] = ntc['borders'].str.split('-',expand=True)
    ntc.borders
    ntc = ntc.set_index('borders')
    # remove from NTC df all that are not included in PyPSA 
    #remove from ntc all lines connecting zones that dont exist in pypsa-eur network
    for ind,bus0, bus1 in zip(ntc.index,ntc.bus0,ntc.bus1):
        if bus0 not in set(network.buses.zone) or bus1 not in set(network.buses.zone):
            print(ind,bus0,bus1)
            ntc = ntc.drop([ind])
    return ntc


# get dictionary of lines (& links) that are assigned to a particular NTC
# dictionary because for each NTC: there is a list of tuples; every tuple includes the line/link of the network that is assigned to NTC with the correct sign +1 or -1
# works also for cross-border flows: instead of ntc input needs to be network_zone.links
def lines_ntc(ntc,network,line_link):
    # assign lines in nodal network to respective NTCs with correct signs
    ntc_lines = dict()
    if line_link=="line":
        ind_buses= zip(network.lines.index,network.lines.bus0,network.lines.bus1)
    elif line_link=="link":
        ind_buses= zip(network.links.index,network.links.bus0,network.links.bus1)
    #else:
        #error define line or link keyword
    list_lines = [(ind,network.buses.zone[bus_from],network.buses.zone[bus_to]) for ind,bus_from,bus_to in ind_buses if network.buses.zone[bus_from]!=network.buses.zone[bus_to]]
    for ntc_name, ntc_bus0,ntc_bus1 in zip(ntc.index,ntc.bus0,ntc.bus1):
        line_list = []
        for line,line_bus0,line_bus1 in list_lines:
            if line_bus0==ntc_bus0 and line_bus1==ntc_bus1:
                line_list.append((line,1))
            if line_bus0==ntc_bus1 and line_bus1==ntc_bus0:
                line_list.append((line,-1))
        ntc_lines[ntc_name]=line_list
    return ntc_lines


