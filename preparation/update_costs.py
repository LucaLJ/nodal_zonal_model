import pypsa
import pandas as pd


#%% Load the cost data
costs = pd.read_csv("D:\\Python\\PyPSA\\Luca\\data\\marginal_cost_2018.csv")
costs = costs.set_index('tech')

#%%
networks = ['nodal', 'zonal']
path_load = 'D:\\Python\\PyPSA\\Luca\\zonal_nodal_networks\\2018\\{}_1024.nc'
path_save = 'D:\\Python\\PyPSA\\Luca\\zonal_nodal_networks\\2018\\{}_1024_costs2018.nc'
for network in networks:
    print("loading network from ", path_load.format(network))
    # Load the base network
    n = pypsa.Network(path_load.format(network))

    # Change all cost data in generators and storage units
    for tech in costs.index:
        for gen in n.generators.index:
            if tech == n.generators.carrier[gen]:
                for parameter in costs:
                    n.generators.loc[gen, parameter] = costs.loc[tech, parameter]
        for stor in n.storage_units.index:
            if tech == n.storage_units.carrier[stor]:
                for parameter in costs:
                    n.storage_units.loc[stor, parameter] = costs.loc[tech, parameter]

    # Save network in new destination
    print("saving network in location ", path_save.format(network))
    n.export_to_netcdf(path_save.format(network))
