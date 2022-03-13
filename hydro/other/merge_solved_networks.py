
import os
import sys
import pandas as pd
import pypsa
import logging


def main():
    log_file = 'log_netw_merge.log'
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    log = open(log_file, 'a')
    sys.stout = log

    directory = '/datafs1/home/jansen/model/hydro/analysis/52_st_168h_closed_NEW/penalty/33/networks/'
    networks = [os.path.join(directory, path) for path in os.listdir(directory)]

    # load network where to save all
    # makes sense to store in one that is already partly solved, because then dataframes have the right structure
    path0 = '/datafs1/home/jansen/model/hydro/analysis/52_st_168h_closed_NEW/penalty/33/networks/nodal_step_0.nc'
    n0 = pypsa.Network(path0)

    # output variables to be saved
    # buses_t
    bus_vars = ['v_mag_pu', 'v_ang', 'p', 'marginal_price']
    # loads_t
    load_vars = ['p']
    # generators_t
    generator_vars = ['p']
    # storage_units_t
    storage_vars = ['p', 'state_of_charge', 'spill']
    # lines_t
    line_vars = ['p0', 'p1', 'mu_lower', 'mu_upper']
    # links_t
    link_vars = ['p0', 'p1', 'mu_lower', 'mu_upper']

    # loop through networks
    for path in networks:
        n = pypsa.Network(path)
        # get snapshots for which it was solved
        sn = n.storage_units_t.state_of_charge.index[n.storage_units_t.state_of_charge.iloc[:, 0].notnull()]
        for var in bus_vars:
            n0.buses_t[var].loc[sn, :] = n.buses_t[var].loc[sn, :]
        for var in load_vars:
            n0.loads_t[var].loc[sn, :] = n.loads_t[var].loc[sn, :]
        for var in generator_vars:
            n0.generators_t[var].loc[sn, :] = n.generators_t[var].loc[sn, :]
        for var in storage_vars:
            n0.storage_units_t[var].loc[sn, :] = n.storage_units_t[var].loc[sn, :]
        for var in line_vars:
            n0.lines_t[var].loc[sn, :] = n.lines_t[var].loc[sn, :]
        for var in link_vars:
            n0.links_t[var].loc[sn, :] = n.links_t[var].loc[sn, :]


    # export merged network
    path_export = '/datafs1/home/jansen/model/hydro/analysis/52_st_168h_closed_NEW/penalty/33/network_merged.nc'
    logging.info(f'exporting merged network to {path_export}')
    n0.export_to_netcdf(path_export)

if __name__ == "__main__":
    main()