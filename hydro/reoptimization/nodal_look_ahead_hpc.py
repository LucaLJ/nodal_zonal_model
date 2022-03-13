# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 18:06:37 2021

@author: Luca
"""

import os
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
import concurrent.futures
import sys
# import multiprocessing
import time
import logging
from multiprocessing_logging import install_mp_handler
import config
import pandas as pd
import numpy as np
import pypsa
import pyomo.environ as pe
import math
from functools import partial
from datetime import timedelta

class Inputs:
    """
    Input data object that takes all inputs from config file.
    """

    def __init__(self, config):
        self.period_start = config.period_start
        self.period_end = config.period_end
        # NOTE: for look_ahead window_size is input NOT number of steps
        self.window_size =config.window_size
        time_start = '{} 00:00:00'.format(self.period_start)
        time_end = '{} 23:00:00'.format(self.period_end)
        self.period = pd.date_range(start=time_start, end=time_end, freq='H')
        # different from parallel file where step is input
        self.steps = int((len(self.period) - self.window_size) / (0.5 * self.window_size))
        self.rest = len(self.period) - ((self.steps + 1) * 0.5 * self.window_size)

        # load network
        path_n = config.path_node
        self.n = pypsa.Network(path_n)
        # clean the nodal networks for lines that have 0 capacity
        self.n = remove_s_nom0_lines(self.n)

        # path for NTCs
        self.ntc_path = config.path_ntc

        # new for reoptimization
        # fill soc_set with nan values
        self.n.storage_units_t.state_of_charge_set[list(self.n.storage_units.index)] = np.nan
        # instead of soc_in assign path to soc in to self, so i can reload it everytime after updating the file
        self.soc_path = config.path_soc_in_and_out
        # turn off cyclic soc
        self.n.storage_units.cyclic_state_of_charge = False


# load ntc data and clean for those zones not included in the network
def load_ntc(network, ntc_path='D:/Python/PyPSA/Luca/NTC/NTC_2020_LJ.csv'):
    ntc = pd.read_csv(ntc_path)
    # split border in from and to
    ntc[['bus0', 'bus1']] = ntc['borders'].str.split('-', expand=True)
    ntc = ntc.set_index('borders')
    # remove from NTC df all that are not included in PyPSA
    # remove from ntc all lines connecting zones that dont exist in pypsa-eur network
    for ind, bus0, bus1 in zip(ntc.index, ntc.bus0, ntc.bus1):
        if bus0 not in set(network.buses.zone) or bus1 not in set(network.buses.zone):
            # print(ind,bus0,bus1)
            ntc = ntc.drop([ind])
    return ntc


# get dictionary of lines (& links) that are assigned to a particular NTC
# dictionary because for each NTC: there is a list of tuples; every tuple includes the line/link of the network that is assigned to NTC with the correct sign +1 or -1
# works also for cross-border flows: instead of ntc input needs to be network_zone.links
def lines_ntc(ntc, network, line_link):
    # assign lines in nodal network to respective NTCs with correct signs
    ntc_lines = dict()
    if line_link == "line":
        ind_buses = zip(network.lines.index, network.lines.bus0, network.lines.bus1)
    elif line_link == "link":
        ind_buses = zip(network.links.index, network.links.bus0, network.links.bus1)
    # else:
    # error define line or link keyword
    list_lines = [(ind, network.buses.zone[bus_from], network.buses.zone[bus_to]) for ind, bus_from, bus_to in ind_buses
                  if network.buses.zone[bus_from] != network.buses.zone[bus_to]]
    for ntc_name, ntc_bus0, ntc_bus1 in zip(ntc.index, ntc.bus0, ntc.bus1):
        line_list = []
        for line, line_bus0, line_bus1 in list_lines:
            if line_bus0 == ntc_bus0 and line_bus1 == ntc_bus1:
                line_list.append((line, 1))
            if line_bus0 == ntc_bus1 and line_bus1 == ntc_bus0:
                line_list.append((line, -1))
        ntc_lines[ntc_name] = line_list
    return ntc_lines


# remove lines in (nodal) network that have 0 capacity; in general not useful for OPF and cause admittance matrix to be singular (because of the way susceptance b is calculated based on properties of lines and line types)
def remove_s_nom0_lines(network):
    line_0_snom = [line for line in network.lines.index if network.lines.s_nom[line] == 0]
    print("removing the following lines because they have 0 capacity: {}".format(line_0_snom))
    network.mremove("Line", line_0_snom)
    return network


# from script nodal_model
def nodal_model(network_node, snapshots, ntc_path='D:/Python/PyPSA/Luca/NTC/NTC_2020_LJ.csv'):
    # load NTC data
    ntc = load_ntc(network_node, ntc_path)

    # get dictionaries for lines and links from nodal network belonging to NTC
    ntc_lines = lines_ntc(ntc, network_node, 'line')
    ntc_links = lines_ntc(ntc, network_node, 'link')

    # extra functionality for NTC soft constraints for full nodal model
    def nodal_ntc_constraint(network, snapshots):
        model = network.model
        ## penalty factor:
        # ntc violation
        f_ntc = config.f_ntc

        # introduce new slack variable for ntc soft constraints v0 for + and v1 for -
        model.v0 = pe.Var(list(ntc.index), list(snapshots), domain=pe.NonNegativeReals)
        model.v1 = pe.Var(list(ntc.index), list(snapshots), domain=pe.NonNegativeReals)

        # objective contributions:
        ntc_violation_pos = sum(f_ntc * model.v0[line, sn] for line in ntc.index for sn in snapshots)
        ntc_violation_neg = sum(f_ntc * model.v1[line, sn] for line in ntc.index for sn in snapshots)

        model.objective.expr += ntc_violation_pos + ntc_violation_neg

        # loop through NTCs and LInks and find which ones
        def constr_NTC_pos(model, ntc_border, sn):
            if not ntc_lines[ntc_border] and not ntc_links[ntc_border]:
                return pe.Constraint.Skip
            line_sum = sum(model.passive_branch_p['Line', line[0], sn] * line[1] for line in ntc_lines[ntc_border])
            link_sum = sum(model.link_p[link[0], sn] * link[1] for link in ntc_links[ntc_border])
            return link_sum + line_sum - ntc.loc[ntc_border, 'NTC_2020'] <= model.v0[ntc_border, sn]

        def constr_NTC_neg(model, ntc_border, sn):
            if not ntc_lines[ntc_border] and not ntc_links[ntc_border]:
                return pe.Constraint.Skip
            line_sum = sum(model.passive_branch_p['Line', line[0], sn] * line[1] for line in ntc_lines[ntc_border])
            link_sum = sum(model.link_p[link[0], sn] * link[1] for link in ntc_links[ntc_border])
            return -(ntc.loc[ntc_border, 'back'] + link_sum + line_sum) <= model.v1[ntc_border, sn]

        model.new_constraint5 = pe.Constraint(list(ntc.index), list(snapshots), rule=constr_NTC_pos)
        model.new_constraint6 = pe.Constraint(list(ntc.index), list(snapshots), rule=constr_NTC_neg)


    network_node.lopf(snapshots, solver_name='gurobi', extra_functionality=nodal_ntc_constraint)

    return network_node

# %%
def timespan_from_window(step, input_data):
    ts_start = (step / 2) * input_data.window_size
    ts_end = (step /2 + 1) * input_data.window_size
    # if there remains a rest of fractions of timesteps they are added to the last period
    ts_start = input_data.period[int(ts_start)]
    if step == input_data.steps-1:#((steps-1)*2)-1:  # this is for the final step so that the 'rest' is added to the last period
        ts_end = ts_end + input_data.rest
        ts_end = input_data.period[int(ts_end - 1)]
    else:
        ts_end = input_data.period[int(ts_end - 1)]  # without the -1 the time slices overlap; then I can set the initial soc in soc_set
    timespan = pd.date_range(start=ts_start, end=ts_end, freq='H')

    return timespan


def run_lopf(step, input_data):
    # get the timespan according to the step number and the length of the whole period
    timespan = timespan_from_window(step, input_data)

    # get the previous timestamp to set initial soc, if its the beginning of the year take the value from the end of the year
    if timespan[0] == pd.to_datetime('2018-01-01 00:00:00'):
        last_previous_time = pd.to_datetime('2018-12-31 23:00:00')
    else:
        last_previous_time = timespan[0] - timedelta(hours=1)

    # load soc data
    soc_in_out = pd.read_csv(input_data.soc_path)
    soc_in_out = soc_in_out.set_index('snapshot')
    soc_in_out.index = pd.to_datetime(soc_in_out.index)

    # set initial soc to value at the end of previous timestep
    input_data.n.storage_units.state_of_charge_initial = soc_in_out.loc[last_previous_time, :]
    # set soc set to value at the end of timespan
    input_data.n.storage_units_t.state_of_charge_set[timespan[-1], :] = soc_in_out.loc[timespan[-1], :]

    logging.info(f'performing LOPF for step {step} from {timespan[0]} to {timespan[-1]}')

    # run nodal model
    input_data.n = nodal_model(input_data.n, timespan, input_data.ntc_path)

    try:  # only if optimization successful assign ofv else infinity
        ofv = input_data.n.objective
        # only do this if optimization is feasbible do this:
        # output SOC dataframe because I need this for evaluation later
        soc_in_out.loc[timespan, :] = input_data.n.storage_units_t.state_of_charge.loc[timespan, :]
    except AttributeError:
        logging.warning(f'LOPF for step {step} infeasible')
        ofv = math.inf    # # this to just test if the parallelization works
    # ofv = step + 5
    logging.info(f'the objective function value for step {step} is {ofv}')

    return soc_in_out


def main():
    start = time.perf_counter()
    # Uncomment following to use log files:
    log_file = config.log_file
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)  # DEBUG
    log = open(log_file, "a")
    sys.stdout = log
    # change current working directory to the path where file is running
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    install_mp_handler()

    logging.info("script has started")
    # read all the input and create a class
    input_data = Inputs(config)


    #TODO: loop through steps
    for step in range(input_data.steps):
        # define step size and length outside of this run_lopf
        soc = run_lopf(step, input_data)
        #save results
        soc.to_csv(input_data.soc_path)

    finish = time.perf_counter()

    print(f'finished in {round(finish - start, 2)} second(s)')

if __name__ == "__main__":
    main()
