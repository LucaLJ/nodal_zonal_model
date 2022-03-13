# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 18:06:37 2021

@author: Luca
"""

import os
os.chdir('./reoptimization')

#%%
import concurrent.futures
import sys
# import multiprocessing
import time
import logging
from multiprocessing_logging import install_mp_handler
import reoptimization.config as config
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
        hour_start = config.hour_start
        hour_end = config.hour_end
        time_start = '{} {}:00:00'.format(self.period_start, hour_start)
        time_end = '{} {}:00:00'.format(self.period_end, hour_end)
        self.period = pd.date_range(start=time_start, end=time_end, freq='H')

        # NOTE: for parallel re-optimization number of steps is input not window_size
        # self.steps = config.no_steps
        self.window_size =config.window_size

        # this if step number is the input
        # different from parallel file where step is input
        # self.window_size = int((len(self.period) / self.steps))
        # self.rest = len(self.period) - self.steps * self.window_size
        # this if window size is the input
        self.steps = int((len(self.period) - self.window_size) / (0.5 * self.window_size))
        self.rest = len(self.period) - ((self.steps + 1) * 0.5 * self.window_size)

        # load network
        path_n = config.path_node
        self.n = pypsa.Network(path_n)
        # clean the nodal networks for lines that have 0 capacity
        self.n = remove_s_nom0_lines(self.n)

        # path for NTCs
        self.ntc_path = config.path_ntc

        # read SOC values
        self.soc_path = config.path_soc_in
        self.soc_in = pd.read_csv(self.soc_path)
        self.soc_in = self.soc_in.set_index('snapshot')
        self.soc_in.index = pd.to_datetime(self.soc_in.index)

        # new for reoptimization
        # fill soc_set with nan values
        self.n.storage_units_t.state_of_charge_set[list(self.n.storage_units.index)] = np.nan

        # turn off cyclic soc
        self.n.storage_units.cyclic_state_of_charge = False

## load ntc data and clean for those zones not included in the network
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
def timespan_from_window_overlap(step, input_data):
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

def timespan_from_window(step, input_data):
    ts_start = (step) * input_data.window_size
    ts_end = (step + 1) * input_data.window_size
    # if there remains a rest of fractions of timesteps they are added to the last period
    ts_start = input_data.period[int(ts_start)]
    if step == input_data.steps-1:#((steps-1)*2)-1:  # this is for the final step so that the 'rest' is added to the last period
        ts_end = ts_end + input_data.rest
        ts_end = input_data.period[int(ts_end - 1)]
    else:
        ts_end = input_data.period[int(ts_end - 1)]  # without the -1 the time slices overlap; then I can set the initial soc in soc_set
    timespan = pd.date_range(start=ts_start, end=ts_end, freq='H')

    return timespan

def timespan_from_steps(step, input_data):
    # get the timespan according to the step number and the length of the whole period
    ts_start = step * input_data.window_size
    ts_end = (step + 1) * input_data.window_size
    # if there remains a rest of fractions of timesteps they are added to the last period
    ts_start = input_data.period[ts_start]
    if step == input_data.steps - 1:  # this is for the final step so that the 'rest' is added to the last period
        ts_end = ts_end + input_data.rest
        ts_end = input_data.period[ts_end - 1]
    else:
        ts_end = input_data.period[
            ts_end - 1]  # without the -1 the time slices overlap; then I can set the initial soc in soc_set
    timespan = pd.date_range(start=ts_start, end=ts_end, freq='H')
    return timespan


def run_lopf(step, input_data):#, shift_index):
    # get the timespan according to the step number or the length of the whole period
    # timespan_from_steps for parallel re-optimization and timespan_from_window for look-ahead
    timespan = timespan_from_window(step, input_data)

    # get the previous timestamp to set initial soc, if its the beginning of the year take the value from the end of the year
    if timespan[0] == pd.to_datetime('2018-01-01 00:00:00'):
        last_previous_time = pd.to_datetime('2018-12-31 23:00:00')
    else:
        last_previous_time = timespan[0] - timedelta(hours=1)

    #TODO: input soc again from input_data

    # TODO: shift timespan unless its the last shift then the timespan should be the same as for 0.
    #  this means that number of shifts should be property of inputs class!
    #  if shift_index=0 do following if not load soc from results
    # if i == 0:
    #     soc_in = input_data.soc_in
    # else:
    #     soc_in = input_data.soc_path
    #     # load the soc_in data
    # load soc data
    soc_in = input_data.soc_in

    # set initial soc to value at the end of previous timestep
    input_data.n.storage_units.state_of_charge_initial = soc_in.loc[last_previous_time, :]
    # set soc set to value at the end of timespan
    input_data.n.storage_units_t.state_of_charge_set.loc[timespan[-1], :] = soc_in.loc[timespan[-1], :]

    logging.warning(f'performing LOPF for step {step} from {timespan[0]} to {timespan[-1]}')

    # run nodal model
    # input_data.n = nodal_model(input_data.n, timespan, input_data.ntc_path)

    # results of soc output: should be nan if optimization is infeasible
    soc_out = pd.read_csv(input_data.soc_path)
    soc_out = soc_out.set_index('snapshot')
    soc_out.index = pd.to_datetime(soc_out.index)
    # if the soc values are nan by default in case of infeasibility this will be passed to the results
    # if this line commented out, the previous soc values will be passed to results even in case of infeasibility
    # soc_out.loc[:, :] = np.nan


    try:  # only if optimization successful assign ofv else infinity
        ofv = input_data.n.objective
        # only do this if optimization is feasbible do this:
        # update output SOC dataframe because I need this for evaluation later
        soc_out.loc[timespan, :] = input_data.n.storage_units_t.state_of_charge.loc[timespan, :]
    except AttributeError:
        logging.warning(f'LOPF for step {step} infeasible')
        ofv = math.inf    # # this to just test if the parallelization works
    # ofv = step + 5
    logging.info(f'the objective function value for step {step} is {ofv}')
    return ofv, soc_out, timespan

#%%
# def main():
input_data = Inputs(config)
#print(input_data.n.storage_units_t.state_of_charge_set)

# soc output dataframe filled with nan in case optimization is infeasible during those times
soc_out_df = pd.read_csv(input_data.soc_path)
soc_out_df = soc_out_df.set_index('snapshot')
soc_out_df.index = pd.to_datetime(soc_out_df.index)
soc_out_df.loc[:, :] = np.nan
#%%
results = run_lopf(1,input_data)

ofv, soc_out, timespan = results
soc_out_df.loc[timespan, :] = soc_out.loc[timespan,:]
print('output from lopf')
print(soc_out)
print('full df')
print(soc_out_df)
#%% test if soc set is same as soc in df
timespan
#if ((input_data.n.storage_units_t.state_of_charge_set.loc[timespan[-1],:] == soc_out_df.loc[timespan[-1],:])==True).all():
    # print('all true')
#%%
input_data.steps
# timespan_from_window(1,input_data)
timespan_from_steps(2,input_data)
timespan
#%%
info_txt = ['the input soc profile comes from:', input_data.soc_path,
            'window size in hours:', str(input_data.window_size),
            'number of steps:', str(input_data.steps),
            'simulation finished in (seconds): ', str(round(10 - 5, 2))]
with open('path_save_info.txt', 'w') as f:
    f.writelines('\n'.join(info_txt))
#%%
    ## normal script starts here
    # start = time.perf_counter()
    # # Uncomment following to use log files:
    # log_file = config.log_file
    # logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)  # DEBUG
    # log = open(log_file, "a")
    # sys.stdout = log
    # # change current working directory to the path where file is running
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # install_mp_handler()
    #
    # logging.info("script has started")
    # # read all the input and create a class
    # input_data = Inputs(config)
    #
    # # TODO: build for loop around this and shift times
    #
    # # how often to shift
    # shifts = 5  # shift time window 3 times and 4th time should be same as 0.
    # # hours how far to shift
    # shift_h = input_data.window_size / (shifts - 1)
    # for i in range(shifts):
    #     # need to reload soc data from results
    #     # unless it is shift 0:
    #
    #
    # # this is to set the default inputs for run_lopf function otherwise cannot use map() properly
    # run_lopf_input = partial(run_lopf, input_data=input_data)
    #
    #
    #
    # #TODO: define output well especially soc dataframe
    # ofv_list = []
    # # soc output dataframe filled with nan in case optimization is infeasible during those times
    # soc_out_df = pd.read_csv(input_data.soc_path)
    # soc_out_df = soc_out_df.set_index('snapshot')
    # soc_out_df.index = pd.to_datetime(soc_out_df.index)
    # soc_out_df.loc[:, :] = np.nan
    #
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = executor.map(run_lopf_input, range(input_data.steps))
    #     print(type(results))
    #
    #     # unpack results
    #     for result in results:
    #         ofv, soc_out, timespan = result
    #         ofv_list.append(ofv)
    #         soc_out_df.loc[timespan, :] = soc_out
    #
    #
    # # save ofv results: df in csv and sum in txt file
    # ofv_df = pd.DataFrame(ofv_list)
    # results_obj_path = config.objective_path
    # ofv_df.to_csv(results_obj_path)
    #
    # # overall objective function value and save
    # ofv_sum = sum(ofv_list)
    #
    # results_sum_save = config.save_obj_txt
    # with open(results_sum_save, 'w') as file:
    #     file.write(str(ofv_sum))
    #
    # # save soc results
    # soc_out_df.to_csv(config.soc_out_save)
    #
    # #TODO: this is where above mentioned for loop should end
    #
    # finish = time.perf_counter()
    # print(f'finished in {round(finish - start, 2)} second(s)')

# if __name__ == "__main__":
#     main()
