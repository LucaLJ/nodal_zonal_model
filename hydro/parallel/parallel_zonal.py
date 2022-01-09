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
import pypsa
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
        self.no_weeks = config.weeks_sim
        self.steps = self.no_weeks
        time_start = '{} 00:00:00'.format(self.period_start)
        time_end = '{} 23:00:00'.format(self.period_end)
        self.period = pd.date_range(start=time_start, end=time_end, freq='H')
        # WHAT tp dp with the remaining
        self.timestep_length = int((len(self.period) / self.steps))
        self.rest = len(self.period) - self.steps * self.timestep_length
        # load zonal network
        path_zone = config.path_zone
        self.n = pypsa.Network(path_zone)

        # read SOC values
        path_soc_hourly = config.path_soc_hourly
        soc_in = pd.read_csv(path_soc_hourly)
        soc_in = soc_in.set_index('snapshot')
        soc_in.index = pd.to_datetime(soc_in.index)
        # save SOC values in network soc_set
        self.n.storage_units_t.state_of_charge_set = soc_in

        # turn off cyclic soc
        self.n.storage_units.cyclic_state_of_charge = True


# %%

def run_lopf(step, input_data):
    # get the timespan according to the step number and the length of the whole period
    ts_start = step * input_data.timestep_length
    ts_end = (step + 1) * input_data.timestep_length
    # I need this to set initial soc
    last_previous_time = input_data.period[ts_start - 1]
    # if there remains a rest of fractions of timesteps they are added to the last period
    ts_start = input_data.period[ts_start]
    if step == input_data.steps - 1:  # this is for the final step so that the 'rest' is added to the last period
        ts_end = ts_end + input_data.rest
        ts_end = input_data.period[ts_end - 1]
    else:
        ts_end = input_data.period[
            ts_end - 1]  # without the -1 the time slices overlap; then I can set the initial soc in soc_set
    timespan = pd.date_range(start=ts_start, end=ts_end, freq='H')
    # get the previous timestamp to set initial soc, if its the beginning of the year take the value from the end of the year
    if timespan[0] == pd.to_datetime('2018-01-01 00:00:00'):
        last_previous_time = pd.to_datetime('2018-12-31 23:00:00')
    else:
        last_previous_time = timespan[0] - timedelta(hours=1)
    # set initial soc to value at the end of previous timestep
    input_data.n.storage_units.state_of_charge_initial = input_data.soc_in.loc[last_previous_time, :]

    print(f'performing LOPF for step {step} from {ts_start} to {ts_end}')
    # commented out to run on own computer
    input_data.n.lopf(timespan, solver_name='gurobi')
    # only if optimization successful assign ofv else infity
    try:
        ofv = input_data.n.objective
    except AttributeError:
        print(f'LOPF for step {step} infeasible')
        ofv = math.inf
    # ofv = step + 5
    print(f'the objective function value for step {step} is {ofv}')
    logging.warning(f'the objective function value for step {step} is {ofv}')
    return ofv


def main():
    start = time.perf_counter()
    # Uncomment following to use log files:
    log_file = "loggi_file_zone_01"
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)  # DEBUG
    log = open(log_file, "a")
    sys.stdout = log
    # change current working directory to the path where file is running
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    install_mp_handler()

    input_data = Inputs(config)

    logging.info("script has started")

    # # this is to set the default inputs for run_lopf function otherwise cannot use map() properly
    # run_lopf_input = partial(run_lopf, input_data=input_data)
    #
    # results_list = []
    #
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = executor.map(run_lopf_input, range(input_data.no_weeks))
    #
    #     for result in results:
    #         print(result)
    #         results_list.append(result)
    #
    # # overall objective function value
    # ofv_sum = sum(results_list)

    # try with one step first
    results_list = []
    result = run_lopf(3, input_data)
    results_list.append(result)

    # output the results for objective function value
    print(results_list)
    results_df = pd.DataFrame(results_list)

    results_df.to_csv('ofv.csv')

    finish = time.perf_counter()

    print(f'finished in {round(finish - start, 2)} second(s)')


if __name__ == "__main__":
    main()
