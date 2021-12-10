import os
import sys
import pypsa
import numpy as np
import pandas as pd
import logging
import pyomo.environ as pe
import config


def main():
    log_file = config.log_file
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)  # DEBUG
    log = open(log_file, "a")
    sys.stdout = log
    # change current working directory to the path where file is running
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # %% prepare running model
    zone_path = config.path_zone
    node_path = config.path_node
    ntc_path = config.path_ntc
    n = pypsa.Network(node_path)
    hydro_soc_in = pd.read_csv(config.path_soc_in_su)
    hydro_soc_sum_in = pd.read_csv(config.path_soc_in_sum_zone)
    ## penalty for violating the input soc constraint
    # individual storage units
    soc_penalty_factor = config.soc_penalty_factor
    # sum of storage units in zone
    soc_penalty_sum_factor = config.soc_penalty_sum_factor

    # %% function to remove lines with 0 capacity
    def remove_s_nom0_lines(network):
        line_0_snom = [line for line in network.lines.index if network.lines.s_nom[line] == 0]
        print("removing the following lines because they have 0 capacity: {}".format(line_0_snom))
        network.mremove("Line", line_0_snom)
        return network

    # %% functions to load ntc data and dictionary for lines and links
    # load ntc data and clean for those zones not included in the network
    def load_ntc(ntc_path, network):
        ntc = pd.read_csv(ntc_path)
        # split border in from and to
        ntc[['bus0', 'bus1']] = ntc['borders'].str.split('-', expand=True)
        ntc = ntc.set_index('borders')
        # remove from NTC df all that are not included in PyPSA
        # remove from ntc all lines connecting zones that dont exist in pypsa-eur network
        for ind, bus0, bus1 in zip(ntc.index, ntc.bus0, ntc.bus1):
            if bus0 not in set(network.buses.zone) or bus1 not in set(network.buses.zone):
                print(ind, bus0, bus1)
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
        list_lines = [(ind, network.buses.zone[bus_from], network.buses.zone[bus_to]) for ind, bus_from, bus_to in
                      ind_buses
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

    # %% def constraint for week 1

    # ZONE 1st constraint WITH SLACK
    # constraint to be enforced at first period
    # constraint that enforces at time t=0 and t=T the sum given in hydro csv
    def hydro_input_constr_zone_slack(network, snapshots):
        model = network.model

        # try removing the cyclic first constraint:
        # model.del_component(model.state_of_charge_constraint[:,snapshots[0]])
        times = [0, -1]
        sn_fringe = snapshots[times]

        # penalty factor for violation of soc constraint
        soc_penalty_factor = config.soc_penalty_factor
        # introduce slack variable
        model.soc_slack = pe.Var(set(network.storage_units.index), list(sn_fringe),
                                 domain=pe.NonNegativeReals)  # list(snapshots[times])list(sn_fringe)
        soc_penalty_sum = sum(
            model.soc_slack[su, time] * soc_penalty_factor for su in network.storage_units.index for time in sn_fringe)
        #    soc_penalty_sum = sum(model.soc_slack[zone,time]*soc_penalty for zone in hydro_dict_only for time in snapshots[times])
        model.objective.expr += soc_penalty_sum

        def constr_input_plus(model, su, time):
            # this is the model variable
            soc = model.state_of_charge[su, snapshots[time]]
            #        return soc_sum==hydro_soc_in.loc[abs(time)+1,zone]
            soc_in = hydro_soc_in.loc[abs(time), su]
            soc_sl = model.soc_slack[su, sn_fringe[time]]
            return soc - soc_in <= soc_sl  # like this feasible!

        model.new_constraint_soc_plus = pe.Constraint(list(network.storage_units.index), list(times),
                                                      rule=constr_input_plus)

        def constr_input_minus(model, su, time):
            # this is the model variable
            soc = model.state_of_charge[su, snapshots[time]]
            #        return soc_sum==hydro_soc_in.loc[abs(time)+1,zone]
            soc_in = hydro_soc_in.loc[abs(time), su]
            soc_sl = model.soc_slack[su, sn_fringe[time]]
            return -soc_sl <= soc - soc_in  # like this feasible!

        model.new_constraint_soc_minus = pe.Constraint(list(network.storage_units.index), list(times),
                                                       rule=constr_input_minus)
        # input for above list(hydro_dict)

        ## Additional constraint for the sum of soc

        # introduce slack variable
        model.soc_slack_sum = pe.Var(set(hydro_dict), list(sn_fringe),
                                     domain=pe.NonNegativeReals)  # list(snapshots[times])list(sn_fringe)
        soc_penalty_sum_factor = config.soc_penalty_sum_factor
        soc_penalty_sum_sum = sum(
            model.soc_slack_sum[zone, time] * soc_penalty_sum_factor for zone in hydro_dict for time in sn_fringe)
        model.objective.expr += soc_penalty_sum_sum

        def constr_input_sum_plus(model, zone, time):
            # this is the model variable
            soc_sum = sum(model.state_of_charge[storage, snapshots[time]] for storage in hydro_dict[zone])
            #        return soc_sum==hydro_soc_in.loc[abs(time)+1,zone]
            soc_in = hydro_soc_sum_in.loc[abs(time), zone]
            soc_sl = model.soc_slack_sum[zone, sn_fringe[time]]
            return soc_sum - soc_in <= soc_sl  # like this feasible!

        model.new_constraint_soc_plus_sum = pe.Constraint(list(hydro_dict), list(times), rule=constr_input_sum_plus)

        def constr_input_sum_minus(model, zone, time):
            # this is the model variable
            soc_sum = sum(model.state_of_charge[storage, snapshots[time]] for storage in hydro_dict[zone])
            #        return soc_sum==hydro_soc_in.loc[abs(time)+1,zone]
            soc_in = hydro_soc_sum_in.loc[abs(time), zone]
            soc_sl = model.soc_slack_sum[zone, sn_fringe[time]]
            return -soc_sl <= soc_sum - soc_in  # like this feasible!

        model.new_constraint_soc_minus_sum = pe.Constraint(list(hydro_dict), list(times), rule=constr_input_sum_minus)

        ## end soc sum zone constraint

        ### constraint for NTC for nodal model
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

    # set cyclic soc to false and initial value =0 just to be sure
    n.storage_units.cyclic_state_of_charge = False
    n.storage_units.state_of_charge_initial = 0

    # %% for nodal model:
    # remove lines with 0 capacity
    n = remove_s_nom0_lines(n)

    # load NTC data
    ntc = load_ntc(ntc_path, n)

    # get dictionaries for lines and links from nodal network belonging to NTC
    ntc_lines = lines_ntc(ntc, n, 'line')
    ntc_links = lines_ntc(ntc, n, 'link')

    # %% Dictionaries with hydro plants
    # #for now drop all phs to test new idea first only for hydro
    # phs_remove = [phs for phs in n.storage_units.index if phs.__contains__('PHS')]
    # n.mremove("StorageUnit", phs_remove)

    # dictionary that contains all hydro storages ordered by zone
    hydro_dict = dict()
    for zone in n.buses.zone.unique():
        buses = n.buses[n.buses.zone == zone].index
        storages = [storage for storage in n.storage_units.index if n.storage_units.bus[storage] in buses]
        if storages:
            hydro_dict[zone] = storages

    # %% define period and length of number of time steps
    period_start = config.period_start
    period_end = config.period_end
    time_start = '{} 00:00:00'.format(period_start)
    time_end = '{} 23:00:00'.format(period_end)
    steps = 52
    period = pd.date_range(start=time_start, end=time_end, freq='H')
    # WHAT tp dp with the remaining
    timestep_length = int((len(period) / steps))
    rest = len(period) - steps * timestep_length
    # get values for slack variables
    slack_var = pd.DataFrame(columns=n.storage_units.index)
    slack_var_sum = pd.DataFrame(columns=hydro_dict)

    # %% run model for 52 weeks
    for timestep in range(5):  # range(steps): # from 0 to 51
        ts_start = timestep * timestep_length
        ts_end = (timestep + 1) * timestep_length
        # if there remains a rest of fractions of timesteps they are added to the last period
        ts_start = period[ts_start]
        if timestep == steps - 1:
            ts_end = ts_end + rest
            ts_end = period[ts_end - 1]
        else:
            ts_end = period[
                ts_end - 1]  # without the -1 the time slices overlap; then I can set the initial soc in soc_set
        timespan = pd.date_range(start=ts_start, end=ts_end, freq='H')
        if timestep == 0:
            print("perform opf for week {} ranging from {} to {}".format(timestep, timespan[0], timespan[-1]))
            n.lopf(snapshots=timespan, solver_name='gurobi', extra_functionality=hydro_input_constr_zone_slack)

            # get values for slack variables

            liste = []
            for su in n.storage_units.index:
                liste.append(n.model.soc_slack[su, timespan[0]].value)
            slack_var.loc[0] = liste
            liste = []
            for su in n.storage_units.index:
                liste.append(n.model.soc_slack[su, timespan[0]].value)
            slack_var.loc[1] = liste

            # same for slack variables for sum
            liste = []
            for zone in hydro_dict:
                liste.append(n.model.soc_slack_sum[zone, timespan[0]].value)
            slack_var_sum.loc[0] = liste
            liste = []
            for zone in hydro_dict:
                liste.append(n.model.soc_slack_sum[zone, timespan[0]].value)
            slack_var_sum.loc[1] = liste
        else:

            def hydro_consecutive_constr_slack(network, snapshots):
                model = network.model

                # introduce slack variable
                model.soc_slack = pe.Var(set(n.storage_units.index), domain=pe.NonNegativeReals)
                soc_penalty_factor = config.soc_penalty_factor
                soc_penalty_sum = sum(model.soc_slack[su] * soc_penalty_factor for su in n.storage_units.index)

                model.objective.expr += soc_penalty_sum

                # constraint for the end of period i.e. at snapshots[-1]
                def constr_input_plus(model, su):
                    soc = model.state_of_charge[su, snapshots[-1]]
                    soc_in = hydro_soc_in.loc[timestep + 1, su]
                    soc_sl = model.soc_slack[su]
                    return soc - soc_in <= soc_sl

                model.new_constraint_plus = pe.Constraint(list(n.storage_units.index), rule=constr_input_plus)

                # constraint for the end of period i.e. at snapshots[-1]
                def constr_input_minus(model, su):
                    soc = model.state_of_charge[su, snapshots[-1]]
                    soc_in = hydro_soc_in.loc[timestep + 1, su]
                    soc_sl = model.soc_slack[su]
                    return -soc_sl <= soc - soc_in

                model.new_constraint_minus = pe.Constraint(list(n.storage_units.index), rule=constr_input_minus)

                ## Additional constraint for the sum of soc

                # introduce slack variable
                model.soc_slack_sum = pe.Var(set(hydro_dict),
                                             domain=pe.NonNegativeReals)  # list(snapshots[times])list(sn_fringe)
                soc_penalty_sum_factor = config.soc_penalty_sum_factor
                soc_penalty_sum_sum = sum(model.soc_slack_sum[zone] * soc_penalty_sum_factor for zone in hydro_dict)
                model.objective.expr += soc_penalty_sum_sum

                def constr_input_sum_plus(model, zone):
                    # this is the model variable
                    soc_sum = sum(model.state_of_charge[storage, snapshots[-1]] for storage in hydro_dict[zone])
                    soc_in = hydro_soc_sum_in.loc[timestep + 1, zone]
                    soc_sl = model.soc_slack_sum[zone]
                    return soc_sum - soc_in <= soc_sl

                model.new_constraint_soc_plus_sum = pe.Constraint(list(hydro_dict), rule=constr_input_sum_plus)

                def constr_input_sum_minus(model, zone):
                    # this is the model variable
                    soc_sum = sum(model.state_of_charge[storage, snapshots[-1]] for storage in hydro_dict[zone])
                    soc_in = hydro_soc_sum_in.loc[timestep + 1, zone]
                    soc_sl = model.soc_slack_sum[zone]
                    return -soc_sl <= soc_sum - soc_in  # like this feasible!

                model.new_constraint_soc_minus_sum = pe.Constraint(list(hydro_dict), rule=constr_input_sum_minus)

                ## end soc sum zone constraint

                ### constraint for NTC for nodal model
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
                    line_sum = sum(
                        model.passive_branch_p['Line', line[0], sn] * line[1] for line in ntc_lines[ntc_border])
                    link_sum = sum(model.link_p[link[0], sn] * link[1] for link in ntc_links[ntc_border])
                    return link_sum + line_sum - ntc.loc[ntc_border, 'NTC_2020'] <= model.v0[ntc_border, sn]

                def constr_NTC_neg(model, ntc_border, sn):
                    if not ntc_lines[ntc_border] and not ntc_links[ntc_border]:
                        return pe.Constraint.Skip
                    line_sum = sum(
                        model.passive_branch_p['Line', line[0], sn] * line[1] for line in ntc_lines[ntc_border])
                    link_sum = sum(model.link_p[link[0], sn] * link[1] for link in ntc_links[ntc_border])
                    return -(ntc.loc[ntc_border, 'back'] + link_sum + line_sum) <= model.v1[ntc_border, sn]

                model.new_constraint5 = pe.Constraint(list(ntc.index), list(snapshots), rule=constr_NTC_pos)
                model.new_constraint6 = pe.Constraint(list(ntc.index), list(snapshots), rule=constr_NTC_neg)

            print("perform opf for week {} ranging from {} to {}".format(timestep, timespan[0], timespan[-1]))
            n.lopf(snapshots=timespan, solver_name='gurobi', extra_functionality=hydro_consecutive_constr_slack)

            # get slack variable values
            liste = []
            for su in n.storage_units.index:
                liste.append(n.model.soc_slack[su].value)
            slack_var.loc[timestep + 1] = liste
            # same for sum over zones
            # same for slack variables for sum
            liste = []
            for zone in hydro_dict:
                liste.append(n.model.soc_slack_sum[zone].value)
            slack_var_sum.loc[timestep + 1] = liste

        # at the end always pass on initial condition
        n.storage_units.state_of_charge_initial = n.storage_units_t.state_of_charge.loc[timespan[-1]]

    # %% save results
    # save_path_zone = 'D:\\Python\\PyPSA\\Luca\\jupyter\\hydro\\hydro_model\\zonal_1024_results_1y_weekly_02.nc'
    # n.export_to_netcdf(save_path_zone)

    slack_path = config.slack_path
    slack_var.to_csv(slack_path)

    # same for sum
    slack_sum_path = config.slack_sum_path
    slack_var_sum.to_csv(slack_sum_path)
    # %%
    soc_path_save = config.soc_path_save
    n.storage_units_t.state_of_charge.to_csv(soc_path_save)

    save_path_zone = config.save_path_zone
    n.export_to_netcdf(save_path_zone)

    # %% LMP results
    weeks5 = pd.date_range(start='{} 00:00:00'.format(period_start), end='{} 23:00:00'.format("2018-02-04"), freq='H')

    # get LMP results
    LMP = n.buses_t.marginal_price.loc[weeks5, :]
    LMP_stats = LMP.describe()
    lmp_path = config.lmp_path
    LMP_stats.to_csv(lmp_path)


if __name__ == "__main__":
    main()
