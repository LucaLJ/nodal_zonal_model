import os
import sys
import pypsa
import pandas as pd
import pyomo.environ as pe
import numpy as np
import logging
import config


# %%
def main():
    log_file = config.log_file
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)  # DEBUG
    log = open(log_file, "a")
    sys.stdout = log
    # change current working directory to the path where file is running
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

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
        soc_penalty = config.soc_penalty_zone
        # introduce slack variable
        model.soc_slack = pe.Var(set(n.storage_units.index), list(sn_fringe),
                                 domain=pe.NonNegativeReals)  # list(snapshots[times])list(sn_fringe)
        soc_penalty_sum = sum(
            model.soc_slack[su, time] * soc_penalty for su in n.storage_units.index for time in sn_fringe)
        #    soc_penalty_sum = sum(model.soc_slack[zone,time]*soc_penalty for zone in hydro_dict_only for time in snapshots[times])
        model.objective.expr += soc_penalty_sum

        def constr_input_plus(model, su, time):
            # this is the model variable
            soc = model.state_of_charge[su, snapshots[time]]
            #        return soc_sum==hydro_soc_in.loc[abs(time)+1,zone]
            soc_in = hydro_soc_in.loc[abs(time), su]
            soc_sl = model.soc_slack[su, sn_fringe[time]]
            return soc - soc_in <= soc_sl  # like this feasible!

        model.new_constraint_soc_plus = pe.Constraint(list(n.storage_units.index), list(times), rule=constr_input_plus)

        def constr_input_minus(model, su, time):
            # this is the model variable
            soc = model.state_of_charge[su, snapshots[time]]
            #        return soc_sum==hydro_soc_in.loc[abs(time)+1,zone]
            soc_in = hydro_soc_in.loc[abs(time), su]
            soc_sl = model.soc_slack[su, sn_fringe[time]]
            return -soc_sl <= soc - soc_in  # like this feasible!

        model.new_constraint_soc_minus = pe.Constraint(list(n.storage_units.index), list(times),
                                                       rule=constr_input_minus)
        # input for above list(hydro_dict)

    # %% run model for all weeks
    path = config.path_zone
    n = pypsa.Network(path)
    path_soc_in_su = config.path_soc_in_su
    hydro_soc_in = pd.read_csv(path_soc_in_su)

    # set cyclic soc to false and initial value =0 just to be sure
    n.storage_units.cyclic_state_of_charge = False
    n.storage_units.state_of_charge_initial = 0

    # %% for now drop all phs to test new idea first only for hydro
    # phs_remove = [phs for phs in n.storage_units.index if phs.__contains__('PHS')]
    # n.mremove("StorageUnit", phs_remove)

    # # dictionary that contains all hydro storages ordered by zone
    # hydro_dict = dict()
    # for zone in n.storage_units.bus.unique():
    #     hydro_dict[zone]=n.storage_units[n.storage_units.bus==zone].index

    # dictionary that contains all 'hydro' hydro storages ordered by zone
    hydro_dict_only = dict()
    for zone in n.storage_units.bus.unique():
        liste = []
        for su in n.storage_units.index:
            if not su.__contains__('PHS'):
                if n.storage_units.bus[su] == zone:
                    liste.append(su)
        hydro_dict_only[zone] = liste

    # dictionary that contains all "PHS" storages ordered by zone
    phs_dict = dict()
    for zone in n.storage_units.bus.unique():
        liste = []
        for su in n.storage_units.index:
            if su.__contains__('PHS'):
                if n.storage_units.bus[su] == zone:
                    liste.append(su)
        phs_dict[zone] = liste

    # %% define period and length of number of timesteps
    period_start = "2018-01-01"
    period_end = "2018-12-31"
    time_start = '{} 00:00:00'.format(period_start)
    time_end = '{} 23:00:00'.format(period_end)
    steps = 52
    period = pd.date_range(start=time_start, end=time_end, freq='H')
    # WHAT tp dp with the remaining
    timestep_length = int((len(period) / steps))
    rest = len(period) - steps * timestep_length
    # get values for slack variables
    slack_var = pd.DataFrame(columns=n.storage_units.index)
    obj_df = pd.DataFrame(columns=['objective in billion EUR'])
    obj_df.index.name = 'week'
    weeks_sim = config.weeks_sim

    # %% run model for 3 weeks
    for timestep in range(weeks_sim):  # range(52): # from 0 to 51 i.e. all weeks of the year
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

            # save objective function value
            obj_df.loc[1] = n.objective

        else:

            def hydro_consecutive_constr_slack(network, snapshots):
                model = network.model

                # penalty factor for violation of soc constraint
                soc_penalty = config.soc_penalty_zone
                # introduce slack variable
                model.soc_slack = pe.Var(set(n.storage_units.index), domain=pe.NonNegativeReals)

                soc_penalty_sum = sum(model.soc_slack[su] * soc_penalty for su in n.storage_units.index)

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

            print("perform opf for week {} ranging from {} to {}".format(timestep, timespan[0], timespan[-1]))
            n.lopf(snapshots=timespan, solver_name='gurobi', extra_functionality=hydro_consecutive_constr_slack)

            # get slack variable values
            liste = []
            for su in n.storage_units.index:
                liste.append(n.model.soc_slack[su].value)
            slack_var.loc[timestep + 1] = liste
            # save objective function value
            obj_df.loc[timestep + 1] = n.objective
        # at the end always pass on initial condition
        n.storage_units.state_of_charge_initial = n.storage_units_t.state_of_charge.loc[timespan[-1]]

    # %% save results
    # save_path_zone = 'D:\\Python\\PyPSA\\Luca\\jupyter\\hydro\\hydro_model\\zonal_1024_results_1y_weekly_02.nc'
    save_path_zone = config.save_path_zone
    n.export_to_netcdf(save_path_zone)

    slack_path_zone = config.slack_path_zone
    slack_var.to_csv(slack_path_zone)

    objective_path = config.objective_path
    obj_df.to_csv(objective_path)


if __name__ == "__main__":
    main()

# %%
import pypsa

path = 'D:\Python\PyPSA\Luca\zonal_nodal_networks\\2018\\zonal_1024_costs2018.nc'
n = pypsa.Network(path)

#%%
len(n.buses.country.unique())
# %%
n.storage_units.cyclic_state_of_charge = True
n.storage_units.state_of_charge_initial = n.storage_units.p_nom * n.storage_units.max_hours

# %%
a = 5
# a.to_csv('out.csv')
with open('out.txt', 'w') as output:
    output.write(str(a))

# %% read csv target values
import pandas as pd

final_soc = pd.read_csv(
    'D:\Python\PyPSA\Luca\\nodal_zonal_model\hydro\hpc_results\hydro_zonal\\52w\soc_zonal_final_1y.csv')
# %% write to soc set

df = pd.DataFrame(columns=n.storage_units.index,
                  index=n.storage_units_t.state_of_charge_set.index)
n.storage_units_t.state_of_charge_set = df
n.storage_units_t.state_of_charge_set.iloc[-1,:] = final_soc.iloc[:,1]

#%% solved network
net = pypsa.Network('D:\Python\PyPSA\Luca\\nodal_zonal_model\hydro\hpc_results\hydro_zonal\\52w\zonal_1024_results_weekly_.nc')