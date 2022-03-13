
import os
import sys
import pypsa
import pyomo.environ as pe
import pandas as pd
import numpy as np


def min_soc_extra(network, snapshots):
    model = network.model

    # def the constraint for min 30% soc at every hour for all storage units
    def min30_function(model, su, sn):
        # model variable
        soc = model.state_of_charge[su, sn]
        # max capacity
        soc_cap = network.storage_units.p_nom[su]*network.storage_units.max_hours[su]
        return (0.3 * soc_cap,soc,None)

    # define constraint
    model.min30soc_constraint = pe.Constraint(list(network.storage_units.index), list(snapshots), rule=min30_function)


def extra_postprocessing(network, snapshots, duals):
    model = network.model
    duals = pd.Series(list(model.dual.values()), index=pd.Index(list(model.dual.keys())),
                      dtype=float)

    def allocate_series_dataframes(network, series):
        for component, attributes in series.items():

            df = network.df(component)
            pnl = network.pnl(component)

            for attr in attributes:
                pnl[attr] = pnl[attr].reindex(columns=df.index,
                                              fill_value=np.nan)
                # network.components[component]["attrs"].at[attr,"default"])

    def set_from_series(df, series):
        df.loc[snapshots] = series.unstack(0).reindex(columns=df.columns)

    def get_shadows(constraint, multiind=True):
        print(len(constraint))
        # if len(constraint) == 0: return pd.Series(dtype=float)

        index = list(constraint.keys())
        #         print(index)
        if multiind:
            index = pd.MultiIndex.from_tuples(index)
        print(index)
        cdata = pd.Series(list(constraint.values()), index=index)
        return cdata.map(duals)

    # only possible to use names of shadow prices that are pre-defined
    allocate_series_dataframes(network, {'StorageUnit': ['mu_upper', 'mu_lower','mu_state_of_charge_set']})
    #     print(model.storage_p_lower)
    #     print(type(model.storage_p_lower))
    #     print(model.link_p_lower)
    #     print(type(model.link_p_lower))
    #BUT then it is possible to get 'all' the shadows you want
    set_from_series(network.storage_units_t.mu_upper, get_shadows(model.state_of_charge_upper, multiind=True))
    set_from_series(network.storage_units_t.mu_lower, get_shadows(model.min30soc_constraint, multiind=True))
    set_from_series(network.storage_units_t.mu_state_of_charge_set, get_shadows(model.state_of_charge_constraint, multiind=True))

#%%
#def main():
period_start = '2018-01-01'
period_end = '2018-01-01'
time_start = '{} 00:00:00'.format(period_start)
time_end = '{} 23:00:00'.format(period_end)
period = pd.date_range(start=time_start, end=time_end, freq='H')

# load network
path = 'D:\Python\PyPSA\Luca\zonal_nodal_networks/2018/zonal_1024_costs2018.nc'
n = pypsa.Network(path)

# run Lopf
n.lopf(period,solver_name='gurobi',extra_functionality=min_soc_extra,extra_postprocessing=extra_postprocessing)
#%%
mu3 = n.storage_units_t.mu_state_of_charge_set.loc[period,:]
lmp = n.buses_t.marginal_price.loc[period,:]
# if __name__ == "__main__":
#     main()
#%%
n.storage_units