# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:54:27 2021

@author: Luca01
"""


import pypsa
import numpy as np
import pandas as pd
import pyomo.environ as pe


# model to run the full nodal OPF
# required inputs: nodal network and the timespan to be optimized (snapshots)

def nodal_model(network_node,snapshots,ntc_path):
    
    # load NTC data
    ntc = prepare.load_ntc(ntc_path,network_node)
    
    # get dictionaries for lines and links from nodal network belonging to NTC
    ntc_lines = prepare.lines_ntc(ntc,network_node,'line')
    ntc_links = prepare.lines_ntc(ntc,network_node,'link')     
   
    # extra functionality for NTC soft constraints for full nodal model
    def nodal_ntc_constraint(network,snapshots):
        model = network.model
        ## penalty factor:
        # ntc violation
        f_ntc = 100
    
        #introduce new slack variable for ntc soft constraints v0 for + and v1 for -
        model.v0 = pe.Var(list(ntc.index), list(snapshots), domain = pe.NonNegativeReals)
        model.v1 = pe.Var(list(ntc.index), list(snapshots), domain = pe.NonNegativeReals)
        
        # objective contributions:
        ntc_violation_pos = sum(f_ntc*model.v0[line,sn] for line in ntc.index for sn in snapshots)
        ntc_violation_neg = sum(f_ntc*model.v1[line,sn] for line in ntc.index for sn in snapshots)
        
        model.objective.expr += ntc_violation_pos + ntc_violation_neg
        # loop through NTCs and LInks and find which ones
        def constr_NTC_pos(model,ntc_border,sn):
            if not ntc_lines[ntc_border] and not ntc_links[ntc_border]:
                return pe.Constraint.Skip
            line_sum = sum(model.passive_branch_p['Line',line[0],sn]*line[1] for line in ntc_lines[ntc_border])
            link_sum = sum(model.link_p[link[0],sn]*link[1] for link in ntc_links[ntc_border])
            return link_sum + line_sum - ntc.loc[ntc_border,'NTC_2020'] <= model.v0[ntc_border,sn]
        def constr_NTC_neg(model,ntc_border,sn):
            if not ntc_lines[ntc_border] and not ntc_links[ntc_border]:
                return pe.Constraint.Skip
            line_sum = sum(model.passive_branch_p['Line',line[0],sn]*line[1] for line in ntc_lines[ntc_border])
            link_sum = sum(model.link_p[link[0],sn]*link[1] for link in ntc_links[ntc_border])
            return -(ntc.loc[ntc_border,'back']+ link_sum + line_sum) <= model.v1[ntc_border,sn]
        model.new_constraint5 = pe.Constraint(list(ntc.index),list(snapshots),rule=constr_NTC_pos)
        model.new_constraint6 = pe.Constraint(list(ntc.index),list(snapshots),rule=constr_NTC_neg)

    network_node(snapshots,solver_name='gurobi',extra_functionality=nodal_ntc_constraint)

    return network_node