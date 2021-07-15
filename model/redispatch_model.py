# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:29:18 2021

@author: Luca01
"""


import pypsa
import numpy as np
import pandas as pd
import pyomo.environ as pe

import prepare

# model that performs 1st an OPF for the zonal DAM followed by 2nd an OPF for the nodal redispatching
# inputs required are zonal and nodal network and the timespan to be optimized

def redispatch_model(network_zone,network_node,snapshots,ntc_path):
    
    # load NTC data
    ntc = prepare.load_ntc(ntc_path,network_node)
    
    # get dictionaries for lines and links from nodal network belonging to NTC
    ntc_lines = prepare.lines_ntc(ntc,network_node,'line')
    ntc_links = prepare.lines_ntc(ntc,network_node,'link')
    
    # lines and links that belong to cross-border flows 
    Xlines = prepare.lines_ntc(network_zone.links,network_node,"line")
    Xlinks = prepare.lines_ntc(network_zone.links,network_node,"link")
    
    def redispatch(network,snapshots):
        model = network.model
    
        ###variable (as defined in PyPSA):
        # redispatch generation
        g_rd = model.generator_p #[gen,t]
        ###parameters: 
        #(output from zonal da (day-ahead)):
        # dispatched generation
        g_da =  np.transpose(network_zone.generators_t.p) #.loc[gen,t]
        # cross border flows
        #flows_DA = np.transpose(network_zone.links_t.p0) #.loc[link,t]
    
        # marginal costs
        mar_cost = network.generators.marginal_cost #.at[gen]
        ## penalty factors:
        # Cross-border flow violation
        f_x = 100
        # ntc violation
        f_ntc = 101
        # new variable s for substitution!
        model.s = pe.Var(list(network.generators.index), list(snapshots), initialize=0, domain=pe.NonNegativeReals)#NonNegative
    
        # redispatching costs: redispatched power*marginal costs of generator
        gen_RD = sum(model.s[gen,sn]*mar_cost[gen] for gen in network.generators.index for sn in snapshots)
    
        model.objective = pe.Objective(expr=gen_RD)
    
        # add extra constraint for substitution var s
        def constraint1(model,gen,sn):
            return -model.s[gen,sn] - g_rd[gen,sn] <= - g_da.loc[gen,sn]
        def constraint2(model,gen,sn):
            return model.s[gen,sn] - g_rd[gen,sn] >= - g_da.loc[gen,sn]
        model.new_constraint1 = pe.Constraint(list(network.generators.index),list(snapshots), rule = constraint1)
        model.new_constraint2 = pe.Constraint(list(network.generators.index),list(snapshots), rule = constraint2)
        
        # CROSS-BORDER SOFT CONSTRAINT
        # introduce slack variable for cross-border flows (min|F_RD-F_DA|)
        model.f = pe.Var(list(network_zone.links.index), list(snapshots), domain=pe.NonNegativeReals)
        
        flow_penalty = sum(model.f[link,sn]*f_x for link in network_zone.links.index for sn in snapshots)
        model.objective.expr += flow_penalty
        
        # add constraints for cross-border flow substitution variable
        def constr_Xflow_1(model,link_zone,sn):
            # if no lines or links correspond to this cross-border flow than need to skip Constraint
            if not Xlines[link_zone] and not Xlinks[link_zone]:
                return pe.Constraint.Skip
            #input (cross-border link flows from DAM)
            flows_DA = network_zone.links_t.p0.loc[sn,link_zone]
            #variable
            flows_RD_lines = sum(model.passive_branch_p['Line',line[0],sn]*line[1] for line in Xlines[link_zone])
            flows_RD_links = sum(model.link_p[link[0],sn]*link[1] for link in Xlinks[link_zone])
            flows_RD = flows_RD_lines + flows_RD_links
            return flows_RD - flows_DA <= model.f[link_zone,sn]
    
        def constr_Xflow_2(model,link_zone,sn):
            if not Xlines[link_zone] and not Xlinks[link_zone]:
                return pe.Constraint.Skip
            #input (cross-border link flows from DAM)
            flows_DA = network_zone.links_t.p0.loc[sn,link_zone]
            #variable
            flows_RD_lines = sum(model.passive_branch_p['Line',line[0],sn]*line[1] for line in Xlines[link_zone])
            flows_RD_links = sum(model.link_p[link[0],sn]*link[1] for link in Xlinks[link_zone])
            flows_RD = flows_RD_lines + flows_RD_links
            return flows_RD - flows_DA >= -model.f[link_zone,sn]
        model.new_constraint3 = pe.Constraint(list(network_zone.links.index),list(snapshots), rule=constr_Xflow_1)
        model.new_constraint4 = pe.Constraint(list(network_zone.links.index),list(snapshots), rule=constr_Xflow_2)
    
        #NTC SOFT CONSTRAINT
        #introduce new slack variable for ntc soft constraints (on lines!) v0 for + and v1 for -
        model.v0 = pe.Var(list(ntc.index), list(snapshots), domain = pe.NonNegativeReals)
        model.v1 = pe.Var(list(ntc.index), list(snapshots), domain = pe.NonNegativeReals)
        
        # objective contributions:
        ntc_violation_pos = sum(f_ntc*model.v0[line,sn] for line in ntc.index for sn in snapshots)
        ntc_violation_neg = sum(f_ntc*model.v1[line,sn] for line in ntc.index for sn in snapshots)
        
        model.objective.expr += ntc_violation_pos + ntc_violation_neg
    
        # loop through NTCs and Lines & Links and find which ones
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

    # run first the zonal DAM OPF
    network_zone.lopf(snapshots,solver_name='gurobi')
    
    # run next the nodal redispatching
    network_node.lopf(snapshots,solver_name='gurobi',extra_functionality=redispatch)
    
    return network_zone,network_node
