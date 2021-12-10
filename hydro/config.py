### Paths to networks and other data
path_zone = 'D:\\Python\\PyPSA\\Luca\\zonal_nodal_networks\\2018\\zonal_1024_costs2018.nc'
path_node = 'D:\\Python\\PyPSA\\Luca\\zonal_nodal_networks\\2018\\nodal_1024_costs2018.nc'
path_ntc = 'D:\\Python\\PyPSA\\Luca\\NTC\\NTC_2020_LJ.csv'
path_soc_in_sum_zone = 'D:\\Python\\PyPSA\\Luca\\data\\hydro\\2018\\hydro_weekly_sum_zone_04_2018.csv'
path_soc_in_su = 'D:\\Python\\PyPSA\\Luca\\data\\hydro\\2018\\hydro_weekly_su_individual_const_2018.csv'


## paths to save results
slack_path = './results/2018/zone/f_sum_10_f_su_0_3w/slack_variable_results_3w_su_nodal_individual_05.csv'
slack_path_zone = './results/2018/zone/f_sum_10_f_su_0_3w/slack_variable_results_3w_su_zonal.csv'
slack_sum_path = './results/2018/zone/f_sum_10_f_su_0_3w/slack_variable_results_3w_su_nodal_sum_05.csv'
soc_path_save = './results/2018/zone/f_sum_10_f_su_0_3w/soc_nodal_3w_both_05.csv'
save_path_zone = './results/2018/zone/f_sum_10_f_su_0_3w/nodal_1024_results_5w_weekly_05.nc'
lmp_path = './results/2018/zone/f_sum_10_f_su_0_3w/lmp_3w_nodal_05.csv'
objective_path = './results/2018/zone/f_sum_10_f_su_0_3w/objective_values_1y_zone.csv'
# log file
log_file = 'zonal_3w.log'

### Time window for optimization
period_start = "2018-01-01"
period_end = "2018-12-31"
# number of weeks to simulate
weeks_sim = 2

### Nodal and redispatch parameters

# penalty for the violation of NTC constraints (used in nodal model & redispatching model)
f_ntc = 100
# penalty for cross-border flow violation
f_x = 100

### Hydro parameters
# individual storage units
soc_penalty_factor = 0
# sum of storage units in zone
soc_penalty_sum_factor = 10 #100000
# penalty for zonal model        
soc_penalty_zone = 1000
