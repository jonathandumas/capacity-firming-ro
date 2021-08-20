# -*- coding: UTF-8 -*-

"""
Compute the dispatch variables using the planning from the BD or CCG algorithms.
"""

import os
import random

import pandas as pd

from ro.determinist.algorithms import Planner_MILP, controller_loop
from ro import build_observations, build_point_intra_forecast, read_file
from root_project import ROOT_DIR

from ro.ro_simulator_configuration import PARAMETERS

# Solver parameters
solver_param = dict()
solver_param['heuristic'] = True
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 1
solver_param['Threads'] = 1

# NB periods
nb_periods = 96
N_Q = 9 # nb quantiles
q_set = [0, 1, 2, 3, 4, 5, 6, 7, 8] # -> 10%, 20%, .., 90%

# --------------------------------------
# Static RO parameters: [q_min, gamma]
GAMMA = 12 # Budget of uncertainty to specify the number of time periods where PV generation lies within the uncertainty interval: 0: to 95 -> 0 = no uncertainty
q_min = 0  # Extreme minimum quantile to define the PV uncertainty interval: 0 to 3 -> 0 = 10 % to 40 %

# Dynamic RO parameters
dyn_RO_params = False
d_gamma = 0.1 # max distance for Gamma selection
gamma_threshold = d_gamma * PARAMETERS["pv_capacity"]
d_pvmin = 0.1 # max distance to min PV quantile
# --------------------------------------

# TEST set
nb_days = 30  # 30, 334 = total number of days of the dataset
TEST_size = str(nb_days)

# Forecasting technique to compute the PV quantiles
q_technique = 'NF' # 'NF', 'LSTM'

# --------------------------------------
# Algorithm: BD or CCG
algo = 'CCG' # BD, CCG
# warm_start
if algo == 'CCG':
    warm_start = False # WARNING no warm start for CCG !!!!
else:
    warm_start = True
# --------------------------------------

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    print('-----------------------------------------------------------------------------------------------------------')
    if dyn_RO_params:
        print('%s TEST size %s depth_threshold %s GAM_threshold %s' % (algo, TEST_size, int(100 * d_pvmin), int(100 * d_gamma)))
    else:
        print('%s TEST size %s gamma %s quantile %s' % (algo, TEST_size, GAMMA, q_min))
    print('%s: %s warm start %s PENALTY_FACTOR %s DEADBAND %s OVERPRODUCTION %s' %(algo, q_technique, warm_start, PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), PARAMETERS['OVERPRODUCTION']))
    print('-----------------------------------------------------------------------------------------------------------')

    # Folder where is stored the planning computed by either BD or CCG
    dirname = 'ro/robust/export_'+algo+'_'+q_technique+'/'
    if dyn_RO_params:
        dirname += TEST_size + '_ro_params/' + str(int(100 * d_pvmin)) + '/' + str(int(100 * d_gamma)) + '/'
    else:
        dirname += TEST_size + '_ro_static/' + str(q_min) + '/' + str(GAMMA) + '/'

    # load data
    pv_solution = build_observations()
    pv_intra = build_point_intra_forecast()
    # Load quantile forecasts: from 0 to 8 -> median = quantile 4
    day_list = [day.strftime('%Y-%m-%d') for day in pv_intra.index]
    # select N days into the dataset
    random.seed(1)
    if nb_days == len(day_list):
        day_list_sampled = day_list
    else:
        day_list_sampled = random.sample(day_list, nb_days)

    res = []
    n_days_remaining = nb_days
    # Loop over all days of the dataset
    for day in day_list_sampled:
        print(' ')
        print('%s %s days remaining' % (day, n_days_remaining))

        pv_solution_day = pv_solution.loc[day].values
        pv_intra_day = pv_intra.loc[day].values

        # BD or CCG planning
        x_algo = read_file(dir=dirname, name=day + '_x_' + str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']))

        # Point controller -> using intra pv point forecasts
        sol_controller_point = controller_loop(pv_forecast=pv_intra_day, pv_solution=pv_solution_day, engagement=x_algo, solver_param=solver_param)
        net_revenue_point = sum([sol['net_revenue'] for sol in sol_controller_point])

        # oracle controller
        planner = Planner_MILP(pv_forecast=pv_solution_day, engagement=x_algo)
        planner.solve()
        sol_oracle = planner.store_solution()
        net_revenue_oracle = -sol_oracle['obj']

        print('%s net revenue controller %.2f € oracle %.2f €' % (day, net_revenue_point, net_revenue_oracle))
        res.append([net_revenue_point, net_revenue_oracle])

        n_days_remaining += -1

    df_res = pd.DataFrame(data=res, index=day_list_sampled, columns=[algo + ' point', algo + ' oracle'])
    df_res.to_csv(dirname + 'res_controller_'+ str(warm_start)+ '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])+'.csv')