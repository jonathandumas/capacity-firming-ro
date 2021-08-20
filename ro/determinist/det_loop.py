# -*- coding: UTF-8 -*-

"""
Loop over all days of the dataset using a deterministic planner and controller.
"""

import os
import random
import pandas as pd
import numpy as np

from root_project import ROOT_DIR
from ro.robust.algorithms.ro_params import compute_ro_dyn_params

from ro.determinist.algorithms import Planner_MILP, controller_loop
from ro import build_observations, build_point_forecast, build_point_intra_forecast, load_lstm_quantiles, read_file, \
    load_nf_quantiles
from ro.ro_simulator_configuration import PARAMETERS

# Solver parameters
solver_param = dict()
solver_param['heuristic'] = True
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 10
solver_param['Threads'] = 1

# NB periods
nb_periods = 96
N_Q = 9
q_set = [0, 1, 2, 3, 4, 5, 6, 7, 8] # -> 10%, 20%, .., 90%
# Validation set
nb_days = 30  # 30, 334 = total number of days of the dataset
TEST_size = str(nb_days) #

# Alternative RO
d_pvmin_set = [0.05, 0.1, 0.2, 0.3, 0.5]
dyn_RO_params = False

# Forecasting technique to compute the PV quantiles
q_technique = 'NF' # 'NF', 'LSTM'

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    print('-----------------------------------------------------------------------------------------------------------')
    print('Controller loop TEST size %s %s dyn_RO_params %s' % (TEST_size, dyn_RO_params, q_technique))
    print('PENALTY_FACTOR %s DEADBAND %s MIN_PROD %s OVERPRODUCTION %s' %(PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), int(100 * PARAMETERS['min_production']), PARAMETERS['OVERPRODUCTION']))
    print('-----------------------------------------------------------------------------------------------------------')

    # Create folder
    # Create folder
    dirname = 'ro/determinist/export_'+q_technique+'/'
    if dyn_RO_params:
        dirname += TEST_size + '_ro_dyn/'
        config = ['oracle'] + d_pvmin_set
        # Create csv name
        csv_name = 'det_point_alt_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty']))
        csv_name_oracle = 'det_oracle_alt_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty']))
    else:
        dirname += TEST_size + '_ro_static/'
        config = ['oracle', 'point'] + q_set  # ['oracle', 'point'] + q_set
        # Create csv name
        csv_name = 'det_point_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty']))
        csv_name_oracle = 'det_oracle_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty']))

    if not os.path.isdir(dirname):  # test if directory exist
        os.makedirs(dirname)

    # --------------------------------------------------------
    # Load data
    pv_solution = build_observations()
    pv_dad = build_point_forecast()
    pv_intra = build_point_intra_forecast()
    day_list = [day.strftime('%Y-%m-%d') for day in pv_intra.index]
    random.seed(1)
    if nb_days == len(day_list):
        day_list_sampled = day_list
    else:
        day_list_sampled = random.sample(day_list, nb_days)

    res_list = []
    res_oracle_list = []
    # Loop on each day of the dataset
    n_days_remaining = len(day_list_sampled)
    for day in day_list_sampled:
        print('day %s %s days remaining' % (day, n_days_remaining))
        pv_solution_day = pv_solution.loc[day].values
        pv_dad_day = pv_dad.loc[day].values
        pv_intra_day = pv_intra.loc[day].values

        if q_technique == 'NF':
            # PV quantiles from NF
            scenarios_NF = load_nf_quantiles()
            pv_quantile_day = scenarios_NF[day].transpose()
        elif q_technique == 'LSTM':
            # PV quantiles from LSTM
            # Load quantile forecasts: from 0 to 8 -> median = quantile 4
            pv_quantile = load_lstm_quantiles()
            pv_quantile_day = pv_quantile.loc[day].values.reshape(nb_periods, N_Q)

        # Store the forecasts into a dict
        pv_forecast = dict()
        for q in q_set:
            pv_forecast[q] = pv_quantile_day[:, q] # 10, 20, 30, 40, etc
        pv_forecast['oracle'] = pv_solution_day
        pv_forecast['point'] = pv_dad_day

        # Alternative RO parameters
        for d_pvmin in d_pvmin_set:
            pv_forecast[d_pvmin], gamma = compute_ro_dyn_params(pv_quantile=pv_quantile_day, d_gamma_threshold=0.1, d_pvmin=d_pvmin)

        # Compute several engagement plans by solving the MILP formulation with different point forecasts
        sol_planner = dict()
        for key in config:
            planner = Planner_MILP(pv_forecast=pv_forecast[key])
            planner.solve()
            sol_planner[key] = planner.store_solution()['x']
            if key == 'oracle':
                print('%s %s net revenue %.2f (€)' % (day, key, - planner.store_solution()['obj']))

        # Point controller -> using intra pv point forecasts
        res_point_controller = dict()
        for key in config:
            sol_controller_point = controller_loop(pv_forecast=pv_intra_day, pv_solution=pv_solution_day, engagement=sol_planner[key], solver_param=solver_param)

            res_point_controller[key] = pd.DataFrame()
            for col in ['net_revenue', 'y_prod', 'y_short_dev', 'y_long_dev', 'y_s', 'y_charge', 'y_discharge', 'y_PV', 'time_total']:
                res_point_controller[key][col] = [sol[col] for sol in sol_controller_point]
            res_point_controller[key]['penalty'] = [sol['y_short_dev'] + sol['y_long_dev'] for sol in sol_controller_point]

            print('%s %s-point net revenue %.2f (€)' % (day, key, res_point_controller[key]['net_revenue'].sum()))
        res_list.append([res_point_controller[key]['net_revenue'].sum() for key in config])

        # Oracle controller -> using PV observations
        res_oracle_controller = dict()
        for key in config:
            planner = Planner_MILP(pv_forecast=pv_forecast['oracle'], engagement=sol_planner[key])
            planner.solve()
            res_oracle_controller[key] = - planner.store_solution()['obj']

            print('%s %s-oracle net revenue %.2f (€)' % (day, key, res_oracle_controller[key]))
        res_oracle_list.append([res_oracle_controller[key] for key in config])

        n_days_remaining += -1

    df_res_controller_point = pd.DataFrame(data=res_list, index=day_list_sampled, columns=config)
    df_res_controller_point.to_csv(dirname + csv_name + '.csv')
    df_res_controller_oracle = pd.DataFrame(data=res_oracle_list, index=day_list_sampled, columns=config)
    df_res_controller_oracle.to_csv(dirname + csv_name_oracle + '.csv')

    # -----------------------------------------------------------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------------------------------------------------------

    print('-----------------------------------------------------------------------------------------------------------')
    print('TEST size %s PENALTY_FACTOR %s DEADBAND %s MIN_PROD %s OVERPRODUCTION %s dyn_RO_params %s' % (len(day_list_sampled), PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), int(100 * PARAMETERS['min_production']), PARAMETERS['OVERPRODUCTION'], dyn_RO_params))

    print('-----------------------------------------------------------------------------------------------------------')
    print('Oracle controller')
    print('-----------------------------------------------------------------------------------------------------------')
    for key in config:
        print('%s-oracle %.2f (k€) %.2f' % (key, df_res_controller_oracle[key].sum()/1000, 100* df_res_controller_oracle[key].sum()/df_res_controller_oracle.sum()['oracle']))

    print('-----------------------------------------------------------------------------------------------------------')
    print('Point controller')
    print('-----------------------------------------------------------------------------------------------------------')
    for key in config:
        print('%s-point %.2f (k€) %.2f of oracle-point %.2f of oracle-oracle' % (key, df_res_controller_point[key].sum()/1000, 100* df_res_controller_point[key].sum()/df_res_controller_point.sum()['oracle'], 100* df_res_controller_point[key].sum()/df_res_controller_oracle.sum()['oracle']))
