# -*- coding: UTF-8 -*-

"""
Loop over all days of the dataset using the Benders dual cutting plane (BD) or the Column-and-Constraint Generation (CCG) algorithm.
"""

import os
import random
import time
import logging
import pandas as pd
import numpy as np

from ro import build_observations, build_point_forecast, load_nf_quantiles, load_lstm_quantiles, check_BESS
from ro.determinist.algorithms import Planner_MILP, SP_primal_LP
from ro.robust.algorithms import bd_algo, ccg_algo, BD_SP
from ro.robust.algorithms.ro_params import compute_ro_dyn_params
from root_project import ROOT_DIR

from ro.ro_simulator_configuration import PARAMETERS

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

# NB periods
nb_periods = 96
N_Q = 9 # nb quantiles
q_set = [0, 1, 2, 3, 4, 5, 6, 7, 8] # -> 10%, 20%, .., 90%

# SP solver parameters
solver_param = dict()
solver_param['heuristic'] = True
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 1
solver_param['Threads'] = 1

# Convergence tolerance of the algorithm (BD or CCG) between the MP (LB) and SP (UB) objectives.
conv_tol = 0.5 # euros

# --------------------------------------
# Static RO parameters: [q_min, gamma]
GAMMA = 12 # Budget of uncertainty to specify the number of time periods where PV generation lies within the uncertainty interval: 0: to 95 -> 0 = no uncertainty
q_min = 0  # Extreme minimum quantile to define the PV uncertainty interval: 0 to 3 -> 0 = 10 % to 40 %

# Dynamic RO parameters: [d_min, d_gamma]
dyn_RO_params = False
d_gamma = 0.1 # max distance for Gamma selection
d_gamma_threshold = d_gamma * PARAMETERS["pv_capacity"]
d_pvmin = 0.1 # max distance to min PV quantile
# --------------------------------------

# --------------------------------------
# Big-M's parameters
M_neg_ini = 1 #
# Increment of big-M's value when convergence is not reached
M_neg_increment = 10
# Increment the big-M's big_M_threshold times before incrementing it of 10 * M_neg_increment
big_M_threshold = 5
max_big_M = 500
# --------------------------------------

# TEST set
nb_days = 30  # 30, 334 = total number of days of the dataset
TEST_size = str(nb_days)

printconsole = False

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

    # Create folder
    dirname = 'ro/robust/export_'+algo+'_'+q_technique+'/'
    if dyn_RO_params:
        dirname += TEST_size + '_ro_params/' + str(int(100 * d_pvmin)) + '/' + str(int(100 * d_gamma)) + '/'
    else:
        dirname += TEST_size + '_ro_static/' + str(q_min) + '/' + str(GAMMA) + '/'

    if not os.path.isdir(dirname):  # test if directory exist
        os.makedirs(dirname)

    pdfname = algo + '_' + q_technique + '_' + str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])

    log_file = dirname + "log_"+str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']) + ".log"
    logging.basicConfig(filename=log_file, level=logging.INFO)

    print('-----------------------------------------------------------------------------------------------------------')
    if dyn_RO_params:
        print('%s TEST size %s depth_threshold %s GAM_threshold %s' % (algo, TEST_size, int(100 * d_pvmin), int(100 * d_gamma)))
    else:
        print('%s TEST size %s gamma %s quantile %s' % (algo, TEST_size, GAMMA, q_min))
    print('%s: %s warm start %s PENALTY_FACTOR %s DEADBAND %s OVERPRODUCTION %s' %(algo, q_technique, warm_start, PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), PARAMETERS['OVERPRODUCTION']))
    print('-----------------------------------------------------------------------------------------------------------')

    logging.info('-----------------------------------------------------------------------------------------------------------')
    if dyn_RO_params:
        logging.info('%s TEST size %s alternative_RO %s' % (algo, TEST_size, str(dyn_RO_params)))
    else:
        logging.info('%s TEST size %s gamma %s quantile %s' % (algo, TEST_size, GAMMA, q_min))
    logging.info('%s: %s warm start %s PENALTY_FACTOR %s DEADBAND %s OVERPRODUCTION %s' %(algo, q_technique, warm_start, PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), PARAMETERS['OVERPRODUCTION']))
    logging.info('-----------------------------------------------------------------------------------------------------------')

    # --------------------------------------------------------
    # Load data
    pv_solution = build_observations() # PV observations
    pv_point_dad = build_point_forecast() # PV point forecast

    day_list = [day.strftime('%Y-%m-%d') for day in pv_point_dad.index]
    # Select randomly N days into the dataset
    random.seed(1)
    if nb_days == len(day_list):
        day_list_sampled = day_list
    else:
        day_list_sampled = random.sample(day_list, nb_days)

    # Loop over all days of the dataset
    err_MP_MILP_list = []
    t_total_list = []
    conv_not_achieved_dict = dict()
    BESS_count = dict()
    n_days_remaining = nb_days
    for day in day_list_sampled:
        print(' ')
        logging.info('  ')

        if q_technique == 'NF':
            # PV quantiles from NF
            scenarios_NF = load_nf_quantiles()
            pv_quantile_day = scenarios_NF[day].transpose()
        elif q_technique == 'LSTM':
            # PV quantiles from LSTM
            # Load quantile forecasts: from 0 to 8 -> median = quantile 4
            pv_quantile = load_lstm_quantiles()
            pv_quantile_day = pv_quantile.loc[day].values.reshape(nb_periods, N_Q)

        # Static RO
        # Compute the PV uncertainty interval [pv_min, pv_max] -> 8 = quantile 90%, 4 = quantile 50%, 0 = quantile 10%
        # In the capacity firming where curtailement is allowed, pv_max = PV quantile 50 %
        pv_min = pv_quantile_day[:, q_min]
        pv_max = pv_quantile_day[:, 4]

        # Compute the alternative RO parameters in case of dynamic RO
        if dyn_RO_params:
            worst_pv, gamma = compute_ro_dyn_params(pv_quantile=pv_quantile_day, d_gamma_threshold=d_gamma_threshold, d_pvmin=d_pvmin)
            # Update the PV uncertainty set and the budget of uncertainty Gamma
            pv_min = worst_pv
            GAMMA = gamma
            print('gamma from ro params %s' % (GAMMA))

        # Max deviation between PV max and min bounds of the uncertainty set
        max_dev = pv_max - pv_min
        max_dev[max_dev < 0] = 0

        # Compute the starting point for the first MP = day-ahead planning from the PV quantile 50 % using the MILP
        planner = Planner_MILP(pv_forecast=pv_quantile_day[:, 4])
        planner.solve()
        sol_planner_ini = planner.store_solution()
        x_ini = sol_planner_ini['x']

        # Initialize the gap between the MP and MILP
        epsilon = 1e20
        # Initialize the big-M's value and the threshold count
        M_neg = M_neg_ini
        big_M_threshold_day = 0
        # Algorithm loop for a given day of the dataset
        while epsilon > conv_tol and M_neg < max_big_M:
            print('%s: M_neg = %s %s days remaining' % (day, M_neg, n_days_remaining))
            logging.info('%s: M_neg = %s %s days remaining' % (day, M_neg, n_days_remaining))
            t_solve = time.time()

            # ------------------------------------------------------------------------------------------------------------------
            # 1. Solve for a given day and big-M's value using CCG or BD
            # ------------------------------------------------------------------------------------------------------------------
            if algo == 'BD':
                x_final, df_objectives, conv_inf = bd_algo(dir=dirname, tol=conv_tol, gamma=GAMMA, pv_min=pv_min, pv_max=pv_max, engagement=x_ini, solver_param=solver_param, day=day, printconsole=printconsole, warm_start=warm_start, M_neg=M_neg)
            elif algo =='CCG':
                # no warm start with the CCG algorithm
                x_final, df_objectives, conv_inf = ccg_algo(dir=dirname, tol=conv_tol, gamma=GAMMA, pv_min=pv_min, pv_max=pv_max, engagement=x_ini, solver_param=solver_param, day=day, printconsole=printconsole, warm_start=False, M_neg=M_neg)
            df_objectives.to_csv(dirname + day + '_obj_MP_SP_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']) + '.csv')

            # ------------------------------------------------------------------------------------------------------------------
            # 2. Get the PV worst trajectory from the last algo iteration
            # ------------------------------------------------------------------------------------------------------------------

            SP_dual = BD_SP(pv_forecast=pv_max, max_dev=max_dev, engagement=x_final, gamma=GAMMA, heuristic=solver_param['heuristic'], M_neg=M_neg)
            SP_dual.solve(LogToConsole=False, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=10)
            SP_dual_sol = SP_dual.store_solution()
            pv_trajectory_final = [pv_max[i] - SP_dual_sol['z_neg'][i] * max_dev[i] for i in range(nb_periods)]

            # ------------------------------------------------------------------------------------------------------------------
            # 3. Check the algo convergence by computing the planning using the PV worst trajectory from the last algo iteration
            # ------------------------------------------------------------------------------------------------------------------
            planner = Planner_MILP(pv_forecast=pv_trajectory_final)
            planner.solve()
            sol_planner = planner.store_solution()
            abs_gap_MP_MILP = abs(df_objectives['MP'].values[-1] - sol_planner['obj'])

            print('-----------------------%s: CONVERGENCE CHECKING------------------------------------' %(algo))
            print('Final iteration %s |MP - SP| = %.2f € SP = %.2f € MP = %.2f € MILP = %.2f €' % (len(df_objectives), abs(df_objectives['MP'].values[-1] - df_objectives['SP'].values[-1]), SP_dual_sol['obj'], df_objectives['MP'].values[-1], sol_planner['obj']))

            logging.info('-----------------------%s: CONVERGENCE CHECKING------------------------------------' %(algo))
            logging.info('Final iteration %s |MP - SP| = %.2f € SP = %.2f € MP = %.2f € MILP = %.2f €' % (len(df_objectives), abs(df_objectives['MP'].values[-1] - df_objectives['SP'].values[-1]), SP_dual_sol['obj'], df_objectives['MP'].values[-1], sol_planner['obj']))

            # ------------------------------------------------------------------------------------------------------------------
            # 4. Increment the big-M's value is convergence is not achieved. Else convergence is achieved: check BESS charge and discharge
            # ------------------------------------------------------------------------------------------------------------------
            if abs_gap_MP_MILP > conv_tol:
                print('     WARNING: %s is not converged, abs error = %.4f € -> M_neg is increased' %(algo, abs_gap_MP_MILP))
                logging.info('     WARNING: %s is not converged, abs error = %.4f € -> M_neg is increased' %(algo, abs_gap_MP_MILP))
                # increment the big-M's value
                if big_M_threshold_day < big_M_threshold:
                    M_neg += M_neg_increment
                else:
                    M_neg += M_neg_increment * 10
                big_M_threshold_day += 1
            else:
                print('     %s is converged with MP/MILP abs error %.4f €' %(algo, abs_gap_MP_MILP))
                logging.info('     %s is converged with MP/MILP abs error %.4f €' % (algo, abs_gap_MP_MILP))

                # Use the SP primal (max-min J) to compute the dispatch variables related to the last engagement computed by the MP
                SP_primal = SP_primal_LP(pv_forecast=pv_trajectory_final, engagement=x_final)
                SP_primal.solve()
                SP_primal_sol = SP_primal.store_solution()
                # Check if there is any simultaneous charge and discharge at the last Benders iteration
                nb_count = check_BESS(SP_primal_sol=SP_primal_sol)
                print('     %s last iteration %d simultaneous charge and discharge' % (algo, nb_count))
                logging.info('     %s last iteration %d simultaneous charge and discharge' % (algo, nb_count))

            # Print T CPU
            t_total = time.time() - t_solve
            print('%s t = %.1f min' % (algo, t_total / 60))
            logging.info('%s t = %.1f min' % (algo, t_total / 60))
            # update the absolute gap between the MP and MILP
            epsilon = abs_gap_MP_MILP
            # If the convergence is not achieved and the big-M's value has reached its maximum, store the last asbolute MP-MILP gap
            if M_neg > max_big_M:
                print('WARNING: %s convergence has not been achieved' %(day))
                logging.info('WARNING: %s convergence has not been achieved' %(day))
                conv_not_achieved_dict[day] = abs_gap_MP_MILP

        # store into a list the number of simultaneous charge and discharge
        BESS_count[day] = conv_inf['BESS_count']
        t_total_list.append(t_total)
        err_MP_MILP_list.append(abs_gap_MP_MILP)
        n_days_remaining += -1

    # ------------------------------------------------------------------------------------------------------------------
    # Loop over all days is terminated: post-processing
    # ------------------------------------------------------------------------------------------------------------------

    # Save the error of convergence between algo and the MILP
    print('-----------------------%s: LOOP over all days is terminated------------------------------------------------' %(algo))
    print('t total: %.1f min t per day %.1f -/+ %.1f s' %((sum(t_total_list)/60), np.mean(np.asarray(t_total_list)), np.std(np.asarray(t_total_list))))
    logging.info('-----------------------%s: LOOP over all days is terminated------------------------------------------------' %(algo))
    logging.info('t total: %.1f min t per day %.1f -/+ %.1f s' %((sum(t_total_list)/60), np.mean(np.asarray(t_total_list)), np.std(np.asarray(t_total_list))))

    df_convergence = pd.DataFrame(index=day_list_sampled, data=err_MP_MILP_list)
    df_convergence.to_csv(dirname +'conv_error_' + pdfname + '.csv')

    df_time = pd.DataFrame(index=day_list_sampled, data=t_total_list)
    df_time.to_csv(dirname +'t_CPU_' + pdfname + '.csv')

    # ------------------------------------------------------------------------------------------------------------------
    # Check if there has been any simultanenaous charge and discharge during all the iterations for all days
    # ------------------------------------------------------------------------------------------------------------------
    for day in BESS_count.keys():
        # check if there is nan value (meaning during an iteration the SP primal has not been solved because infeasible, etc)
        BESS_count_day = BESS_count[day]
        if sum(np.isnan(BESS_count_day)) > 0:
            print('WARNING %s: %s nan values' % (day, sum(np.isnan(BESS_count_day))))
            logging.info('WARNING %s: %s nan values' % (day, sum(np.isnan(BESS_count_day))))

        # “python list replace nan with 0” Code
        BESS_count_day = [0 if x != x else x for x in BESS_count_day]
        if sum(BESS_count_day) > 0:
            print('WARNING %s: %d total charge and discharge over all iterations' %(day, sum(BESS_count_day)))
            logging.info('WARNING %s: %d total charge and discharge over all iterations' %(day, sum(BESS_count_day)))
        # else:
        #     print('%s: no simultaneous charge and discharge over all iterations' %(day))

    for day in conv_not_achieved_dict:
        print('WARNING: convergence has not been achieved for day %s with an absolute MILP-MP gap %.4f' %(day, conv_not_achieved_dict[day]))
        logging.info('WARNING: convergence has not been achieved for day %s with an absolute MILP-MP gap %.4f' %(day, conv_not_achieved_dict[day]))

    import matplotlib.pyplot as plt

    FONTSIZE = 20
    plt.figure()
    plt.plot(df_convergence.values, linewidth=3, label='abs err')
    plt.hlines(y=conv_tol, xmin=0, xmax=df_convergence.values.shape[0], label='conv tol')
    plt.ylim(0, int(df_convergence.values.max()+1))
    plt.ylabel('€', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + 'convergence_' + pdfname+'.pdf')
    plt.close('all')

    import numpy as np
    plt.figure()
    plt.plot(t_total_list, linewidth=3, label='t')
    plt.ylim(0, int(max(t_total_list)+1))
    plt.hlines(y=np.mean(t_total_list), xmin=0, xmax=len(t_total_list), label='av')
    plt.ylabel('s', fontsize=FONTSIZE, rotation='horizontal')
    plt.xlabel('day', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + 'time_'+ pdfname+'.pdf')
    plt.close('all')