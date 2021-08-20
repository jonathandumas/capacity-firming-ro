# -*- coding: UTF-8 -*-

import os
import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt

from ro.determinist.algorithms.SP_dual_LP import SP_dual_LP
from ro.robust.algorithms import BD_MP, BD_SP
from ro.utils import load_lstm_quantiles, build_observations, build_point_forecast, build_point_intra_forecast
from root_project import ROOT_DIR

from ro.determinist.algorithms.planner_MILP import Planner_MILP
from ro.ro_simulator_configuration import PARAMETERS

def compute_pv_trajectories(pv_min: np.array, pv_max: np.array, gamma: int, printconsole:bool=False):
    """
    ONLY for the Benders dual cutting plane algorithm
    Compute PV trajectories between two PV bounds (min & max).
    1. By using a sliding window of length GAMMA time periods and set PV max bound to PV min bound values.
        -> The window is moving time step per time step from the first to the last non null PV max bound value.
    2. By selecting the GAMMA PV max quantile values and set them to PV min quantile values.
    Note: for the CCG we use only 2. because due to the high number of PV trajectories, the MP will be too big.
    :param pv_min: PV min bound of the uncertainty set.
    :param pv_max: PV max bound of the uncertainty set.
    :return: PV trajectories into a dict.
    """

    pv_traj_dict = dict()
    # 1. Sample trajectories by using a sliding window of length GAMMA
    first_no_null = next(val for val in pv_max.tolist() if val)
    last_no_null = next(val for val in pv_max.tolist()[::-1] if val)
    i_first_no_null = np.where(pv_max == first_no_null)[0][0]
    i_last_no_null = np.where(pv_max == last_no_null)[0][0]
    nb_traj = i_last_no_null - (i_first_no_null + gamma - 1)

    # there is at least one window
    if nb_traj > 0:
        # compute trajectories
        for i in range(0, nb_traj):
            pv_traj_dict[i] = pv_max.copy()  # set all values to the quantile 50 %
            # sliding window
            for j in range(i_first_no_null + i, i_first_no_null + gamma + i):
                pv_traj_dict[i][j] = pv_min[j]

        # 2. Sample trajectories by selecting the GAMMA PV max quantile values
        pv_traj_max = pv_max.copy()  # set all values to the quantile 50 %
        i_max_gamma = pv_max.argsort()[-gamma:]
        for i in i_max_gamma:
            pv_traj_max[i] = pv_min[i]
        pv_traj_dict[nb_traj] = pv_traj_max

        # 3. check if the GAMMA max trajectory is not already sampled
        df_pv_traj = pd.DataFrame(pv_traj_dict)
        true_list = []
        for col in df_pv_traj.columns:
            comparison_column = np.where(df_pv_traj[nb_traj] == df_pv_traj[col], True, False)
            true_list.append(all(comparison_column))
        if True in true_list:
            if printconsole:
                print('Max trajectory is already into the sample')
            pv_traj_dict.pop(nb_traj)

    # case where there is not window
    else:
        # 2. Sample trajectories by selecting the GAMMA PV max quantile values
        pv_traj_max = pv_max.copy()  # set all values to the quantile 50 %
        i_max_gamma = pv_max.argsort()[-gamma:]
        for i in i_max_gamma:
            pv_traj_max[i] = pv_min[i]
        pv_traj_dict[nb_traj] = pv_traj_max

    if printconsole:
        print('%s PV trajectories' %(len(pv_traj_dict)))

    return pv_traj_dict

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

FONTSIZE = 20

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

# Budget of uncertainty
GAMMA = 24

# Select the day
day_list = ['2019-08-04']
day = day_list[0]

# Select the extreme quantile
quantile = 0 # from 0 to 3

M_neg = 1

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    print('%s gamma %s quantile %s' %(day, GAMMA, quantile))
    print('PENALTY_FACTOR %s DEADBAND %s (percentage)' %(PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty'])))

    # Create folder
    dirname = 'ro/robust/export_test_warm_start/'+day+'_quantile_' + str(quantile) + '/gamma_' + str(GAMMA) + '/'
    if not os.path.isdir(dirname):  # test if directory exist
        os.makedirs(dirname)

    # load data
    pv_solution = build_observations()
    pv_dad = build_point_forecast()
    pv_intra = build_point_intra_forecast()
    # Load quantile forecasts: from 0 to 8 -> median = quantile 4
    pv_quantile = load_lstm_quantiles()

    # Select a paticular day of the dataset
    pv_solution_day = pv_solution.loc[day].values
    pv_point_day = pv_dad.loc[day].values
    pv_intra_day = pv_intra.loc[day].values
    pv_quantile_day = pv_quantile.loc[day].values.reshape(nb_periods, N_Q)

    # Store the forecasts into a dict
    pv_forecast_dict = dict()
    pv_forecast_dict['oracle'] = pv_solution_day
    pv_forecast_dict['point'] = pv_point_day
    for q in q_set:
        pv_forecast_dict[q] = pv_quantile_day[:, q]  # 10, 20, 30, 40, etc

    # Compute the min and max deviation -> 0 = quantile 90%, 4 = quantile 50%, 8 = quantile 10%
    dev_min = pv_quantile_day[:,4] - pv_quantile_day[:, quantile]

    # ------------------------------------------------------------------------------------------------------------------
    # Sample PV trajectories
    # ------------------------------------------------------------------------------------------------------------------

    pv_traj_dict = compute_pv_trajectories(pv_min=pv_forecast_dict[quantile], pv_max=pv_forecast_dict[4], gamma=GAMMA)

    # ------------------------------------------------------------------------------------------------------------------
    # Compute the cuts corresponding to the trajectories
    # ------------------------------------------------------------------------------------------------------------------

    # Init the MP
    MP = BD_MP()
    MP.model.update()
    MP.export_model(dirname + 'MP_ini')

    # Loop on all PV trajectories
    objectives = []
    for i in pv_traj_dict:
        # 1. Compute the planning using the MILP planner
        planner = Planner_MILP(pv_forecast=pv_traj_dict[i])
        # planner = Planner_LP(pv_forecast=pv_traj_dict[i])
        planner.solve()
        sol_planner = planner.store_solution()
        print('Trajectory %s planner obj %.2f €' % (i, sol_planner['obj']))

        # 2. Compute the cut using the dual of the SP based on the planning from the MILP planner
        SP_dual = SP_dual_LP(pv_forecast=pv_traj_dict[i], engagement=sol_planner['x'])
        SP_dual.solve()
        SP_dual_sol = SP_dual.store_solution()
        print('Trajectory %s dual SP obj %.2f €' % (i, SP_dual_sol['obj']))

        # 3. compute the cut and add to the MP
        lin_exp = gp.quicksum(SP_dual_sol['phi_devlong'][i] * MP.model.getVars()[i] - SP_dual_sol['phi_devshort'][i] * MP.model.getVars()[i] for i in range(nb_periods))  # x1 to x96 are the 0th to the 95th variables of the MP model
        MP.model.addConstr(MP.model.getVars()[-1] >= SP_dual_sol['cut']  + lin_exp, name='c_cut_trajectory_'+str(i))  # theta is the last variable of the MP

        MP.solve()
        MP_sol = MP.store_solution()
        print('Trajectory %s MP obj %.2f €' % (i, MP_sol['obj']))

        objectives.append([MP_sol['obj'], SP_dual_sol['obj'], sol_planner['obj']])


    # Export final MP
    MP.export_model(dirname + 'MP_warm_started')

    plt.figure(figsize=(10, 10))
    plt.plot(MP_sol['x'], linewidth=3, label='x MP')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.tight_layout()
    plt.legend(fontsize=FONTSIZE)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(-np.asarray(objectives)[:,0], linewidth=3,label='MP')
    plt.plot(-np.asarray(objectives)[:, 1], linewidth=3,label='SP dual')
    plt.plot(-np.asarray(objectives)[:, 2], linewidth=3,label='MILP')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('trajectory', fontsize=FONTSIZE, rotation='horizontal')
    plt.ylabel('euros', fontsize=FONTSIZE, rotation='horizontal')
    plt.tight_layout()
    plt.legend(fontsize=FONTSIZE, ncol=2)
    plt.show()

    # 1. Compute the planning using the MILP planner
    planner = Planner_MILP(pv_forecast=pv_forecast_dict[4])
    planner.solve()
    sol_planner = planner.store_solution()
    print('Quantile 50 planner obj %.2f €' % (sol_planner['obj']))

    SP = BD_SP(pv_forecast=pv_forecast_dict[4], max_dev=dev_min, engagement=sol_planner['x'],
               gamma=GAMMA, heuristic=solver_param['heuristic'], M_neg=M_neg)
    SP.solve(Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'],
             TimeLimit=solver_param['TimeLimit'])
    SP_dual_sol = SP.store_solution()
    print('x from planner quantile 50 SP dual obj %.2f €' % (SP_dual_sol['obj']))

    SP = BD_SP(pv_forecast=pv_forecast_dict[4], max_dev=dev_min, engagement=MP_sol['x'],
               gamma=GAMMA, heuristic=solver_param['heuristic'], M_neg=M_neg)
    SP.solve(Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'],
             TimeLimit=solver_param['TimeLimit'])
    SP_dual_sol = SP.store_solution()
    print('x from MP SP dual obj %.2f €' % (SP_dual_sol['obj']))
