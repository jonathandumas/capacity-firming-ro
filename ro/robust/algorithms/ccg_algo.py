# -*- coding: UTF-8 -*-

"""
Column-and-Constraint Generation (CCG) algorithm to solve a two-stage robust optimization problem in the capacity firming framework.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ro.robust.algorithms import CCG_MP, BD_SP
from root_project import ROOT_DIR
from ro import check_BESS, dump_file, load_nf_quantiles, build_observations, build_point_forecast, load_lstm_quantiles
from ro.robust.algorithms.ro_params import compute_ro_dyn_params

from ro.determinist.algorithms.SP_primal_LP import SP_primal_LP
from ro.determinist.algorithms.planner_MILP import Planner_MILP
from ro.ro_simulator_configuration import PARAMETERS

def ccg_algo(dir:str, tol: float, gamma: int, pv_min: np.array, pv_max: np.array, engagement: np.array, solver_param: dict, day:str, log:bool=False, printconsole:bool=False, warm_start:bool=False, M_neg:float=None):
    """
    CCG = Column-and-Constraint Generation
    Column-and-Constraint Generation algorithm.
    Iteration between the MP and SP until convergence criteria is reached.
    :param tol: convergence tolerance.
    :param gamma: budget of uncertainty.
    :param pv_min: PV min bound of the uncertainty set (kW).
    :param pv_max: PV max bound of the uncertainty set (kW).
    :param engagement: initial CCG starting point.
    :param solver_param: Gurobi solver parameters.
    :return: the final engagement plan when the convergence criteria is reached and some data1.
    """

    # Compute the maximal deviation between the max and min PV uncertainty set bounds
    max_dev = pv_max - pv_min # (kW)
    max_dev[max_dev < 0] = 0
    nb_periods = max_dev.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    # CCG initialization: build the initial MP
    # ------------------------------------------------------------------------------------------------------------------

    # Building the MP
    MP = CCG_MP()
    MP.model.update()
    print('MP initialized: %d variables %d constraints' % (len(MP.model.getVars()), len(MP.model.getConstrs())))
    MP.export_model(dir + day + '_ccg_MP_initialized')

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop until convergence criteria is reached
    # ------------------------------------------------------------------------------------------------------------------

    if printconsole:
        print('-----------CCG ITERATION STARTING-----------')

    t_solve = time.time()
    objectives = []
    computation_times = []
    mipgap = []
    SP_dual_status = []
    SP_primal_status = []
    alpha_neg_list = []
    epsilon = 1e20
    # With CCG the convergence is stable.
    epsilon_list = [epsilon] * 2
    iteration = 1
    BESS_count_list = []
    BESS_charge_discharge_list = []
    max_iteration = 50

    while all(i < tol for i in epsilon_list) is not True and iteration < max_iteration:
        logfile = ""
        if log:
            logfile = dir + 'logfile_' + str(iteration) + '.log'
        if printconsole:
            print('i = %s solve SP dual' % (iteration))

        # ------------------------------------------------------------------------------------------------------------------
        # 1. SP part
        # ------------------------------------------------------------------------------------------------------------------

        # 1.1 Solve the SP and get the worst PV trajectory to add the new constraints of the MP
        SP_dual = BD_SP(pv_forecast=pv_max, max_dev=max_dev, engagement=engagement, gamma=gamma, heuristic=solver_param['heuristic'], M_neg=M_neg)
        SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
        SP_dual_sol = SP_dual.store_solution()
        SP_dual_status.append(SP_dual_sol['status'])
        mipgap.append(SP_dual.model.MIPGap)
        alpha_neg_list.append(SP_dual_sol['alpha_neg'])

        # 1.2 Compute the worst PV trajectory from the SP dual solution
        pv_worst_case_from_SP = [pv_max[i] - SP_dual_sol['z_neg'][i] * max_dev[i] for i in range(nb_periods)]
        if printconsole:
            print('     i = %s : SP dual status %s solved in %.1f s MIPGap = %.6f' % (iteration, SP_dual_sol['status'], SP_dual_sol['time_total'], SP_dual.model.MIPGap))

        # 1.3 Solve the primal of the SP to check if the objectives of the primal and dual are equal to each other
        SP_primal = SP_primal_LP(pv_forecast=pv_worst_case_from_SP, engagement=engagement)
        SP_primal.solve()
        SP_primal_sol = SP_primal.store_solution()
        SP_primal_status.append(SP_primal_sol['status'])

        if printconsole:
            print('     i = %s : SP primal status %s' % (iteration, SP_primal_sol['status']))
            print('     i = %s : SP primal %.1f € SP dual %.1f € -> |SP primal - SP dual| = %.2f €' % (iteration, SP_primal_sol['obj'], SP_dual_sol['obj'], abs(SP_primal_sol['obj'] - SP_dual_sol['obj'])))

        # 1.4 SP solved to optimality ? -> Check if there is any simultaneous charge and discharge in the SP primal solution
        if SP_primal_sol['status'] == 2 or SP_primal_sol['status'] == 9: # 2 = optimal, 9 = timelimit has been reached
            nb_count = check_BESS(SP_primal_sol=SP_primal_sol)
            if nb_count > 0:
                BESS_charge_discharge_list.append([iteration, SP_primal_sol['y_charge'], SP_primal_sol['y_discharge']])
        else: #
            nb_count = float('nan')
        BESS_count_list.append(nb_count)
        if printconsole:
            print('     i = %s : %s simultaneous charge and discharge' % (iteration, nb_count))

        # ------------------------------------------------------------------------------------------------------------------
        # 2. MP part
        # ------------------------------------------------------------------------------------------------------------------

        # Check Sub Problem status -> bounded or unbounded
        if SP_dual_sol['status'] == 2 or SP_dual_sol['status'] == 9:  # 2 = optimal, 9 = timelimit has been reached
            # Add an optimality cut to MP and solve
            MP.update_MP(pv_trajectory=pv_worst_case_from_SP, iteration=iteration)
            if printconsole:
                print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
            # MP.export_model(dir + 'MP_' + str(iteration))
            if printconsole:
                print('i = %s : solve MP' % (iteration))
            MP.solve()
            MP_sol = MP.store_solution()
            MP.update_sol(MP_sol=MP_sol, i=iteration)
            if MP_sol['status'] == 3 or MP_sol['status'] == 4:
                print('i = %s : WARNING MP status %s -> Create a new MP, increase big-M value and compute a new PV trajectory from SP' % (iteration, MP_sol['status']))

                # MP unbounded or infeasible -> increase big-M's value to get another PV trajectory from the SP
                SP_dual = BD_SP(pv_forecast=pv_max, max_dev=max_dev, engagement=engagement, gamma=gamma, heuristic=solver_param['heuristic'], M_neg=M_neg+50)
                SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'],
                              TimeLimit=solver_param['TimeLimit'])
                SP_dual_sol = SP_dual.store_solution()

                # Compute a new worst PV trajectory from the SP dual solution
                pv_worst_case_from_SP = [pv_max[i] - SP_dual_sol['z_neg'][i] * max_dev[i] for i in range(nb_periods)]

                # Create a new MP
                MP = CCG_MP()
                MP.model.update()
                MP.update_MP(pv_trajectory=pv_worst_case_from_SP, iteration=iteration)
                if printconsole:
                    print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
                # MP.export_model(dir + 'MP_' + str(iteration))
                if printconsole:
                    print('i = %s : solve new MP' % (iteration))
                MP.solve()
                MP_sol = MP.store_solution()
                MP.update_sol(MP_sol=MP_sol, i=iteration)

            computation_times.append([SP_dual_sol['time_total'], MP_sol['time_total']])


        else: # 4 = Model was proven to be either infeasible or unbounded.
            print('SP is unbounded: a feasibility cut is required to be added to the Master Problem')

        objectives.append([iteration, MP_sol['obj'], SP_dual_sol['obj'], SP_primal_sol['obj']])

        # ------------------------------------------------------------------------------------------------------------------
        # 3. Update: the engagement, lower and upper bounds using the updated MP
        # ------------------------------------------------------------------------------------------------------------------

        # Solve the MILP with the worst case trajectory
        planner = Planner_MILP(pv_forecast=pv_worst_case_from_SP)
        planner.solve()
        sol_planner = planner.store_solution()

        # Update engagement
        engagement = MP_sol['x']
        # Update the lower and upper bounds
        # MP -> give the lower bound
        # SP -> give the upper bound
        epsilon = abs(MP_sol['obj'] - SP_dual_sol['obj'])
        print('i = %s : |MP - SP dual| = %.2f €' % (iteration, epsilon))
        abs_err = abs(MP_sol['obj'] - sol_planner['obj'])
        epsilon_list.append(epsilon)
        epsilon_list.pop(0)
        if printconsole:
            print('i = %s : MP %.2f € SP dual %.2f € -> |MP - SP dual| = %.2f €' % (iteration, MP_sol['obj'], SP_dual_sol['obj'], epsilon))
            print('i = %s : MP %.2f € MILP %.2f € -> |MP - MILP| = %.2f €' % (iteration, MP_sol['obj'], sol_planner['obj'], abs_err))
            print(epsilon_list)
            print('                                                                                                       ')

        iteration += 1

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop terminated
    # ------------------------------------------------------------------------------------------------------------------
    if printconsole:
        print('-----------CCG ITERATION TERMINATED-----------')
    print('Final iteration  = %s : MP %.2f € SP dual %.2f € -> |MP - SP dual| = %.2f €' % (iteration-1, MP_sol['obj'], SP_dual_sol['obj'], epsilon))

    # Export last MP
    MP.export_model(dir + day + '_MP_' + str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']))

    # MP.model.printStats()

    # Dump last engagement plan at iteration
    dump_file(dir=dir, name=day+'_x_' + str(warm_start)+ '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']), file=engagement)

    # Print T CPU
    t_total = time.time() - t_solve
    computation_times = np.asarray(computation_times)
    SP_dual_status = np.asarray(SP_dual_status)
    SP_primal_status = np.asarray(SP_primal_status)

    if printconsole:
        print('Total CCG loop t CPU %.1f min' % (t_total / 60))
        print('T CPU (s): Sup Problem max %.1f Master Problem max %.1f' % (computation_times[:, 0].max(), computation_times[:, 1].max()))
        print('nb Sup Problem status 2 %d status 9 %d' % (SP_dual_status[SP_dual_status == 2].shape[0], SP_dual_status[SP_dual_status == 9].shape[0]))

    # Store data
    objectives = np.asarray(objectives)
    df_objectives = pd.DataFrame(index=objectives[:, 0], data=objectives[:, 1:], columns=['MP', 'SP', 'SP_primal'])

    # store convergence information
    conv_inf = dict()
    conv_inf['mipgap'] = mipgap
    conv_inf['computation_times'] = computation_times
    conv_inf['SP_status'] = SP_dual_status
    conv_inf['SP_primal_status'] = SP_primal_status
    conv_inf['alpha_neg'] = alpha_neg_list
    conv_inf['BESS_count'] = BESS_count_list
    conv_inf['BESS_charge_discharge'] = BESS_charge_discharge_list

    return engagement, df_objectives, conv_inf

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

FONTSIZE = 20

# NB periods
nb_periods = 96
N_Q = 9
q_set = [0, 1, 2, 3, 4, 5, 6, 7, 8] # -> 10%, 20%, .., 90%

# Solver parameters
solver_param = dict()
solver_param['heuristic'] = True
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 10
solver_param['Threads'] = 1

# Convergence threshold between MP and SP objectives
conv_tol = 0.5
printconsole = True

# Select the day
day_list = ['2019-09-14', '2020-05-26',  '2020-09-26', '2019-08-05']
day = day_list[0]
day = '2019-10-21'

# --------------------------------------
# Static RO parameters: [q_min, gamma]
GAMMA = 12 # Budget of uncertainty to specify the number of time periods where PV generation lies within the uncertainty interval: 0: to 95 -> 0 = no uncertainty
q_min = 0  # Extreme minimum quantile to define the PV uncertainty interval: 0 to 3 -> 0 = 10 % to 40 %

# Dynamic RO parameters: [d_min, d_gamma]
dyn_RO_params = False
d_gamma = 0.1 # max distance for Gamma selection
gamma_threshold = d_gamma * PARAMETERS["pv_capacity"]
d_pvmin = 0.1 # max distance to min PV quantile
# --------------------------------------

# warm_start
warm_start = False
M_neg = 1

# quantile from NF or LSTM
NF_quantile = True

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    # Create folder
    dirname = 'ro/robust/export_test_ccg/'
    if NF_quantile:
        dirname += 'NF/'
        scenarios_NF = load_nf_quantiles()

    if dyn_RO_params:
        dirname += 'ro_params/' + day + '/'+str(int(100 * d_pvmin)) + '/' + str(int(100 * d_gamma)) + '/'
        pdfname = str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])
    else:
        dirname += day + '_' + str(q_min) + '/' + str(GAMMA) + '/'
        pdfname = str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']) + '_' + str(q_min) + '_' + str(GAMMA)

    if not os.path.isdir(dirname):  # test if directory exist
        os.makedirs(dirname)

    print('-----------------------------------------------------------------------------------------------------------')
    if dyn_RO_params:
        print('CCG: day %s depth_threshold %s d_gamma %s' % (day, int(100 * d_pvmin),  int(100 * d_gamma)))
    else:
        print('CCG: day %s gamma %s quantile %s' % (day, GAMMA, d_pvmin))
    print('NF_quantile %s PENALTY_FACTOR %s DEADBAND %s OVERPRODUCTION %s' %(NF_quantile, PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), PARAMETERS['OVERPRODUCTION']))
    print('CCG warm start %s' %(warm_start))
    print('-----------------------------------------------------------------------------------------------------------')


    # load data
    pv_solution = build_observations()
    pv_dad = build_point_forecast()
    # Load quantile forecasts: from 0 to 8 -> median = quantile 4
    pv_quantile = load_lstm_quantiles()

    # Select a paticular day of the dataset
    pv_solution_day = pv_solution.loc[day].values
    pv_point_day = pv_dad.loc[day].values
    if NF_quantile:
        # quantile from NF
        pv_quantile_day = scenarios_NF[day].transpose()
    else:
        # quantile from LSTM
        pv_quantile_day = pv_quantile.loc[day].values.reshape(nb_periods, N_Q)

    # PLOT quantile, point, and observations
    x_index = [i for i in range(0, nb_periods)]
    plt.figure()
    for j in range(1, N_Q // 2 + 1):
        plt.fill_between(x_index, pv_quantile_day[:, j + N_Q // 2], pv_quantile_day[:, (N_Q // 2) - j], alpha=0.5 / j,
                         color=(1 / j, 0, 1))
    plt.plot(x_index, pv_quantile_day[:, 4], 'b', linewidth=3, label='50 %')
    plt.plot(x_index, pv_solution_day, 'r', linewidth=3, label='obs')
    plt.plot(x_index, pv_point_day, 'k--', linewidth=3, label='nominal')
    plt.ylim(0, PARAMETERS["pv_capacity"])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname +str(day) + '_quantile_forecast.pdf')
    plt.close('all')

    # Store the forecasts into a dict
    pv_forecast_dict = dict()
    pv_forecast_dict['oracle'] = pv_solution_day
    pv_forecast_dict['point'] = pv_point_day
    for q in q_set:
        pv_forecast_dict[q] = pv_quantile_day[:, q]  # 10, 20, 30, 40, etc

    # Compute the starting point for the first MP = day-ahead planning from the PV quantile 50 % using the MILP
    planner = Planner_MILP(pv_forecast=pv_quantile_day[:, 4])
    planner.solve()
    sol_planner_ini = planner.store_solution()
    x_ini = sol_planner_ini['x']

    # Compute the min and max deviation -> 8 = quantile 90%, 4 = quantile 50%, 0 = quantile 10%
    pv_min = pv_quantile_day[:, q_min]
    pv_max = pv_quantile_day[:, 4]

    if dyn_RO_params:
        # Alternative RO parameters
        worst_pv, gamma = compute_ro_dyn_params(pv_quantile=pv_quantile_day, d_gamma_threshold=gamma_threshold, d_pvmin=d_pvmin)
        pv_min = worst_pv
        GAMMA = gamma
        print('gamma from ro params %s' % (GAMMA))

    # Max deviation between PV max and min bounds of the uncertainty set
    max_dev = pv_max - pv_min
    max_dev[max_dev < 0] = 0

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop
    # ------------------------------------------------------------------------------------------------------------------
    final_engagement, df_objectives, conv_inf = ccg_algo(dir=dirname, tol=conv_tol, gamma=GAMMA, pv_min=pv_min, pv_max=pv_max, engagement=x_ini, solver_param=solver_param, day=day, printconsole=printconsole, M_neg=M_neg)
    df_objectives.to_csv(dirname + day+'_obj_MP_SP_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])+'.csv')

    if dyn_RO_params:
        print(
            '-----------------------------------------------------------------------------------------------------------')
        print('CCG: %s %s gamma %s alternative_RO %s' % (day, GAMMA, dyn_RO_params))
        print(
            '-----------------------------------------------------------------------------------------------------------')
    else:
        print('-----------------------------------------------------------------------------------------------------------')
        print('CCG: %s gamma %s quantile %s' %(day, GAMMA, q_min))
        print('-----------------------------------------------------------------------------------------------------------')

    # ------------------------------------------------------------------------------------------------------------------
    # Get the final worst case PV generation trajectory computed by the Sub Problem
    # ------------------------------------------------------------------------------------------------------------------

    # Get the worst case related to the last engagement plan by using the Sub Problem dual formulation
    SP_dual = BD_SP(pv_forecast=pv_max, max_dev=max_dev, engagement=final_engagement, gamma=GAMMA, heuristic=solver_param['heuristic'], M_neg=M_neg)
    SP_dual.solve(LogToConsole=False, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=10)
    SP_dual_sol = SP_dual.store_solution()
    # Compute the worst PV path from the SP dual solution
    pv_worst_case = [pv_max[i] - SP_dual_sol['z_neg'][i] * max_dev[i] for i in range(nb_periods)]
    dump_file(dir=dirname, name=day+'_pv_worst_case_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']), file=pv_worst_case)

    # Check if the worst PV path is on the extreme quantile
    if sum(SP_dual_sol['z_neg']) == GAMMA:
        print('Worst PV path is the extreme min quantile')
    else:
        print('%d points on extreme min quantile, %d points on median' % (sum(SP_dual_sol['z_neg']), GAMMA - sum(SP_dual_sol['z_neg'])))

    plt.figure()
    for j in range(1, 5 + 1):
        plt.plot(x_index, pv_quantile_day[:, 5 - j], alpha=1 / j, color=(1 / j, 0, 1), linewidth=3)
    plt.plot(pv_worst_case, 'r.', markersize=10, linewidth=3, label='PV traj')
    plt.plot(x_index, pv_min, 'g', alpha=0.5, linewidth=3, label='PV min')
    plt.plot(x_index, pv_solution_day, 'k', linewidth=1, label='obs')
    plt.plot(x_index, pv_point_day, ':', linewidth=3, label='point')
    plt.ylim(0, PARAMETERS["pv_capacity"])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title(day)
    plt.tight_layout()
    plt.savefig(dirname +  day + '_PV_trajectory_' + pdfname +'.pdf')
    plt.close('all')

    # ------------------------------------------------------------------------------------------------------------------
    # Second-stage variables comparison: y_prod, s, etc
    # ------------------------------------------------------------------------------------------------------------------

    # Use the SP primal (SP worst case dispatch max min formulation) to compute the dispatch variables related to the last CCG engagement computed by the MP
    # Use the worst case dispatch to get the equivalent of the max min formulation
    SP_primal = SP_primal_LP(pv_forecast=pv_worst_case, engagement=final_engagement)
    SP_primal.solve()
    SP_primal_sol = SP_primal.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # Check if there has been any simultanenaous charge and discharge during all CCG iterations
    # ------------------------------------------------------------------------------------------------------------------

    # 1. Check if there is any simultaneous charge and discharge at the last CCG iteration
    nb_count = check_BESS(SP_primal_sol=SP_primal_sol)
    print('CCG last iteration %d simultaneous charge and discharge' % (nb_count))

    # 2. Check if there is any simultaneous charge and discharge over all CCG iteration
    # check if there is nan value (meaning during an iteration the SP primal has not been solved because infeasible, etc)
    BESS_count = conv_inf['BESS_count']
    if sum(np.isnan(BESS_count)) > 0:
        print('WARNING %s nan values' %(sum(np.isnan(conv_inf['BESS_count']))))
    # “python list replace nan with 0” Code
    BESS_count = [0 if x != x else x for x in BESS_count]

    print('%d total simultaneous charge and discharge over all CCG iterations' % (sum(BESS_count)))
    if sum(conv_inf['BESS_count']) > 0:
        plt.figure()
        plt.plot(conv_inf['BESS_count'], 'k', linewidth=3)
        plt.ylim(0, max(conv_inf['BESS_count']))
        plt.xlabel('iteration $j$', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(dirname + day + '_BESS_count_' + pdfname + '.pdf')
        plt.close('all')

        # Plot at each iteration where there has been a simultaneous charge and discharge
        for l in conv_inf['BESS_charge_discharge']:
            plt.figure()
            plt.plot(l[1], linewidth=3, label='charge')
            plt.plot(l[2], linewidth=3, label='discharge')
            plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
            plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.legend(fontsize=FONTSIZE)
            plt.title('simultaneous charge discharge at iteration %s' %(l[0]))
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Check CCG convergence by computing the planning for the PV worst trajectory from CCG last iteration
    # ------------------------------------------------------------------------------------------------------------------
    planner = Planner_MILP(pv_forecast=pv_worst_case)
    planner.solve()
    sol_planner = planner.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # First-stage variables comparison: x and objectives
    # ------------------------------------------------------------------------------------------------------------------

    # Compute solution with the oracle
    planner = Planner_MILP(pv_forecast=pv_solution_day)
    planner.solve()
    sol_oracle = planner.store_solution()

    # Compute solution with the point-forecasts
    planner = Planner_MILP(pv_forecast=pv_point_day)
    planner.solve()
    sol_point = planner.store_solution()

    # Compute solution with the 50% quantile
    planner = Planner_MILP(pv_forecast=pv_quantile_day[:, 4])
    planner.solve()
    sol_50_quantile = planner.store_solution()

    # Compute solution with the q_min quantile
    planner = Planner_MILP(pv_forecast=pv_quantile_day[:, q_min])
    planner.solve()
    sol_q_min = planner.store_solution()

    plt.figure()
    # plt.plot(engagement_dict[4], linewidth=3, label='x 50 %')
    plt.plot(sol_point['x'], 'k', linewidth=3, label='x nominal')
    plt.plot(final_engagement, 'b', linewidth=3, label='x RO')
    plt.plot(sol_oracle['x'], 'r', linewidth=3, label='x perfect')
    # plt.plot(sol_planner['x'], linewidth=3, label='x MILP')
    plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_x_' + pdfname + '.pdf')
    plt.close('all')

    # Convergence plot
    error_MP_SP = np.abs(df_objectives['MP'].values - df_objectives['SP'].values)
    error_SP = np.abs(df_objectives['SP'].values - df_objectives['SP_primal'].values)

    plt.figure()
    plt.plot(error_MP_SP, label='|MP - SP dual| €')
    plt.plot(error_SP, label='|SP primal - SP dual| €')
    plt.plot(100 * np.asarray(conv_inf['mipgap']), label='SP Dual mipgap %')
    plt.xlabel('iteration', fontsize=FONTSIZE)
    plt.ylim(-1, 10)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + 'error_conv_' + pdfname + '.pdf')
    plt.close('all')

    plt.figure(figsize=(7, 7))
    plt.plot(df_objectives['MP'].values, linewidth=3, label='$MP^j$')
    plt.plot(df_objectives['SP'].values, linewidth=3, label='$SP^j$')
    # plt.plot(df_objectives['SP_primal'].values, linewidth=3, label='SP primal')
    plt.hlines(y=sol_planner['obj'] + conv_tol, xmin=1, xmax=len(df_objectives['SP'].values), colors='k',
               linestyles=':', linewidth=1, label='$MILP^J \pm\epsilon$')
    plt.hlines(y=sol_planner['obj'], xmin=1, xmax=len(df_objectives['SP'].values), linewidth=3, label='$MILP^J$')
    plt.hlines(y=sol_planner['obj'] - conv_tol, xmin=1, xmax=len(df_objectives['SP'].values), colors='k',
               linestyles=':', linewidth=1)
    plt.xlabel('iteration $j$', fontsize=FONTSIZE)
    plt.ylabel('€', fontsize=FONTSIZE, rotation='horizontal')
    plt.ylim(round(sol_planner['obj'] * 1.05, 0), round(sol_planner['obj'] * 0.95, 0))
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, ncol=2)
    plt.tight_layout()
    plt.savefig(dirname + 'convergence_' + pdfname + '.pdf')
    plt.close('all')

    print('')
    print('-----------------------CHECK BENDERS CONVERGENCE------------------------------------')
    print('Final iteration %s |MP - SP dual| %.2f €' % (
    len(df_objectives), abs(df_objectives['MP'].values[-1] - df_objectives['SP'].values[-1])))
    print('SP primal %.2f € SP dual %.2f € -> |SP primal - SP dual| = %.2f' % (
    SP_primal_sol['obj'], SP_dual_sol['obj'], abs(SP_primal_sol['obj'] - SP_dual_sol['obj'])))
    err_planner_benders = abs(df_objectives['MP'].values[-1] - sol_planner['obj'])
    print('MILP planner %.2f € MP Benders %.2f € -> |MILP planner - MP Benders| = %.2f' % (
    sol_planner['obj'], df_objectives['MP'].values[-1], err_planner_benders))

    if err_planner_benders > conv_tol:
        print('-----------------------WARNING BENDERS IS NOT CONVERGED------------------------------------')
        print('abs error %.4f €' % (err_planner_benders))
    else:
        print('-----------------------BENDERS IS CONVERGED------------------------------------')
        print('Benders is converged with |MILP planner - MP Benders| = %.4f €' % (err_planner_benders))

    print('-----------------------Compare to deterministic formulation------------------------------------')
    print('q 50 %.2f q %s %.2f point %.2f oracle %.2f' % (
    sol_50_quantile['obj'], 10 * q_min + 10, sol_q_min['obj'], sol_point['obj'], sol_oracle['obj']))

    plt.figure()
    plt.plot(sol_point['y_s'], 'k', linewidth=3, label='s nominal')
    plt.plot(SP_primal_sol['y_s'], 'b', linewidth=3, label='s RO')
    plt.plot(sol_oracle['y_s'], 'r', linewidth=3, label='s perfect')
    plt.ylabel('kWh', fontsize=FONTSIZE, rotation='horizontal')
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_s_' + pdfname + '.pdf')
    plt.close('all')

    plt.figure()
    plt.plot(sol_50_quantile['y_prod'], linewidth=3, label='y_prod 50 %')
    plt.plot(SP_primal_sol['y_prod'], linewidth=3, label='y_prod benders')
    plt.plot(sol_planner['y_prod'], linewidth=3, label='y_prod MILP')
    plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_y_prod_' + pdfname + '.pdf')
    plt.close('all')

    plt.figure()
    plt.plot(SP_primal_sol['y_charge'], linewidth=3, label='charge')
    plt.plot(SP_primal_sol['y_discharge'], linewidth=3, label='discharge')
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_charge_discharge_' + pdfname + '.pdf')
    plt.close('all')