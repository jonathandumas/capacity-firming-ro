# -*- coding: UTF-8 -*-

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ro.utils import build_observations, build_point_forecast, build_point_intra_forecast, load_lstm_quantiles
from root_project import ROOT_DIR
from ro.ro_simulator_configuration import PARAMETERS

def compute_ro_dyn_params(pv_quantile: np.array, d_gamma_threshold: float, d_pvmin: float):
    """
    Compute the max depth and Gamma parameters for dynamic Robust Optimization given the parameters d_gamma_threshold and d_pvmin.
    :param pv_quantile:
    :param d_gamma_threshold:
    :param d_pvmin:
    :return: max depth and gamma.
    """

    df = pd.DataFrame()
    df['50-10'] = pv_quantile[:, 4] - pv_quantile[:, 0]
    df['50-20'] = pv_quantile[:, 4] - pv_quantile[:, 1]
    df['50-30'] = pv_quantile[:, 4] - pv_quantile[:, 2]
    df['50-40'] = pv_quantile[:, 4] - pv_quantile[:, 3]

    # Time periods where q 50 = 0
    q50_0_indx = np.argwhere((pv_quantile[:, 4] > 0).astype(np.int) == 0)[:, 0]

    # Count time periods where d 50-10 > gamma_threshold
    tab_gamma = (df['50-10'].values > d_gamma_threshold).astype(np.int)
    gamma = sum(tab_gamma)

    # Count time periods where d 50-20, d 50-30, d50-40  > depth_threshold
    tab_50_40 = (df['50-40'].values > d_pvmin * df['50-10'].values).astype(np.int)
    tab_50_30 = (df['50-30'].values > d_pvmin * df['50-10'].values).astype(np.int)
    tab_50_20 = (df['50-20'].values > d_pvmin * df['50-10'].values).astype(np.int)
    tab_depth = tab_50_40 + tab_50_30 + tab_50_20
    for i in q50_0_indx:
        tab_depth[i] = 0
        tab_gamma[i] = 0

    # FONTSIZE = 10
    # x_index = [i for i in range(0, 96)]
    # plt.figure()
    # plt.plot(x_index, tab_depth, label='depth')
    # plt.plot(x_index, tab_gamma, label=r'$\Gamma=$'+str(gamma))
    # plt.ylim(0, 4)
    # plt.xticks(fontsize=FONTSIZE)
    # plt.yticks(fontsize=FONTSIZE)
    # plt.title(day)
    # plt.tight_layout()
    # plt.legend(fontsize=FONTSIZE)
    # plt.show()

    # Build the worst PV trajectory
    worst_pv = pv_quantile[:, 3].copy()  # quantile 40%
    for k in range(0, 95 + 1):
        if tab_depth[k] == 1:
            worst_pv[k] = pv_quantile[k, 2]  # quantile 30%
        elif tab_depth[k] == 2:
            worst_pv[k] = pv_quantile[k, 1]  # quantile 20%
        elif tab_depth[k] == 3:
            worst_pv[k] = pv_quantile[k, 0]  # quantile 10%

    return worst_pv, gamma

nb_periods = 96
N_Q = 9
q_set = [0, 1, 2, 3, 4, 5, 6, 7, 8] # -> 10%, 20%, .., 90%
config = ['oracle', 'point'] + q_set # ['oracle', 'point'] + q_set
# Validation set
nb_days = 30  # 30, 334 = total number of days of the dataset
VS = str(nb_days) #

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    print('-----------------------------------------------------------------------------------------------------------')

    # load data
    pv_solution = build_observations()
    pv_dad = build_point_forecast()
    pv_intra = build_point_intra_forecast()
    # Load quantile forecasts: from 0 to 8 -> median = quantile 4
    pv_quantile = load_lstm_quantiles()
    day_list = [day.strftime('%Y-%m-%d') for day in pv_intra.index]
    random.seed(1)
    if nb_days == len(day_list):
        day_list_sampled = day_list
    else:
        day_list_sampled = random.sample(day_list, nb_days)

    # -----------------------------------------------------------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------------------------------------------------------

    gamma_threshold = 0.1 * PARAMETERS["pv_capacity"]
    depth_threshold = 0.3

    for day in day_list_sampled[:10]:

        pv_solution_day = pv_solution.loc[day].values
        pv_point_day = pv_dad.loc[day].values
        pv_quantile_day = pv_quantile.loc[day].values.reshape(nb_periods, N_Q)

        FONTSIZE = 10
        x_index = [i for i in range(0, nb_periods)]
        plt.figure()
        for j in range(1, N_Q // 2+1):
            plt.fill_between(x_index, pv_quantile_day[:,j + N_Q // 2], pv_quantile_day[:,(N_Q // 2) - j], alpha=0.5 / j, color=(1 / j, 0, 1))
        plt.plot(x_index, pv_quantile_day[:,4], 'b', linewidth=3, label='50 %')
        plt.plot(x_index, pv_solution_day, 'r', linewidth=3, label='obs')
        plt.plot(x_index, pv_point_day, 'k--', linewidth=3, label='point')
        plt.ylim(0, PARAMETERS["pv_capacity"])
        plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.title(day)
        plt.tight_layout()
        plt.show()

        # plt.figure()
        # for j in range(0, 3+1):
        #     plt.plot(pv_quantile_day[:, 4] - pv_quantile_day[:, j], label='50 % - ' + str(10+10*j) + ' %')
        # plt.plot(depth_threshold * (pv_quantile_day[:, 4] - pv_quantile_day[:, 0]), 'k',label='depth threshold')
        # plt.hlines(y=gamma_threshold, xmin=0, xmax=95, colors='r', label='gamma threshold')
        # plt.ylim(0, 0.5 * PARAMETERS["pv_capacity"])
        # plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
        # plt.xticks(fontsize=FONTSIZE)
        # plt.yticks(fontsize=FONTSIZE)
        # plt.legend(fontsize=FONTSIZE)
        # plt.title(day)
        # plt.tight_layout()
        # plt.show()

        worst_pv, gamma = compute_ro_dyn_params(pv_quantile=pv_quantile_day, d_gamma_threshold=gamma_threshold, d_pvmin=depth_threshold)

        plt.figure()
        for j in range(1, 5+1):
            plt.plot(x_index, pv_quantile_day[:,5-j], alpha=1 / j, color=(1 / j, 0, 1), linewidth=3, label=str(10*(5-j+1)) + '%')
        plt.plot(x_index, worst_pv, 'g',linewidth=3, label='worst PV with '+r'$\Gamma=$'+str(gamma))
        plt.plot(x_index, pv_solution_day, 'r:', linewidth=3, label='obs')
        plt.plot(x_index, pv_point_day, 'k--', linewidth=3, label='point')
        plt.ylim(0, PARAMETERS["pv_capacity"])
        plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.title(day)
        plt.tight_layout()
        plt.show()