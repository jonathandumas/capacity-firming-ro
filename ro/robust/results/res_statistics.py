# -*- coding: UTF-8 -*-

"""
Compute time computation statistics.
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from root_project import ROOT_DIR
from ro.ro_simulator_configuration import PARAMETERS

N_Q = 9 # nb quantiles
q_set = np.array([i / (N_Q+1) for i in range(1, N_Q + 1)]) # Set of quantiles

# TEST set size
TEST_size = '30' # '30', 'aggregation'

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

    # Folder with results
    dirname = 'ro/robust/export_'+algo+'_'+q_technique+'/'
    pdfname = algo + '_' + q_technique + '_' + str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])
    print('%s: TEST size %s %s warm start %s PENALTY_FACTOR %s DEADBAND %s OVERPRODUCTION %s' %(algo, TEST_size, q_technique, warm_start, PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), PARAMETERS['OVERPRODUCTION']))

    time_list = []
    for q_min in [0, 1, 2, 3]:
        for gamma in [12, 24, 36, 48]:
            dir_tmp = dirname + TEST_size  + '_ro_static/' + str(q_min) + '/' + str(gamma) + '/'
            df_name = dir_tmp + 't_CPU_' + pdfname + '.csv'
            df_time = pd.read_csv(df_name, index_col=0)
            time_list.append(df_time.values)
    time_arr_ro_static = np.asarray(time_list).reshape(-1)

    print('t av %.2f -/+ %.2f t min %.2f t max %.2f (s)' %(np.mean(time_arr_ro_static), np.std(time_arr_ro_static), np.min(time_arr_ro_static), np.max(time_arr_ro_static)))

    time_list = []
    for depth_threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for GAM_threshold in [0.05, 0.1, 0.2]:
            dir_tmp = dirname + TEST_size + '_ro_params/' + str(int(100 * depth_threshold)) + '/' + str(int(100 * GAM_threshold)) + '/'
            df_name = dir_tmp + 't_CPU_' + pdfname + '.csv'
            df_time = pd.read_csv(df_name, index_col=0)
            time_list.append(df_time.values)
    time_arr_ro_dyn = np.asarray(time_list).reshape(-1)

    print('t av %.2f -/+ %.2f t min %.2f t max %.2f (s)' %(np.mean(time_arr_ro_dyn), np.std(time_arr_ro_dyn), np.min(time_arr_ro_dyn), np.max(time_arr_ro_dyn)))