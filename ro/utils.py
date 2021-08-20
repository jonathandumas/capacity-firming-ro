# -*- coding: UTF-8 -*-

"""
Utils file containing several functions to be used.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ro.ro_simulator_configuration import PARAMETERS

def check_BESS(SP_primal_sol: dict):
    """
    Check if there is any simultanenaous charge and discharge of the BESS.
    :param SP_primal_sol: solution of the SP primal, dict with at least the keys arguments y_charge and y_charge.
    :return: number of simultanenaous charge and discharge.
    """
    df_check = pd.DataFrame(SP_primal_sol['y_charge'], columns=['y_charge'])
    df_check['y_discharge'] = SP_primal_sol['y_discharge']

    nb_count = 0
    for i in df_check.index:
        if (df_check.loc[i]['y_charge'] > 0) and (df_check.loc[i]['y_discharge'] > 0):
            nb_count += 1
    return nb_count

def build_observations(dir:str= 'ro/data/'):
    """
    Load the pv observations, and PV dad point forecasts of VS1 and VS2.
    :return: pv_solution, pv_dad_VS1, and pv_dad_VS2
    """

    # ------------------------------------------------------------------------------------------------------------------
    # pv_solution
    pv_solution = pd.read_csv(dir+'PV_uliege_parking_power_15min.csv', parse_dates=True, index_col=0)['power_total'].to_frame()
    pv_solution.columns = ['Pm']
    pv_solution = pd.DataFrame(data=pv_solution.values.reshape(int(pv_solution.shape[0] / 96), 96),
                               index=[day for day in pv_solution.resample('D').mean().dropna().index])


    return pv_solution

def build_point_forecast(dir:str= 'ro/data/'):
    """
    Load PV dad point forecasts of VS1 and VS2.
    :return: pv_solution, pv_dad_VS1, and pv_dad_VS2
    """

    k1 = 11  # 0 or 11
    k2 = 80  # 95 or 80
    pv_dad_VS = pd.read_csv(dir + 'aggregated_dad_point_24_LSTM1_500_0.001_11_80.csv', parse_dates=True, index_col=0)

    # Add 0 for 0 <= k < k1 and  k2 +1 <= k < 95
    pv_dad_VS = pv_dad_VS.sort_index(axis=1)
    for k in range(k2 + 1, 95 + 1):
        pv_dad_VS[str(k)] = 0

    df_to_add = pd.DataFrame(index=pv_dad_VS.index)
    for k in range(0, k1):
        df_to_add[str(k)] = 0
    pv_dad_VS = pd.concat([df_to_add, pv_dad_VS], axis=1, join='inner')

    return pv_dad_VS

def build_point_intra_forecast(dir:str= 'ro/data/'):
    """
    Load the PV intra point forecasts of VS1 and VS2.
    :return: pv_solution, pv_dad_VS1, and pv_dad_VS2
    """

    k1 = 11  # 0 or 11
    k2 = 80  # 95 or 80
    pv_intra_VS = pd.read_csv(dir + 'aggregated_intra_point_0_lstm_mlp_500_0.001_11_80.csv', parse_dates=True, index_col=0)

    # Add 0 for 0 <= k < k1 and  k2 +1 <= k < 95
    pv_intra_VS = pv_intra_VS.sort_index(axis=1)
    for k in range(k2 + 1, 95 + 1):
        pv_intra_VS[str(k)] = 0

    df_to_add = pd.DataFrame(index=pv_intra_VS.index)
    for k in range(0, k1):
        df_to_add[str(k)] = 0
    pv_intra_VS = pd.concat([df_to_add, pv_intra_VS], axis=1, join='inner')

    return pv_intra_VS

def load_lstm_quantiles():
    """
    Load LSTM pv quantile forecast into a pd.DataFrame.
    """
    pv_quantile = pd.read_csv('ro/data/aggregated_dad_quantile_24_LSTM1_500_0.001_11_80.csv', parse_dates=True, index_col=0)

    k1 = 11
    k2 = 80
    N_Q = 9
    df_to_add2 = pd.DataFrame(index=pv_quantile.index)
    col_list = []
    for i in range(k2 + 1, 95 + 1):
        col_list += ['q' + str(i) + '_' + str(j) for j in range(1, N_Q + 1)]
    for col in col_list:
        df_to_add2[col] = 0

    df_to_add1 = pd.DataFrame(index=pv_quantile.index)
    col_list = []
    for i in range(0, k1):
        col_list += ['q' + str(i) + '_' + str(j) for j in range(1, N_Q + 1)]
    for col in col_list:
        df_to_add1[col] = 0

    return pd.concat([df_to_add1, pv_quantile, df_to_add2], axis=1, join='inner')

def load_nf_quantiles():
    """
    Load NF PV quantile forecast into a dict.
    """
    scenarios_NF = read_file(dir='ro/data/', name='NF_quantiles_aggregated_100_pvcs_scaled_5_3_150_100_100')

    # 0 <= k <= 95
    k1 = 11
    k2 = 80
    N_Q = 9

    array1 = np.zeros((N_Q, k1))
    array2 = np.zeros((N_Q, 95 - k2))

    scenarios_NF_new = dict()
    for day in scenarios_NF.keys():
        scenarios_NF_new[day.strftime('%Y-%m-%d')] = np.concatenate((array1, scenarios_NF[day], array2), axis=1)

    return scenarios_NF_new

def dump_file(dir:str, name: str, file):
    """
    Dump a file into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def read_file(dir:str, name: str):
    """
    Read a file dumped into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'rb')
    file = pickle.load(file_name)
    file_name.close()

    return file

def dump_sol(dir:str, sol: list, sol_eval: list, planner:str):
    """
    Dump all the files from STEP1.
    :param dir: directory.
    :param sol: solutions from STEP1.
    :param sol_eval: evaluation of the solutions from STEP1.
    :param planner: 'STEP1D', STEP1S', 'STEP1D_perfect'.
    """

    dump_file(dir=dir, name='solutions' + planner, file=sol)
    dump_file(dir=dir, name='solutions_eval' + planner, file=sol_eval)

def read_sol(dir:str, planner:str):
    """
    Read all the files from STEP1.
    :param dir: directory.
    :param planner: 'STEP1D', STEP1S', 'STEP1D_perfect'.
    :return solutions, solutions_eval
    """

    solutions = read_file(dir=dir, name='solutions' + planner)
    solutions_eval = read_file(dir=dir, name='solutions_eval' + planner)

    return solutions, solutions_eval

def create_dir(dirName: str):
    """
    Create target directory if does not exist
    :param dirName: path and name of the directory to create.
    """
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

def compute_res_ro(dir:str, day_VS_list: list, VS: str = 'aggregation', warm_start:bool=False):
    """
    :param dir: directory.
    :param day_VS_list: list of days of the considered VS.
    :param VS: VS name.
    """
    df_results = pd.DataFrame(index=day_VS_list)
    for quantile in [0, 1, 2, 3]:
        for gamma in [12, 24, 36, 48]:
            csv_name = 'res_controller_' + str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']) + '.csv'
            path_file = dir + VS + '/quantile_' + str(quantile) + '/gamma_' + str(gamma) + '/' + csv_name
            if os.path.isfile(path_file):
                interval_width = 100 - 20 - 20 * quantile
                df_results[str(interval_width) + '|' + str(gamma)] = pd.read_csv(path_file, parse_dates=True, index_col=0)['benders']
    return df_results

def scale_data(df_data_ls:pd.DataFrame, df_data_vs:pd.DataFrame):
    """
    1. Scale data using the StandardScaler on the learning set.
    2. Transform the inputs of the learning and valiation sets.
    :param df_data_ls: pd.DataFrame with the learning set data to fit the scaler.
    :param df_data_vs: pd.DataFrame with the validation set data.
    :return the learning and validation scaled data.
    """

    scaler = StandardScaler()
    scaler.fit(df_data_ls)
    return scaler.transform(df_data_ls), scaler.transform(df_data_vs)