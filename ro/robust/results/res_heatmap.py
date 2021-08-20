# -*- coding: UTF-8 -*-

"""
Draw heat-maps of the results for a given algorithm (BD or CCG) for both the LSTM and NF quantiles.
Compare the results to the deterministic approach.
WARNING: the results must have been computed before.
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from root_project import ROOT_DIR
from ro.ro_simulator_configuration import PARAMETERS


def RO_static_res(dir:str, TEST_size:str, tag:str, warm_start:bool=False):
    """
    Compute results of the robust approach with fixed parameters over the TEST set.
    """
    twoD_res_sum = []
    for q_min in [0, 1, 2, 3]:
        oneD_res_sum = []
        for gamma in [12, 24, 36, 48]:
            dirname = dir + TEST_size  + '_ro_static/' + str(q_min) + '/' + str(gamma) + '/'
            csv_name = 'res_controller_'+ str(warm_start)+ '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])+ '.csv'

            if os.path.isfile(dirname + csv_name):
                df_res = pd.read_csv(dirname + csv_name, parse_dates=True, index_col=0)
                oneD_res_sum.append(df_res.sum()[tag])
            else:
                print('WARNING %s does not exist' % (dirname + csv_name))
                oneD_res_sum.append(math.nan)
        twoD_res_sum.append(oneD_res_sum)

    cols = ['12', '24', '36', '48']
    idx = ['10', '20', '30', '40']

    return pd.DataFrame(twoD_res_sum, index=idx, columns=cols) / 1000


def RO_dyn_res(dir:str, TEST_size:str, tag:str, warm_start:bool=False):
    """
    Compute results of the robust approach with dyn parameters over the TEST set.
    """
    twoD_res_sum = []
    for depth_threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
        oneD_res_sum = []
        oneD_res_mean = []
        for GAM_threshold in [0.05, 0.1, 0.2]:
            dirname = dir + TEST_size + '_ro_params/' + str(int(100 * depth_threshold)) + '/' + str(int(100 * GAM_threshold)) + '/'
            csv_name = 'res_controller_' + str(warm_start) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])+ '.csv'

            if os.path.isfile(dirname + csv_name):
                df_res = pd.read_csv(dirname + csv_name, parse_dates=True, index_col=0)
                oneD_res_sum.append(df_res.sum()[tag])
            else:
                print('WARNING %s does not exist' % (dirname + csv_name))
                oneD_res_sum.append(math.nan)
        twoD_res_sum.append(oneD_res_sum)

    cols = [5, 10, 20]
    idx = [5, 10, 20, 30, 50]

    return pd.DataFrame(twoD_res_sum, index=idx, columns=cols) / 1000


N_Q = 9 # nb quantiles
q_set = np.array([i / (N_Q+1) for i in range(1, N_Q + 1)]) # Set of quantiles

CONTROLLER = 'point' # point, oracle
# TEST set size
TEST_size = '30' # '30', 'aggregation'

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

    # Folders where the results have been saved
    dirname_RO_LSTM = 'ro/robust/export_' + algo + '_LSTM/'
    dirname_RO_NF = 'ro/robust/export_' + algo + '_NF/'
    dirname_det_LSTM = 'ro/determinist/export_LSTM/'
    dirname_det_NF = 'ro/determinist/export_NF/'
    pdf_export = 'ro/robust/export_heatmap/' + algo + '_'+ TEST_size + '/'
    if not os.path.isdir(pdf_export):  # test if directory exist
        os.makedirs(pdf_export)

    print('%s PENALTY_FACTOR %s DEADBAND %s CONTROLLER %s warm_start %s' %(algo, PARAMETERS['penalty_factor'], int(100 * PARAMETERS['tol_penalty']), CONTROLLER, warm_start))

    # --------------------------------------
    # 1. Fixed risk averse parameters
    # --------------------------------------

    # 1.1 RO LSTM & NF
    df_RO_fixed_sum_LSTM = RO_static_res(dir=dirname_RO_LSTM, TEST_size=TEST_size, tag=algo + ' ' + CONTROLLER, warm_start=warm_start)
    df_RO_fixed_sum_NF = RO_static_res(dir=dirname_RO_NF, TEST_size=TEST_size, tag=algo + ' ' + CONTROLLER, warm_start=warm_start)

    # 1.2 Deterministic
    # LSTM
    df_det_fixed_LSTM = pd.read_csv(dirname_det_LSTM + TEST_size + '_ro_static/det_' + CONTROLLER + '_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '.csv', parse_dates=True, index_col=0)
    df_det_fixed_LSTM = df_det_fixed_LSTM / 1000
    df_det_fixed_sum_LSTM = df_det_fixed_LSTM.sum()
    df_det_fixed_sum_LSTM.index = ['oracle', 'nominal', '10', '20', '30', '40', '50', '60', '70', '80', '90']

    # NF
    df_det_fixed_NF = pd.read_csv(dirname_det_NF + TEST_size + '_ro_static/det_' + CONTROLLER + '_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '.csv', parse_dates=True, index_col=0)
    df_det_fixed_NF = df_det_fixed_NF / 1000
    df_det_fixed_sum_NF = df_det_fixed_NF.sum()
    df_det_fixed_sum_NF.index = ['oracle', 'nominal', '10', '20', '30', '40', '50', '60', '70', '80', '90']

    # --------------------------------------
    # 2. Dynamic risk averse parameters
    # --------------------------------------
    #
    # # 2.1 RO
    df_RO_dyn_sum_LSTM = RO_dyn_res(dir=dirname_RO_LSTM, TEST_size=TEST_size, tag=algo + ' ' + CONTROLLER, warm_start=warm_start)
    df_RO_dyn_sum_NF = RO_dyn_res(dir=dirname_RO_NF, TEST_size=TEST_size, tag=algo + ' ' + CONTROLLER, warm_start=warm_start)
    #
    # 2.2 Deterministic
    df_det_dyn_LSTM = pd.read_csv(dirname_det_LSTM + TEST_size + '_ro_dyn/det_' + CONTROLLER + '_alt_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '.csv', parse_dates=True, index_col=0)
    df_det_dyn_LSTM = df_det_dyn_LSTM / 1000
    df_det_dyn_sum_LSTM = df_det_dyn_LSTM.sum()

    df_det_dyn_NF = pd.read_csv(dirname_det_NF + TEST_size + '_ro_dyn/det_' + CONTROLLER + '_alt_' + str(PARAMETERS['penalty_factor']) + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '.csv', parse_dates=True, index_col=0)
    df_det_dyn_NF = df_det_dyn_NF / 1000
    df_det_dyn_sum_NF = df_det_dyn_NF.sum()
    #
    # --------------------------------------
    # 3. Post-processing
    # --------------------------------------

    # Normalize results by the reference
    reference_LSTM = df_det_fixed_sum_LSTM['oracle']
    reference_NF = df_det_fixed_sum_NF['oracle']
    # LSTM
    df_RO_fixed_sum_norm_LSTM = 100 * df_RO_fixed_sum_LSTM / reference_LSTM
    df_det_fixed_sum_norm_LSTM = 100 * df_det_fixed_sum_LSTM / reference_LSTM
    df_RO_dyn_sum_norm_LSTM = 100 * df_RO_dyn_sum_LSTM / reference_LSTM
    df_det_dyn_sum_norm_LSTM = 100 * df_det_dyn_sum_LSTM / reference_LSTM
    # NF
    df_RO_fixed_sum_norm_NF = 100 * df_RO_fixed_sum_NF / reference_LSTM
    df_det_fixed_sum_norm_NF = 100 * df_det_fixed_sum_NF / reference_LSTM
    df_RO_dyn_sum_norm_NF = 100 * df_RO_dyn_sum_NF / reference_LSTM
    df_det_dyn_sum_norm_NF = 100 * df_det_dyn_sum_NF / reference_LSTM

    for col in df_det_fixed_LSTM.columns[:2]:
        print('%s-%s %.2f k€ %.2f' % (col, CONTROLLER, df_det_fixed_LSTM[col].sum(), 100 * df_det_fixed_LSTM[col].sum() / reference_LSTM))
    for col in df_det_fixed_LSTM.columns[2:]:
        print('q %s-%s %.2f k€ %.2f' % (int(col) * 10 + 10, CONTROLLER, df_det_fixed_LSTM[col].sum(), 100 * df_det_fixed_LSTM[col].sum() / reference_LSTM))

    # --------------------------------------
    # 4. Heat map of the results
    # --------------------------------------

    import seaborn as sns
    FONTSIZE = 20
    cmap = 'RdYlGn' # RdYlGn_r -> _r reverses the normal order of the color map 'RdYlGn'
    vmin = 50
    vmax = 75

    # ------------------------------------------------------------------------------------------------------------------
    # 4.1.1 NF-RO & deterministic with fixed parameters
    # ------------------------------------------------------------------------------------------------------------------
    df_sum_det_norm_plot_NF = df_det_fixed_sum_norm_NF['nominal':'40'].to_frame()
    df_sum_det_norm_plot_NF.columns = ['']
    df_sum_det_norm_plot_LSTM = df_det_fixed_sum_norm_LSTM['nominal':'40'].to_frame()
    df_sum_det_norm_plot_LSTM.columns = ['']

    df_RO_fixed_plot_NF = pd.concat([df_RO_fixed_sum_norm_NF, df_sum_det_norm_plot_NF['10':]], axis=1, join='inner')
    df_RO_fixed_plot_NF.columns = ['12', '24', '36', '48', '/']

    plt.figure(figsize=(6, 5))
    sns.set(font_scale=1.5)
    sns_plot = sns.heatmap(df_RO_fixed_plot_NF, cmap=cmap, fmt=".1f", linewidths=0.5, annot=True,
                           cbar_kws={'label': '%'}, vmin=vmin, vmax=vmax)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation='horizontal')
    plt.xlabel('$\Gamma$', fontsize=FONTSIZE, labelpad=-20, x=0)
    plt.ylabel('$q$', rotation='horizontal', fontsize=FONTSIZE, labelpad=-10, y=0.95)
    plt.title('Nominal = ' + str(round(df_sum_det_norm_plot_LSTM.loc['nominal'].values[0], 1)) + ' %')
    plt.tight_layout()
    plt.show()

    figure = sns_plot.get_figure()
    figure.savefig(pdf_export + algo+'_NF_RO_fixed_vs_det' + CONTROLLER + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor']) + '.pdf', dpi=400)

    # ------------------------------------------------------------------------------------------------------------------
    # 4.1.2 NF: RO & deterministic with dyn parameters
    # ------------------------------------------------------------------------------------------------------------------
    df_sum_det_alt_norm_plot_NF = df_det_dyn_sum_norm_NF['0.05':].to_frame()
    df_sum_det_alt_norm_plot_NF.columns = ['']
    df_sum_det_alt_norm_plot_NF.index = [5, 10, 20, 30, 50]
    df_sum_det_alt_norm_plot_LSTM = df_det_dyn_sum_norm_LSTM['0.05':].to_frame()
    df_sum_det_alt_norm_plot_LSTM.columns = ['']
    df_sum_det_alt_norm_plot_LSTM.index = [5, 10, 20, 30, 50]

    df_RO_dyn_plot_NF = pd.concat([df_RO_dyn_sum_norm_NF, df_sum_det_alt_norm_plot_NF], axis=1, join='inner')
    df_RO_dyn_plot_NF.columns = ['5', '10', '20', '/']

    plt.figure(figsize=(6,5))
    sns.set(font_scale=1.5)
    sns_plot = sns.heatmap(df_RO_dyn_plot_NF, cmap=cmap, fmt=".1f", linewidths=0.5, annot=True, cbar_kws={'label': '%'}, vmin=vmin, vmax=vmax)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation='horizontal')
    plt.xlabel('$d_\Gamma$', fontsize=FONTSIZE, labelpad=-20, x=0)
    plt.ylabel('$d_q$', rotation='horizontal', fontsize=FONTSIZE, labelpad=-10, y=0.95)
    plt.tight_layout()
    plt.show()

    figure = sns_plot.get_figure()
    figure.savefig(pdf_export+ algo+'_NF_RO_dyn_vs_det'+CONTROLLER+'_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])+'.pdf', dpi=400)

    # ------------------------------------------------------------------------------------------------------------------
    # 4.2.1 LSTM: RO & deterministic with fixed parameters
    # ------------------------------------------------------------------------------------------------------------------

    df_RO_fixed_plot_LSTM = pd.concat([df_RO_fixed_sum_norm_LSTM, df_sum_det_norm_plot_LSTM['10':]], axis=1, join='inner')
    df_RO_fixed_plot_LSTM.columns = ['12', '24', '36', '48', '/']

    plt.figure(figsize=(6, 5))
    sns.set(font_scale=1.5)
    sns_plot = sns.heatmap(df_RO_fixed_plot_LSTM, cmap=cmap, fmt=".1f", linewidths=0.5, annot=True,
                           cbar_kws={'label': '%'}, vmin=vmin, vmax=vmax)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation='horizontal')
    plt.xlabel('$\Gamma$', fontsize=FONTSIZE, labelpad=-20, x=0)
    plt.ylabel('$q$', rotation='horizontal', fontsize=FONTSIZE, labelpad=-10, y=0.95)
    plt.title('Nominal = ' + str(round(df_sum_det_norm_plot_LSTM.loc['nominal'].values[0], 1)) + ' %')
    plt.tight_layout()
    plt.show()

    figure = sns_plot.get_figure()
    figure.savefig(pdf_export + algo+'_LSTM_RO_fixed_vs_det' + CONTROLLER + '_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(
            PARAMETERS['penalty_factor']) + '.pdf', dpi=400)

    # ------------------------------------------------------------------------------------------------------------------
    # 4.2.2 NF: RO & deterministic with dyn parameters
    # ------------------------------------------------------------------------------------------------------------------

    df_RO_dyn_plot_LSTM = pd.concat([df_RO_dyn_sum_norm_LSTM, df_sum_det_alt_norm_plot_LSTM], axis=1, join='inner')
    df_RO_dyn_plot_LSTM.columns = ['5', '10', '20', '/']

    plt.figure(figsize=(6,5))
    sns.set(font_scale=1.5)
    sns_plot = sns.heatmap(df_RO_dyn_plot_LSTM, cmap=cmap, fmt=".1f", linewidths=0.5, annot=True, cbar_kws={'label': '%'}, vmin=vmin, vmax=vmax)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation='horizontal')
    plt.xlabel('$d_\Gamma$', fontsize=FONTSIZE, labelpad=-20, x=0)
    plt.ylabel('$d_q$', rotation='horizontal', fontsize=FONTSIZE, labelpad=-10, y=0.95)
    plt.tight_layout()
    plt.show()

    figure = sns_plot.get_figure()
    figure.savefig(pdf_export+ algo+'_LSTM_RO_dyn_vs_det'+CONTROLLER+'_' + str(int(100 * PARAMETERS['tol_penalty'])) + '_' + str(PARAMETERS['penalty_factor'])+'.pdf', dpi=400)