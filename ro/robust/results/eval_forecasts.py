# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ro.utils import load_lstm_quantiles, build_observations, build_point_forecast, dump_file, load_nf_quantiles
from root_project import ROOT_DIR
from ro.ro_simulator_configuration import PARAMETERS

def plf_per_quantile(quantiles:np.array, y_true:np.array):
    """
    Compute PLF per quantile.
    :param quantiles: (nb_periods, nb_quantiles)
    :param y_true:  (nb_periods,)
    :return: PLF per quantile into an array (nb_quantiles, )
    """
    # quantile q from 0 to N_q -> add 1 to be from 1 to N_q into the PLF score
    N_q = quantiles.shape[1]
    plf = []
    for q in range(0 ,N_q):
        # for a given quantile compute the PLF over the entire dataset
        diff = y_true - quantiles[:,q]
        plf_q = sum(diff[diff >= 0] * ((q+1) / (N_q+1))) / len(diff) + sum(-diff[diff < 0] * (1 - (q+1) / (N_q+1))) / len(diff) # q from 0 to N_q-1 -> add 1 to be from 1 to N_q
        plf.append(plf_q)
    return np.asarray(plf)

def compute_reliability(y_true: np.array, y_quantile: np.array):
    """
    Compute averaged reliability score per day over all quantiles.
    :param y_true: true values (n_periods, ).
    :param y_quantile: quantiles (n_periods, n_quantiles).
    :return: PLF array of shape (n_quantiles,)
    """
    nb_q = y_quantile[0].shape[0]

    aq = []
    # WARNING REMOVE TIME PERIODS WHERE PV GENERATION IS 0 during night hours !!!!
    # indices where PV generation is 0 at day d
    indices = np.where(y_true == 0)[0]
    y_true = np.delete(y_true, indices).copy()
    y_quantile = np.delete(y_quantile, indices, axis=0).copy()

    nb_periods = len(y_true)
    for q in range(0, nb_q):
        aq.append(sum(y_true < y_quantile[:, q]) / nb_periods)

    return 100 * np.asarray(aq)

def crps_nrg_k(y_true:float, y_sampled:np.array):
    """
    Compute the CRPS NRG for a given leadtime k.
    :param y_true: true value for this leadtime.
    :param y_sampled: nb quantile/scenarios for leadtime k with shape(nb,)
    """
    nb = y_sampled.shape[0] # Nb of quantiles/scenarios sampled.
    simple_sum = np.sum(np.abs(y_sampled - y_true)) / nb
    double_somme = 0
    for i in range(nb):
        for j in range(nb):
            double_somme += np.abs(y_sampled[i] - y_sampled[j])
    double_sum = double_somme / (2 * nb * nb)

    crps = simple_sum  - double_sum

    return crps

def crps_per_period(scenarios: np.array, y_true: np.array):
    """
    Compute the CRPS per period.
    :param scenarios: of shape (n_periods, Q_quantiles)
    :param y_true: observations of shape (n_periods, ) with  n_periods = n_d * 24
    :return: averaged CRPS per time period CRPS into a np.array of shape (24,)
    """
    n_periods = scenarios.shape[0]
    n_d = int(len(y_true) / 96)

    crps_t = np.asarray([crps_nrg_k(y_true=y_true[t], y_sampled=scenarios[t,:]) for t in range(n_periods)])
    crps_averaged = crps_t.reshape(n_d, 96).mean(axis=0)  # averaged CRPS per time period

    return crps_averaged, crps_t.reshape(n_d, 96)


def compare_plf(dir: str, plf: list, name: str, ylim:list, labels:str):
    """
    Plot the quantile score (PLF = Pinball Loss Function) per quantile on TEST sets of multiple generative models.
    :param plf: list of the plf_score of multiple generative models. Each element of the list is an array.
    """
    x_index = [int(10*q) for q in range(1, 9+1)]
    plt.figure()
    for l, label in zip(plf, labels):
        plt.plot(x_index, l, label=label)
    plt.ylim(ylim[0], ylim[1])
    plt.vlines(x=50, colors='k', ymin=0, ymax=ylim[1])
    plt.xlim(10, 90)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xticks(ticks=x_index)
    plt.xlabel('%', fontsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE, rotation='horizontal')
    # plt.legend(fontsize=1.5*FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir + name + '.pdf')
    plt.show()

def compare_reliability_diag(dir: str, aq: list, name: str, labels:str):
    """
    Reliablity diagram per quantile.
    :param aq: list of the aq scores of multiple generative models. Each element of the list is an array of shape (n_q,).
    """

    N_q = aq[0].shape[0]
    q_set = [i / (N_q + 1) for i in range(1, N_q + 1)]
    x_index = np.array(q_set) * 100

    plt.figure()
    plt.plot(x_index, x_index, 'k', linewidth=2)
    for a, label in zip(aq, labels):
        plt.plot(x_index, a, label=label)
    plt.xlim(10, 90)
    plt.ylim(10, 90)
    plt.xlabel('$q$', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xticks(ticks=[i for i in range(10, 100, 10)])
    plt.yticks(ticks=[i for i in range(10, 100, 10)])
    plt.ylabel('%', fontsize=FONTSIZE, rotation='horizontal')
    # plt.title(name)
    plt.legend(fontsize=1.5*FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir + name + '.pdf')
    plt.show()

def compare_crps(dir: str, crps: list, name: str, ylim:list, labels:str):
    """
    :param crps: list of the crps scores of multiple generative models. Each element of the list is an array.
    """

    plt.figure()
    for c, label in zip(crps, labels):
        plt.plot(100 * c, label=label)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE, rotation='horizontal')
    plt.xlabel('lead time', fontsize=FONTSIZE)
    # plt.legend(fontsize=1.5*FONTSIZE)
    plt.grid(True)
    plt.xlim(0, 95+1)
    plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(dir + name + '.pdf')
    plt.show()


# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

FONTSIZE = 20

# NB periods
nb_periods = 96
N_Q = 9
q_set = [0, 1, 2, 3, 4, 5, 6, 7, 8] # -> 10%, 20%, .., 90%

# Select the day
day_list = ['2019-09-14', '2020-05-26',  '2020-09-26', '2019-08-05']
day = day_list[0]

# RO parameters
quantile = 0 # from 0 to 3

# quantile from NF or LSTM
NF_quantile = True

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    # Create folder
    dirname = 'ro/robust/export/forecast_eval/'

    if not os.path.isdir(dirname):  # test if directory exist
        os.makedirs(dirname)

    # load PV forecasts
    pv_dad = build_point_forecast()
    day_list = [day.strftime('%Y-%m-%d') for day in pv_dad.index]
    pv_solution = build_observations()
    # reshape into an array (n_periods,)
    y_true = np.concatenate([pv_solution.loc[day].values for day in day_list], axis=0)
    pv_quantile_NF = load_nf_quantiles()
    # reshape into an array (n_periods, Q_quantiles)
    q_NF = np.concatenate([pv_quantile_NF[day].transpose() for day in day_list], axis=0)

    # Load quantile forecasts: from 0 to 8 -> median = quantile 4
    pv_quantile_LSTM = load_lstm_quantiles()
    # reshape into an array (n_periods, Q_quantiles)
    q_LSTM = np.concatenate([pv_quantile_LSTM.loc[day].values.reshape(nb_periods, N_Q) for day in day_list], axis=0)

    # Normalized PLF
    plf_q_NF = 100 * plf_per_quantile(quantiles=q_NF, y_true=y_true) / 466.4
    plf_q_LSTM = 100 * plf_per_quantile(quantiles=q_LSTM, y_true=y_true) / 466.4
    print('PLF %.2f %.2f' % (plf_q_NF.mean(), plf_q_LSTM.mean()))
    print('')

    compare_plf(dir=dirname, plf=[plf_q_NF, plf_q_LSTM], name='plf', ylim=[0.8, 2.2], labels=['NF', 'LSTM'])

    a_NF = compute_reliability(y_true=y_true, y_quantile=q_NF)
    a_LSTM = compute_reliability(y_true=y_true, y_quantile=q_LSTM)

    compare_reliability_diag(dir=dirname, aq=[a_NF, a_LSTM], name='reliability', labels=['NF', 'LSTM'])

    # Normalized CRPS
    crps_NF, crps_d_NF = crps_per_period(scenarios=q_NF, y_true=y_true)
    crps_LSTM, crps_d_LSTM = crps_per_period(scenarios=q_LSTM, y_true=y_true)

    compare_crps(dir=dirname, crps=[crps_NF/466.4, crps_LSTM/466.4], name='crps', ylim=[0, 10], labels=['NF', 'LSTM'])
    print('CRPS %.2f %.2f' % (100*(crps_NF/466.4).mean(), (100*crps_LSTM/466.4).mean()))
    print('')

    # Select a paticular day of the dataset
    pv_solution_day = pv_solution.loc[day].values
    pv_point_day = pv_dad.loc[day].values
    # quantile from NF
    pv_quantile_day_NF = pv_quantile_NF[day].transpose()
    # quantile from LSTM
    pv_quantile_day_LSTM = pv_quantile_LSTM.loc[day].values.reshape(nb_periods, N_Q)

    # PLOT quantile, point, and observations
    x_index = [i for i in range(0, nb_periods)]
    plt.figure()
    for j in range(1, N_Q // 2 + 1):
        plt.fill_between(x_index, pv_quantile_day_NF[:, j + N_Q // 2], pv_quantile_day_NF[:, (N_Q // 2) - j], alpha=0.5 / j, color=(1 / j, 0, 1))
    plt.plot(x_index, pv_quantile_day_NF[:, 4], 'b', linewidth=3, label='50 %')
    plt.plot(x_index, pv_solution_day, 'r', linewidth=3, label='obs')
    plt.plot(x_index, pv_point_day, 'k--', linewidth=3, label='nominal')
    plt.ylim(0, PARAMETERS["pv_capacity"])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()
    # plt.savefig(dirname +str(day) + '_quantile_forecast.pdf')
    # plt.close('all')

    # PLOT quantile, point, and observations
    x_index = [i for i in range(0, nb_periods)]
    plt.figure()
    for j in range(1, N_Q // 2 + 1):
        plt.fill_between(x_index, pv_quantile_day_LSTM[:, j + N_Q // 2], pv_quantile_day_LSTM[:, (N_Q // 2) - j], alpha=0.5 / j, color=(1 / j, 0, 1))
    plt.plot(x_index, pv_quantile_day_LSTM[:, 4], 'b', linewidth=3, label='50 %')
    plt.plot(x_index, pv_solution_day, 'r', linewidth=3, label='obs')
    plt.plot(x_index, pv_point_day, 'k--', linewidth=3, label='nominal')
    plt.ylim(0, PARAMETERS["pv_capacity"])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


    plt.figure()
    plt.plot(pv_solution.values[19,:], linewidth=3)
    plt.tight_layout()
    plt.savefig(dirname+ 'pv.pdf')
    plt.show()

    import scipy
    mean = 0
    standard_deviation = 2
    x_values = np.arange(-5, 5, 0.1)
    y_values = scipy.stats.norm(mean, standard_deviation)
    plt.figure()
    plt.plot(x_values, y_values.pdf(x_values), 'k', linewidth=3)
    plt.tight_layout()
    plt.savefig(dirname+ 'gaussian.pdf')
    plt.show()
