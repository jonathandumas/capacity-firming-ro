# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from root_project import ROOT_DIR


FONTSIZE = 20

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    pv_solution = pd.read_csv('data1/uliege/uliege_15min.csv', index_col=0, parse_dates=True)['Pm']
    pv_forecast = pd.read_csv('data1/uliege/uliege_pv_prediction_python_15min.csv', parse_dates=True, index_col=0)


    # day 1 -> moyen
    # day 2 -> sunny
    # day 3 -> cloudy
    day = "3"

    day_list = ['2019-08-24', '2019-11-16', '2019-12-02']
    day_names = ['soleil_08-24-19', 'moyen_16-11-19', 'nuage_02-12-19']
    df_quantiles = []
    for name, day in zip(day_names, day_list):
        df_input = pd.read_csv('data1/uliege/quantiles/'+name+'.csv', index_col=0, parse_dates=True)
        time_range = pd.date_range(start=pd.Timestamp(day+' 03:30:00'), periods=63, freq='15T')
        df = pd.DataFrame(data=df_input.values, index=time_range)
        df.index = df.index.tz_localize('UTC')
        df = df.reindex(index=pv_solution[day].index, fill_value=0)
        df_quantiles.append(df)
    df_quantiles = pd.concat(df_quantiles, axis=0, join='inner')

    for day in day_list:
        index = df_quantiles[day].index
        plt.figure()
        for col in df_quantiles.columns:
            plt.plot(df_quantiles[day][col].values)
        plt.plot(pv_solution[day].values, 'k', linewidth=3, label='Pm')
        plt.plot(pv_forecast[day].values, 'r', linewidth=3, label='Pp')
        plt.ylabel('kW', fontsize=FONTSIZE)
        plt.xlabel('lead time', fontsize=FONTSIZE)
        plt.ylim(0, 450)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig('figures/'+index[0].strftime('%Y-%m-%d')+'.pdf')
        plt.show()


    # quantile = pd.read_csv('data1/uliege/quantiles/prediction'+day+'.csv', index_col=0, parse_dates=True)
    # generation = pd.read_csv('data1/uliege/quantiles/true'+day+'.csv', index_col=0, parse_dates=True)
    #
    # # 11 quantiles symmétrique 1/12 : [0.08333333 0.16666667 0.25       0.33333333 0.41666667 0.5
    # #  0.58333333 0.66666667 0.75       0.83333333 0.91666667]
    # concat = pd.concat([quantile, generation], axis=1, join='inner')
    # concat.columns = [i for i in range(1, 11+1)] + ['Pm']
    #
    # plt.figure()
    # concat[[1, 6, 11, 'Pm']].plot()
    # plt.ylim(0, PV_CAPACITY)
    # plt.show()


