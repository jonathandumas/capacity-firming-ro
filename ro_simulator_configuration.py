# -*- coding: UTF-8 -*-

"""
Define the simulation parameters for the robust and deterministic algorithms.
"""

import numpy as np
import pandas as pd
# ------------------------------------------------------------------------------------------------------------------
# 1. PV configuration of the Uliège case study
PV_CAPACITY = 466.4  # kWp
STEP1_PERIOD_min = 15 # time resolution of the planner
STEP2_PERIOD_min = 15 # time resolution of the controller

# ------------------------------------------------------------------------------------------------------------------
# 2. Simulator parameters
# ------------------------------------------------------------------------------------------------------------------
# CRE Tender parameters
# price (€ / MWh)
# price_peak (€ / MWh)
# tol_penalty (%)
# tol_ramping (% / min)
# tol_ramping_peak (% / min)
# min_nomination (%)
# min_nomination_peak (%)
# max_nomination (%)
# min_production (%)
# min_production_peak (%)
# max_production (%)
# penalty_factor (/)
CRE_parameters = {"price": 100,
                  "price_peak": 300,
                  "tol_penalty": 0.05,
                  "tol_ramping": 0.005,
                  "tol_ramping_peak": 2 * 0.005,
                  "min_nomination": -0.05,
                  "min_nomination_peak": 0.2,
                  "max_nomination": 1,
                  "min_production": -0.05,
                  "min_production_peak": 0.15,
                  "max_production": 1,
                  "penalty_factor": 1
                  }

IEEE_paper_parameters = {"price": 100,
                         "price_peak": 300,
                         "tol_penalty": 0.01,
                         "tol_ramping": 0.005,
                         "tol_ramping_peak": 2 * 0.005,
                         "min_nomination": 0,
                         "min_nomination_peak": 0,
                         "max_nomination": 1,
                         "min_production": 0,
                         "min_production_peak": 0,
                         "max_production": 1,
                         "penalty_factor": 5
                         }

# Select the capacity firming parameters
PARAMETERS = IEEE_paper_parameters # IEEE_paper_parameters, CRE_parameters

PARAMETERS["period_hours"] = STEP1_PERIOD_min / 60  # (hours)
PARAMETERS["pv_capacity"] = PV_CAPACITY # kW
PARAMETERS['OVERPRODUCTION'] = False # to forbid overproduction by adding a hard constraint

STEP1_PERIOD_hour = STEP1_PERIOD_min / 60  # (hours)
DEADBAND_DEVIATION_kW = PARAMETERS['tol_penalty'] * PV_CAPACITY # Maximum power between nomination and export before applying penalty (kW)

STEP2_PERIOD_hour = STEP2_PERIOD_min / 60 # (hours)

# peak hours: 7 - 9 pm
# parameters: selling price, ramping power constraint, max nomination, min nomination, max production, and min production
# CONVENTION delta_t = 15min: t = 19h00 -> period from 19h00 to 19h15
# CONVENTION delta_t = 1min: t = 19h00 -> period from 19h00 to 19h01
# Ex: Production => 0.15 * Pc during peak hours -> production_t => 0.15*Pc from t = 19h00 to t= 18h45 with delta_t = 15 min
# Ex: Production => 0.15 * Pc during peak hours -> production_t => 0.15*Pc from t = 19h00 to t= 19h59 with delta_t = 1 min
# Ramping power constraints on nomination

df_parameters_1min = pd.DataFrame(index=pd.date_range(start=pd.Timestamp('2000'), freq='1min', periods=1440))
df_parameters_1min['price'] = PARAMETERS['price'] / 1000
df_parameters_1min['price']['2000-01-01 19':'2000-01-01 20'] = PARAMETERS['price_peak'] / 1000
df_parameters_1min['deadband_nomination'] = PARAMETERS['tol_ramping'] * PV_CAPACITY * STEP1_PERIOD_min
df_parameters_1min['deadband_nomination']['2000-01-01 19':'2000-01-01 20'] = PARAMETERS['tol_ramping_peak'] * PV_CAPACITY * STEP1_PERIOD_min
df_parameters_1min['min_nomination'] = PARAMETERS['min_nomination'] * PV_CAPACITY
df_parameters_1min['min_nomination']['2000-01-01 19':'2000-01-01 20'] = PARAMETERS['min_nomination_peak'] * PV_CAPACITY
df_parameters_1min['max_nomination'] = PARAMETERS['max_nomination'] * PV_CAPACITY
df_parameters_1min['min_production'] = PARAMETERS['min_production'] * PV_CAPACITY
df_parameters_1min['min_production']['2000-01-01 19':'2000-01-01 20'] = PARAMETERS['min_production_peak'] * PV_CAPACITY
df_parameters_1min['max_production'] = PARAMETERS['max_production'] * PV_CAPACITY
df_parameters_15min = df_parameters_1min.resample('15min').mean()
df_parameters_60min = df_parameters_1min.resample('60min').mean()

# select the parameters
PARAMETERS['df_params'] = df_parameters_15min

# ------------------------------------------------------------------------------------------------------------------
# 3. BESS PARAMETERS
PV_BESS_ratio = 100 # 100 * (BATTERY_CAPACITY / PV_CAPACITY) (%)
BATTERY_CAPACITY = (PV_BESS_ratio / 100) * PV_CAPACITY # (kWh)
SOC_INI = 0 # (kWh)
SOC_END = SOC_INI # (kWh)

SOC_MAX = BATTERY_CAPACITY # (kWh)
SOC_MIN = 0 #  (kWh)

CHARGE_EFFICIENCY = 0.95 # (%)
DISCHARGE_EFFICIENCY = 0.95 # (%)
CHARGING_POWER = BATTERY_CAPACITY # (kW)
DISCHARGING_POWER = BATTERY_CAPACITY # (kW)
HIGH_SOC_PRICE = 0 # (euros/kWh) Fee to use the BESS

bess_params = {"BESS_capacity": BATTERY_CAPACITY,  # (kWh)
               "soc_min": 0,  # (kWh)
               "soc_max": BATTERY_CAPACITY,  # (kWh)
               "soc_ini": 0,  # (kWh)
               "soc_end": 0,  # (kWh)
               "charge_eff": 0.95,  # (/)
               "discharge_eff": 0.95,  # (/)
               "charge_power": BATTERY_CAPACITY,  # (kW)
               "discharge_power": BATTERY_CAPACITY,  # (kW)
               "HIGH_SOC_PRICE": 0  # (euros/kWh)
               }

PARAMETERS['BESS'] = bess_params
