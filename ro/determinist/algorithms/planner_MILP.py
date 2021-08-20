# -*- coding: UTF-8 -*-
import math
import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from ro.utils import dump_file, build_observations, build_point_forecast
from root_project import ROOT_DIR

import matplotlib.pyplot as plt
from ro.ro_simulator_configuration import PARAMETERS

class Planner_MILP():
    """
    MILP capacity firming formulation: binary variables to avoid simultaneous charge and discharge.

    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar deadband_penalty: deadband deviation between production and nomination, % of the total installed capacity (kW)
          constant for all market periods
    :ivar pv_forecast: PV point forecasts (kW)
    :ivar engagement: Nomination to the grid (injection > 0, withdrawal < 0) (kW)
          shape = (nb_market periods,)
    :ivar df_parameters: pd.DataFrame with price, deadband_nomination, min_nomination, max_nomination, min_production, max_production
                        shape = (nb_market periods, 6)
    :ivar selling_price: selling price (€/ kWh)
          different between of-peak and peak hours
    :ivar deadband_nomination: deadband between two consecutive nominations, % of the total installed capacity (kW)
          different between of-peak and peak hours
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)


    """

    def __init__(self, pv_forecast:np.array, engagement:np.array=None):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)

        self.pv_forecast = pv_forecast  # (kW)
        self.engagement = engagement # (kW)
        self.PVcapacity = PARAMETERS['pv_capacity'] # (kWp)
        self.tol_penalty = PARAMETERS['tol_penalty'] # (%)
        self.deadband_penalty = self.tol_penalty * self.PVcapacity # (kW)

        self.penalty_factor = PARAMETERS['penalty_factor']  # penalty factor
        self.overproduction = PARAMETERS['OVERPRODUCTION']  # forbid overproduction
        if all([item in PARAMETERS['df_params'].columns for item in ['price', 'deadband_nomination', 'min_nomination', 'max_nomination', 'min_production','max_production']]):
            self.selling_price = PARAMETERS['df_params']['price'].values  # (€/ kWh)
            self.deadband_nomination = PARAMETERS['df_params']['deadband_nomination'].values  # (kW/period)
            self.min_nomination = PARAMETERS['df_params']['min_nomination'].values  # (kW)
            self.max_nomination = PARAMETERS['df_params']['max_nomination'].values  # (kW)
            self.min_production = PARAMETERS['df_params']['min_production'].values  # (kW)
            self.max_production = PARAMETERS['df_params']['max_production'].values  # (kW)
        else:
            print("df_parameters is not ok ")

        # BESS parameters
        self.BESScapacity = PARAMETERS['BESS']['BESS_capacity']  # (kWh)
        self.soc_ini = PARAMETERS['BESS']['soc_ini']  # (kWh)
        self.soc_end = PARAMETERS['BESS']['soc_end']  # (kWh)
        self.soc_min = PARAMETERS['BESS']['soc_min']  # (kWh)
        self.soc_max = PARAMETERS['BESS']['soc_max']  # (kWh)
        self.charge_eff = PARAMETERS['BESS']['charge_eff']  # (/)
        self.discharge_eff = PARAMETERS['BESS']['discharge_eff']  # (/)
        self.charge_power = PARAMETERS['BESS']['charge_power']  # (kW)
        self.discharge_power = PARAMETERS['BESS']['discharge_power']  # (kW)
        self.high_soc_price = PARAMETERS['BESS']['HIGH_SOC_PRICE']   # (euros/kWh) -> fictive price to incentivize to charge BESS

        self.time_building_model = None
        self.time_solving_model = None

        # Create model
        self.model = self.create_model()

        # Solve model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem.
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. create model
        model = gp.Model("planner_MILP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create variables
        # 2.1 First-stage variables -> x

        x = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x") # Nomination at the grid coupling point (injection > 0, withdrawal < 0) (kW)
        if self.engagement is not None:
            for i in self.t_set:
                x[i].setAttr("ub", self.engagement[i])
                x[i].setAttr("lb", self.engagement[i])

        # 2.2 Second-stage variables -> y

        y_prod = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=-0, vtype=GRB.CONTINUOUS, name="y_prod")  # Production at the grid coupling point (injection > 0, withdrawal < 0) (kW)
        y_short_dev = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_short_dev") # y_short_dev >= - (y_prod - x-) with x- = x - deadband_deviation_kW
        y_long_dev = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_long_dev")   # y_long_dev  >= - (x+ - y_prod) with x+ = x + deadband_deviation_kW

        y_s = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_s")  # State of charge of the battery (kWh)
        y_charge = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_charge") # Charging power (kW)
        y_discharge = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_discharge")  # Discharging power (kW)
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")  # PV generation (kW)

        y_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="y_b")  # binary variable -> y_b = 1 -> charge / y_b = 0 -> discharge

        # -------------------------------------------------------------------------------------------------------------
        # 3. create objective
        objective = gp.quicksum(self.period_hours * self.selling_price[i] * (-y_prod[i] + self.penalty_factor * (y_short_dev[i] + y_long_dev[i])) - self.high_soc_price * y_s[i] for i in self.t_set)
        model.setObjective(objective, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. create constraints

        # 4.1 First stage constraints
        # engagement min cst
        model.addConstrs((x[i] >= self.min_nomination[i] for i in self.t_set), name='c_xmin')
        # engagement max cst
        model.addConstrs((x[i] <= self.max_nomination[i] for i in self.t_set), name='c_xmax')
        # engagement ramping constraint up -> skip first period -> nb periods -1 constraints
        model.addConstrs((x[i] - x[i - 1] <= self.deadband_nomination[i] for i in range(1, self.nb_periods)), name='c_x_rampingUp')
        # engagement ramping constraint down -> skip first period -> nb periods -1 constraints
        model.addConstrs((x[i - 1] - x[i] <= self.deadband_nomination[i] for i in range(1, self.nb_periods)), name='c_x_rampingDown')

        # 4.2 Second-stage constraints
        # max charge cst
        model.addConstrs((y_charge[i] <= y_b[i] * self.charge_power for i in self.t_set), name='c_max_charge')
        # max discharge cst
        model.addConstrs((y_discharge[i] <= (1 - y_b[i]) * self.discharge_power for i in self.t_set), name='c_max_discharge')
        # min soc cst
        model.addConstrs((y_s[i] >= self.soc_min for i in self.t_set), name='c_min_s')
        # min soc cst
        model.addConstrs((y_s[i] <= self.soc_max for i in self.t_set), name='c_max_s')

        # power balance equation
        model.addConstrs((y_prod[i] - y_PV[i] - (y_discharge[i] - y_charge[i]) == 0 for i in self.t_set), name='c_power_balance_eq')
        # min production
        model.addConstrs((y_prod[i] >= self.min_production[i] for i in self.t_set), name='c_min_y_prod')
        # max production
        model.addConstrs((y_prod[i] <= self.max_production[i] for i in self.t_set), name='c_max_y_prod')
        if self.overproduction:
            # forbid overproduction
            model.addConstrs((y_prod[i] <= (x[i] + self.deadband_penalty) for i in self.t_set), name='c_over_prod')


        # BESS dynamics first period
        model.addConstr((y_s[0] - self.period_hours * (self.charge_eff * y_charge[0] - y_discharge[0] / self.discharge_eff) == self.soc_ini), name='c_BESS_first_period')
        # BESS dynamics from second to last periods
        model.addConstrs((y_s[i] - y_s[i-1]- self.period_hours * (self.charge_eff * y_charge[i] - y_discharge[i] / self.discharge_eff) == 0 for i in range(1, self.nb_periods)), name='c_BESS_dynamics')
        # BESS dynamics last period
        model.addConstr((y_s[self.nb_periods-1]  == self.soc_end ), name='c_BESS_last_period')

        # Short penalty cst
        model.addConstrs((y_short_dev[i] >= (x[i] - self.deadband_penalty) - y_prod[i] for i in self.t_set), name='c_short_penalty')
        # Long penalty cst
        model.addConstrs((y_long_dev[i] >= y_prod[i] - (x[i] + self.deadband_penalty) for i in self.t_set), name='c_long_penalty')

        # PV generation cst
        model.addConstrs((y_PV[i] <= self.pv_forecast[i] for i in self.t_set), name='c_PV_generation')


        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['x'] = x
        self.allvar['y_prod'] = y_prod
        self.allvar['y_short_dev'] = y_short_dev
        self.allvar['y_long_dev'] = y_long_dev
        self.allvar['y_s'] = y_s
        self.allvar['y_charge'] = y_charge
        self.allvar['y_discharge'] = y_discharge
        self.allvar['y_PV'] = y_PV
        self.allvar['y_b'] = y_b

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):

        t_solve = time.time()

        self.model.setParam('LogToConsole', LogToConsole) # no log in the console if set to False
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        # self.model.setParam('DualReductions', 0) # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.

        # If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
        # If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
        # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

        self.model.setParam('LogFile', logfile) # no log in file if set to ""
        self.model.setParam('Threads', Threads) # Default value = 0 -> use all threads

        self.model.optimize()
        self.solver_status = self.model.status
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            solution['obj'] = m.objVal

            # 1 dimensional variables
            for var in ['x', 'y_prod', 'y_short_dev', 'y_long_dev', 'y_s', 'y_charge', 'y_discharge', 'y_PV', 'y_b']:
                solution[var] = [self.allvar[var][t].X for t in self.t_set]
        else:
            print('WARNING planner MILP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            solution['obj'] = math.nan

        # 3. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution

    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)
        # self.model.write("%s.mps" % filename)

# Validation set
VS = 'VS1' # 'VS1', 'VS2

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = 'ro/determinist/export/'

    # load data
    pv_solution = build_observations()
    pv_dad = build_point_forecast()

    # Select a paticular day of the dataset
    day = '2020-05-26'
    pv_solution_day = pv_solution.loc[day].values
    pv_forecast_day = pv_dad.loc[day].values

    # Plot point forecasts vs observations
    FONTSIZE = 20
    plt.figure()
    plt.plot(pv_solution_day, label='observations')
    plt.plot(pv_forecast_day, label='forecast')
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()

    # MILP planner with perfect forecasts -> oracle
    planner_oracle = Planner_MILP(pv_forecast=pv_solution_day)
    planner_oracle.export_model(dirname + 'planner_MILP')
    planner_oracle.solve()
    solution_oracle = planner_oracle.store_solution()

    print('objective oracle %.2f' % (solution_oracle['obj']))

    # MILP planner with point forecasts
    planner = Planner_MILP(pv_forecast=pv_solution_day)
    planner.solve()
    solution = planner.store_solution()
    # dump_file(dir=dirname, name='solution_point_forecasts', file=solution['x'])

    print('objective point forecasts %.2f' % (solution['obj']))

    plt.figure()
    plt.plot(solution_oracle['x'], label='x oracle')
    plt.plot(solution['x'], label='x LSTM point')
    # plt.plot(pv_solution_day, label= 'Pm')
    # plt.plot(pv_forecast_day, label= 'Pp')
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    plt.title('MILP formulation')
    plt.legend(fontsize=FONTSIZE)
    # plt.savefig(dirname+ 'MILP_oracle_vs_point.pdf')
    plt.show()

