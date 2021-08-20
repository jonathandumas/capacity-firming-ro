# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB
from numpy.testing import assert_almost_equal

from ro import build_observations, build_point_forecast, build_point_intra_forecast, load_lstm_quantiles, read_file
from ro.determinist.algorithms import Planner_MILP
from root_project import ROOT_DIR
from ro.ro_simulator_configuration import PARAMETERS


class Controller_MILP():
    """
    MILP controller capacity firming formulation: binary variables to avoid simultaneous charge and discharge.

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

    def __init__(self, soc_ini:float, df_params:pd.DataFrame, pv_forecast:np.array, engagement:np.array=None):
        """
        Init the controller.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = pv_forecast.shape[0]
        self.t_set = range(self.nb_periods)
        self.nb_market_periods = range(1, self.nb_periods + 1)


        self.pv_forecast = pv_forecast  # (kW)
        self.engagement = engagement # (kW)
        self.PVcapacity = PARAMETERS['pv_capacity'] # (kWp)
        self.tol_penalty = PARAMETERS['tol_penalty'] # (%)
        self.deadband_penalty = self.tol_penalty * self.PVcapacity # (kW)

        self.penalty_factor = PARAMETERS['penalty_factor']  # penalty factor
        self.overproduction = PARAMETERS['OVERPRODUCTION']  # forbid overproduction
        if all([item in df_params.columns for item in ['price', 'deadband_nomination', 'min_nomination', 'max_nomination', 'min_production','max_production']]):
            self.selling_price = df_params['price'].values  # (€/ kWh)
            self.deadband_nomination = df_params['deadband_nomination'].values  # (kW/period)
            self.min_nomination = df_params['min_nomination'].values  # (kW)
            self.max_nomination = df_params['max_nomination'].values  # (kW)
            self.min_production = df_params['min_production'].values  # (kW)
            self.max_production = df_params['max_production'].values  # (kW)
        else:
            print("df_parameters is not ok ")

        # BESS parameters
        self.BESScapacity = PARAMETERS['BESS']['BESS_capacity']  # (kWh)
        self.soc_ini = soc_ini  # (kWh)
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
        objective = gp.quicksum(self.period_hours * self.selling_price[i] * (-y_prod[i] + self.penalty_factor * (y_short_dev[i] + y_long_dev[i])) for i in self.t_set)
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
        solution['obj'] = m.objVal

        # 1 dimensional variables -> store only the results of the first period
        for var in ['x', 'y_prod', 'y_short_dev', 'y_long_dev', 'y_s', 'y_charge', 'y_discharge', 'y_PV', 'y_b']:
            solution[var] = self.allvar[var][0].X

        # 2. Net revenue of the first period
        # solution['net_revenue'] = self.selling_price[0] * self.period_hours * (solution['y_prod'] - solution['y_short_dev'] - solution['y_long_dev'])
        solution['obj_bis'] = [-self.selling_price[t] * self.period_hours * (self.allvar['y_prod'][t].X - self.penalty_factor * (self.allvar['y_short_dev'][t].X + self.allvar['y_long_dev'][t].X)) for t in self.t_set]
        solution['net_revenue'] = -solution['obj_bis'][0]

        assert_almost_equal(solution["obj"], sum(solution["obj_bis"]), decimal=5)

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


def controller_loop(pv_forecast: np.array, pv_solution: np.array, engagement: np.array, solver_param: dict):
    """
    Controller loop from the first period to the last.
    :param pv_forecast:
    :param engagement: engagement plan.
    :return: solution into a list.
    """

    s_ini = PARAMETERS['BESS']['soc_ini']  # (kWh) state of charge at the first period of the day
    sol_controller_point = []
    for i in range(0, 96):
        # make a copy of the PV forecast
        pv_point_tmp = pv_forecast.copy()
        # Replace the PV forecast at t by the PV measurement at t.
        pv_point_tmp[i] = pv_solution[i]
        # Solve the optimization problem from t to the end of the day
        controller_point = Controller_MILP(soc_ini=s_ini, pv_forecast=pv_point_tmp[i:], df_params=PARAMETERS['df_params'].iloc[i:], engagement=engagement[i:])
        controller_point.solve(Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
        sol = controller_point.store_solution()
        s_ini = sol['y_s'] # update the soc for the next period
        sol_controller_point.append(sol)
    return sol_controller_point

# Benders starting point
x_start = 'median' # zero, median, oracle, point

day_list = ['2020-04-18', '2020-05-05', '2020-05-07', '2020-08-02']
day = day_list[1]

quantile = 0
GAMMA = 48

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/ro/determinist/export/'

    # TODO 1 a new point forecast file each six hours to use intraday updates
    # TODO 2 sensitivity analysis on the uncertainty budget gamma and the quantile
    # load data
    pv_solution = build_observations()
    pv_dad = build_point_forecast()
    pv_intra = build_point_intra_forecast()

    # Load quantile forecasts: from 0 to 8 -> median = quantile 4
    pv_quantile = load_lstm_quantiles()

    # Select a paticular day of the dataset
    pv_solution_day =pv_solution.loc[day].values
    pv_dad_day = pv_dad.loc[day].values
    pv_intra_day = pv_intra.loc[day].values
    pv_quantile_day = pv_quantile.loc[day].values.reshape(96, 9)

    # Plot point forecasts vs observations
    FONTSIZE = 20
    plt.figure()
    plt.plot(pv_solution_day, label='observations')
    plt.plot(pv_dad_day, label='dad')
    plt.plot(pv_intra_day, label='intra 00:00')
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.ylim(0, PARAMETERS['pv_capacity'])
    plt.tight_layout()
    plt.show()

    # Store the forecasts into a dict
    pv_forecast_dict = dict()
    pv_forecast_dict['oracle'] = pv_solution_day
    pv_forecast_dict['point'] = pv_dad_day

    # Compute several engagement plans by solving the MILP formulation with different forecasts
    sol_planner_dict = dict()
    for key in ['oracle', 'point']:
        planner = Planner_MILP(pv_forecast=pv_forecast_dict[key])
        planner.solve()
        sol_planner_dict[key] = planner.store_solution()['x']

        # print('Planner objective %s %.2f (euros)' % (key, sol_planner_dict[key]['obj']))

    # Benders planning
    dirname_benders = 'sandbox/ro/robust/export/' + day + '_quantile_' + str(quantile) + '/gamma_' + str(GAMMA) + '/'
    sol_planner_dict['benders'] = read_file(dir=dirname_benders, name='MP_benders_last_x_' + x_start)

    # Oracle controller -> perfect forecasts
    for key in ['oracle', 'point', 'benders']:
        controller_oracle = Planner_MILP(pv_forecast=pv_solution_day, engagement=sol_planner_dict[key])
        controller_oracle.solve()
        sol_controller_oracle = controller_oracle.store_solution()
        print('%s-oracle %.2f (euros)' % (key, sol_controller_oracle['obj']))

    # Point controller -> using intra pv point forecasts
    results = dict()
    for key in ['oracle', 'point', 'benders']:
        sol_controller_point = controller_loop(pv_forecast=pv_intra_day, pv_solution=pv_solution_day, engagement=sol_planner_dict[key])

        results[key] = pd.DataFrame()
        for col in ['net_revenue', 'y_prod', 'y_short_dev', 'y_long_dev', 'y_s', 'y_charge', 'y_discharge', 'y_PV', 'time_total']:
            results[key][col] = [sol[col] for sol in sol_controller_point]
        results[key]['penalty'] = [sol['y_short_dev'] + sol['y_long_dev'] for sol in sol_controller_point]

        print('%s-point net revenue %.2f (euros)' % (key, results[key]['net_revenue'].sum()))
        # print('%s-point penalty %.2f (euros)' % (key, results[key]['penalty'].sum()))

    plt.figure()
    for key in ['oracle', 'point', 'benders']:
        plt.plot(results[key]['y_s'].values, label='y_s '+key)
    plt.ylabel('kWh', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'] )
    plt.legend(fontsize=FONTSIZE)
    plt.show()

    plt.figure()
    for key in ['oracle', 'point', 'benders']:
        plt.plot(results[key]['y_prod'].values, label='y_prod '+key)
    plt.plot(sol_planner_dict['oracle'], label='x oracle')
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    plt.legend(fontsize=FONTSIZE)
    plt.show()