# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
import matplotlib.pyplot as plt


from ro import read_file
from ro.ro_simulator_configuration import PARAMETERS


class SP_primal_LP():
    """
    SP primal of the benders decomposition using gurobi.

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

    def __init__(self, pv_forecast:np.array, engagement:np.array):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)

        self.pv_forecast = pv_forecast  # (kW)
        self.x = engagement # (kW)
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
        model = gp.Model("SP_primal_LP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create second-stage variables -> y

        y_prod = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=-0, vtype=GRB.CONTINUOUS, name="y_prod")  # Production at the grid coupling point (injection > 0, withdrawal < 0) (kW)
        y_short_dev = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_short_dev") # y_short_dev >= - (y_prod - x-) with x- = x - deadband_deviation_kW
        y_long_dev = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_long_dev")   # y_long_dev  >= - (x+ - y_prod) with x+ = x + deadband_deviation_kW

        y_s = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_s")  # State of charge of the battery (kWh)
        y_charge = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_charge") # Charging power (kW)
        y_discharge = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_discharge")  # Discharging power (kW)
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")  # PV generation (kW)

        # -------------------------------------------------------------------------------------------------------------
        # 3. create objective
        objective = gp.quicksum(self.period_hours * self.selling_price[i] * (-y_prod[i] + self.penalty_factor * (y_short_dev[i] + y_long_dev[i]))  - self.high_soc_price * y_s[i] for i in range(self.nb_periods))
        model.setObjective(objective, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. create Second-stage constraints

        # max charge cst
        model.addConstrs((y_charge[i] <= self.charge_power for i in self.t_set), name='c_max_charge')
        # max discharge cst
        model.addConstrs((y_discharge[i] <= self.discharge_power for i in self.t_set), name='c_max_discharge')
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
            model.addConstrs((y_prod[i] <= (self.x[i] + self.deadband_penalty) for i in self.t_set), name='c_over_prod')

        # BESS dynamics first period
        model.addConstr((y_s[0] - self.period_hours * (self.charge_eff * y_charge[0] - y_discharge[0] / self.discharge_eff) == self.soc_ini), name='c_BESS_first_period')
        # BESS dynamics from second to last periods
        model.addConstrs((y_s[i] - y_s[i-1]- self.period_hours * (self.charge_eff * y_charge[i] - y_discharge[i] / self.discharge_eff) == 0 for i in range(1, self.nb_periods)), name='c_BESS_dynamics')
        # BESS dynamics last period
        model.addConstr((y_s[self.nb_periods-1]  == self.soc_end ), name='c_BESS_last_period')

        # Short penalty cst
        model.addConstrs((y_short_dev[i] >= (self.x[i] - self.deadband_penalty) - y_prod[i] for i in self.t_set), name='c_short_penalty')
        # Long penalty cst
        model.addConstrs((y_long_dev[i] >= y_prod[i] - (self.x[i] + self.deadband_penalty) for i in self.t_set), name='c_long_penalty')

        # PV generation cst
        model.addConstrs((y_PV[i] <= self.pv_forecast[i] for i in self.t_set), name='c_PV_generation')

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, outputflag:bool=False):

        t_solve = time.time()
        self.model.setParam('OutputFlag', outputflag)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status

        if solution['status'] == 2 or  solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (subject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            solution['obj'] = m.objVal

            varname = ['y_prod', 'y_short_dev', 'y_long_dev', 'y_s', 'y_charge', 'y_discharge', 'y_PV']
            for key in varname:
                solution[key] = []

            sol = m.getVars()
            solution['all_var'] = sol
            for v in sol:
                for key in varname:
                    if v.VarName.split('[')[0] == key:
                        solution[key].append(v.x)

        else:
            print('WARNING planner SP primal status %s -> problem not solved, objective is set to nan' %(solution['status']))
            # solutionStatus = 3: Model was proven to be infeasible.
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['obj'] = float('nan')

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


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = 'sandbox/ro/determinist/export/'

    pv_solution = pd.read_csv('data1/uliege/uliege_15min.csv', index_col=0, parse_dates=True)['Pm']['2019-08-22'].values
    pv_forecast = pd.read_csv('data1/uliege/uliege_pv_prediction_python_15min.csv', parse_dates=True, index_col=0)['2019-08-22'].values

    engagement = read_file(dir=dirname, name='sol_LP_oracle')

    SP_primal = SP_primal_LP(pv_forecast=pv_solution, engagement=engagement)
    SP_primal.export_model(dirname + 'SP_primal_LP')
    SP_primal.solve()
    solution_perfect = SP_primal.store_solution()

    print('objective SP primal %.2f' %(solution_perfect['obj']))

    plt.figure()
    plt.plot(solution_perfect['y_charge'], label='y_charge')
    plt.plot(solution_perfect['y_discharge'], label='y_discharge')
    plt.plot(solution_perfect['y_s'], label='y s')
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(engagement, label='x oracle')
    plt.plot(solution_perfect['y_prod'], label='y prod')
    plt.plot(pv_solution, label= 'Pm')
    plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    plt.legend()
    plt.show()

    # Get dual values
    # for c in SP_primal.model.getConstrs():
    #     print('The dual value of %s : %g' % (c.constrName, c.pi))
