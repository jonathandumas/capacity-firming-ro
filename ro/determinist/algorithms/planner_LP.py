# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR

import matplotlib.pyplot as plt
from ro.ro_simulator_configuration import PARAMETERS
from ro import dump_file


class Planner_LP():
    """
    LP capacity firming formulation: no binary variables to ensure not simultaneous charge and discharge.

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
        model = gp.Model("planner_LP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create variables
        # 2.1 First-stage variables -> x
        x = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x") # Nomination at the grid coupling point (injection > 0, withdrawal < 0) (kW)

        # Set the engagement x variable to the values in self.engagement
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

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, outputflag:bool=False):

        t_solve = time.time()
        self.model.setParam('OutputFlag', outputflag)
        self.model.optimize()
        self.solver_status = self.model.status
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal

        # 1 dimensional variables
        for var in ['x', 'y_prod', 'y_short_dev', 'y_long_dev', 'y_s', 'y_charge', 'y_discharge', 'y_PV']:
            solution[var] = [self.allvar[var][t].X for t in self.t_set]

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
    day = '2019-08-22' # sunny
    # day = '2019-08-18' # cloudy

    pv_solution = pd.read_csv('data1/uliege/uliege_15min.csv', index_col=0, parse_dates=True)['Pm'][day].values
    pv_forecast = pd.read_csv('data1/uliege/uliege_pv_prediction_python_15min.csv', parse_dates=True, index_col=0)[day].values

    planner_perfect = Planner_LP(pv_forecast=pv_solution)
    planner_perfect.export_model(dirname + 'LP')
    planner_perfect.solve()
    solution_perfect = planner_perfect.store_solution()

    print('objective oracle %.2f' % (solution_perfect['obj']))

    dump_file(dir=dirname, name='sol_LP_oracle'+day, file=solution_perfect['x'])


    plt.figure()
    plt.plot(solution_perfect['y_charge'], label='y_charge')
    plt.plot(solution_perfect['y_discharge'], label='y_discharge')
    plt.plot(solution_perfect['y_s'], label='y s')
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
    plt.legend()
    plt.show()

    planner = Planner_LP(pv_forecast=pv_forecast)
    planner.solve()
    solution = planner.store_solution()
    dump_file(dir=dirname, name='sol_LP_PVUSA'+day, file=solution['x'])

    print('objective PVUSA %.2f' % (solution['obj']))

    plt.figure()
    plt.plot(solution_perfect['x'], label='x oracle')
    plt.plot(solution['x'], label='x PVUSA')
    plt.plot(pv_solution, label= 'Pm')
    plt.plot(pv_forecast, label= 'Pp')
    plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    plt.title('LP formulation')
    plt.legend()
    plt.savefig(dirname+ 'LP_oracle_vs_PVUSA.pdf')
    plt.show()

