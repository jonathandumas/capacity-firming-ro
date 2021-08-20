# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from ro import read_file
from root_project import ROOT_DIR

import matplotlib.pyplot as plt
from ro.ro_simulator_configuration import PARAMETERS


class SP_dual_LP():
    """
    SP dual of the benders decomposition using gurobi.

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
        model = gp.Model("SP_dual_LP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create dual variables -> phi

        phi_charge = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_charge")
        phi_discharge = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_discharge")
        phi_Smin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smin")
        phi_Smax = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smax")
        phi_y = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_y")  # free dual of power balance
        phi_ymin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_ymin")
        phi_ymax = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_ymax")
        phi_ini = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_ini")  # free -> dual variable of a =
        phi_s = model.addVars(self.nb_periods - 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_s")  # free dual of BESS dynamics (=)
        phi_end = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_end")  # free -> dual variable of a =
        phi_devshort = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_devshort")
        phi_devlong = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_devlong")
        phi_PV = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_PV")
        if self.overproduction:
            # forbid overproduction
            phi_overprod = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_overprod")

        # -------------------------------------------------------------------------------------------------------------
        # 3. create objective
        obj_exp = 0
        for i in self.t_set:
            obj_exp += phi_charge[i] * self.charge_power + phi_discharge[i] * self.discharge_power - self.soc_min * phi_Smin[i] + self.soc_max * phi_Smax[i]
            obj_exp += self.max_production[i] * phi_ymax[i] - self.min_production[i] * phi_ymin[i]
            obj_exp += phi_devlong[i] * (self.x[i] + self.deadband_penalty) - phi_devshort[i] * (self.x[i] - self.deadband_penalty) + phi_PV[i] * self.pv_forecast[i]
        obj_exp += phi_ini * self.soc_ini + phi_end * self.soc_end
        if self.overproduction:
            for i in range(self.nb_periods):
                obj_exp += phi_overprod[i] * (self.x[i] + self.deadband_penalty)

        model.setObjective(obj_exp, GRB.MAXIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints

        # Bloc Constraints (18) from latex formulation -> constraints of the dual SP
        if self.overproduction:
            model.addConstrs((phi_y[i] - phi_ymin[i] + phi_ymax[i] - phi_devshort[i] + phi_devlong[i] + phi_overprod[i] == -self.period_hours * self.selling_price[i] for i in self.t_set), name='c_y')
        else:
            model.addConstrs((phi_y[i] - phi_ymin[i] + phi_ymax[i] - phi_devshort[i] + phi_devlong[i] == -self.period_hours * self.selling_price[i] for i in range(self.nb_periods)), name='c_y')

        model.addConstrs((- phi_devshort[i] <= self.penalty_factor * self.period_hours * self.selling_price[i] for i in self.t_set), name='c_devshort')
        model.addConstrs((- phi_devlong[i] <= self.penalty_factor * self.period_hours * self.selling_price[i] for i in self.t_set), name='c_devlong')

        # bloc constraints related to the BESS primal variables
        model.addConstr(phi_discharge[0] - phi_y[0] + phi_ini * self.period_hours / self.discharge_eff <= 0, name='c_discharge_first')  # time period 1
        model.addConstrs((phi_discharge[i] - phi_y[i] + phi_s[i - 1] * self.period_hours / self.discharge_eff <= 0 for i in range(1, self.nb_periods)), name='c_discharge')  # time period 2 to nb_periods
        model.addConstr(phi_charge[0] + phi_y[0] - phi_ini * self.period_hours * self.charge_eff <= 0, name='c_charge_last')  # time period 1
        model.addConstrs((phi_charge[i] + phi_y[i] - phi_s[i - 1] * self.period_hours * self.charge_eff <= 0 for i in range(1, self.nb_periods)), name='c_charge')  # time period 2 to nb_periods
        model.addConstr(-phi_Smin[0] + phi_Smax[0] + phi_ini - phi_s[0] <= - self.high_soc_price, name='c_s_first')  # time period 1 for phi_Smin/phi_Smax and time period 2 for phi_s
        model.addConstrs((-phi_Smin[i] + phi_Smax[i] + phi_s[i - 2] - phi_s[i - 1] <= -self.high_soc_price for i in range(2, self.nb_periods - 1)), name='c_s')  # time period 3 to nb_periods - 1
        model.addConstr(-phi_Smin[self.nb_periods - 1] + phi_Smax[self.nb_periods - 1] + phi_end + phi_s[self.nb_periods - 2] <= -self.high_soc_price, name='c_s_last')  # Last time period

        model.addConstrs((- phi_y[i] + phi_PV[i] <= 0 for i in range(self.nb_periods)), name='c_PV')

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, LogToConsole:bool=False):

        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        # self.model.setParam('OutputFlag', outputflag)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal

        varname = ['phi_charge', 'phi_discharge', 'phi_Smin', 'phi_Smax', 'phi_y', 'phi_ymin', 'phi_ymax', 'phi_s', 'phi_devshort', 'phi_devlong', 'phi_PV']
        if self.overproduction:
            varname +=['phi_overprod']

        for key in varname:
            solution[key] = []

        sol = m.getVars()
        solution['all_var'] = sol
        for v in sol:
            for key in varname:
                if v.VarName.split('[')[0] == key:
                    solution[key].append(v.x)

        solution['phi_ini'] = m.getVarByName("phi_ini").x
        solution['phi_end'] = m.getVarByName("phi_end").x
        solution['cut'] = solution['obj'] + sum(x * y for x, y in zip(solution['phi_devshort'], self.x)) - sum(x * y for x, y in zip(solution['phi_devlong'], self.x))

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

    # plt.figure()
    # plt.plot(engagement, label='x oracle')
    # plt.plot(pv_solution, label='Pm')
    # plt.plot(pv_forecast, label='Pp')
    # plt.ylim(-0.05 * PV_CAPACITY, PV_CAPACITY)
    # plt.legend()
    # plt.show()

    SP_dual = SP_dual_LP(pv_forecast=pv_forecast, engagement=engagement)
    SP_dual.export_model(dirname + 'SP_dual_LP')
    SP_dual.solve()
    solution_perfect = SP_dual.store_solution()

    print('objective %.2f' %(solution_perfect['obj']))


    plt.figure()
    plt.plot(solution_perfect['phi_PV'], label='phi_PV')
    # plt.ylim(-1, 0)
    plt.legend()
    plt.show()


    # Get dual values
    # for c in SP_dual.model.getConstrs():
    #     print('The dual value of %s : %g' % (c.constrName, c.pi))
