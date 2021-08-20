# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd

from root_project import ROOT_DIR

import gurobipy as gp
from gurobipy import GRB

from ro.ro_simulator_configuration import PARAMETERS


class BD_MP():
    """
    BD = Benders Dual.
    MP = Master Problem of the Benders dual cutting plane algorithm.
    The MP is a Linear Programming.

    :ivar nb_periods: number of market periods (-)
    :ivar df_parameters: pd.DataFrame with price, deadband_nomination, min_nomination, max_nomination, min_production, max_production
                        shape = (nb_market periods, 6)
    :ivar deadband_nomination: deadband between two consecutive nominations, % of the total installed capacity (kW)
          different between of-peak and peak hours
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)
    """

    def __init__(self, pv_forecast:np.array=None):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)

        # parameters required for the MP in the CCG algorithm
        self.pv_forecast = pv_forecast  # (kW)
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
        model = gp.Model("MP")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create First-stage variables -> x
        x = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x") # Nomination at the grid coupling point (injection > 0, withdrawal < 0) (kW)
        theta = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name="theta")

        # -------------------------------------------------------------------------------------------------------------
        # 3. create objective
        model.setObjective(theta, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints

        # 4.1 First stage constraints
        # engagement min cst
        model.addConstrs((x[i] >= self.min_nomination[i] for i in self.t_set), name='c_xmin')
        # engagement max cst
        model.addConstrs((x[i] <= self.max_nomination[i] for i in self.t_set), name='c_xmax')
        # engagement ramping constraint up -> skip first period -> nb periods -1 constraints
        model.addConstrs((x[i] - x[i - 1] <= self.deadband_nomination[i] for i in range(1, self.nb_periods)), name='c_x_rampingUp')
        # engagement ramping constraint down -> skip first period -> nb periods -1 constraints
        model.addConstrs((x[i - 1] - x[i] <= self.deadband_nomination[i] for i in range(1, self.nb_periods)), name='c_x_rampingDown')

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['x'] = x
        self.allvar['theta'] = theta

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def update_MP(self, SP_dual_sol:dict, iteration:int):
        """
        Add an optimality cut to MP at iteration i.
        :param MP: MP to update in the CCG algorithm.
        :param SP_dual_sol: solution of the SP in the BD algorithm computed at iteration i.
        :param iteration: update at iteration i.
        :return: the model is directly updated
        """
        # x1 to x96 are the 0th to the 95th variables of the MP model
        # -> theta is the variable with index 96 = nb_periods
        lin_exp = gp.quicksum(SP_dual_sol['phi_devlong'][i] * self.model.getVars()[i] - SP_dual_sol['phi_devshort'][i] * self.model.getVars()[i] for i in range(self.nb_periods))
        # theta = MP.model.getVars()[nb_periods] is the 96-th = nb_periods variable of the MP model
        self.model.addConstr(self.model.getVars()[self.nb_periods] >= SP_dual_sol['cut'] + lin_exp, name='c_cut_' + str(iteration))

        # Update model to implement the modifications
        self.model.update()

    def solve(self, LogToConsole:bool=False):

        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal

        # 0 dimensional variables
        solution['theta'] = self.allvar['theta'].X
        # 1D variable
        solution['x'] = [self.allvar['x'][t].X for t in self.t_set]

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