# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
from ro.ro_simulator_configuration import PARAMETERS


class CCG_MP():
    """
    CCG = Column-and-Constraint Generation.
    MP = Master Problem of the CCG algorithm.
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

    def update_MP(self, pv_trajectory:np.array, iteration:int):
        """
        Add the second-stage variables at CCG iteration i.
        :param MP: MP to update in the CCG algorithm.
        :param pv_trajectory: PV trajectory computed by the SP at iteration i.
        :param iteration: update at iteration i.
        :return: the model is directly updated
        """
        # -------------------------------------------------------------------------------------------------------------
        # 1 Second-stage variables -> y
        # Production at the grid coupling point (injection > 0, withdrawal < 0) (kW)
        y_prod = self.model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=-0, vtype=GRB.CONTINUOUS, name="y_prod_" + str(iteration))
        # y_short_dev >= - (y_prod - x-) with x- = x - deadband_deviation_kW
        y_short_dev = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_short_dev_" + str(iteration))
        # y_long_dev  >= - (x+ - y_prod) with x+ = x + deadband_deviation_kW
        y_long_dev = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_long_dev_" + str(iteration))

        # State of charge of the battery (kWh)
        y_s = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_s_" + str(iteration))
        # Charging power (kW)
        y_charge = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_charge_" + str(iteration))
        # Discharging power (kW)
        y_discharge = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_discharge_" + str(iteration))
        # PV generation (kW)
        y_PV = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV_" + str(iteration))
        # binary variable -> y_b = 1 -> charge / y_b = 0 -> discharge
        y_b = self.model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="y_b_" + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 2 Add the constraint related to the objective
        objective = gp.quicksum(self.period_hours * self.selling_price[i] * (-y_prod[i] + self.penalty_factor * (y_short_dev[i] + y_long_dev[i])) - self.high_soc_price * y_s[i] for i in self.t_set)
        # theta = MP.model.getVars()[nb_periods] is the 96-th = nb_periods variable of the MP model
        self.model.addConstr(self.model.getVars()[self.nb_periods] >= objective, name='theta_' + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 3 Add the constraint related to the feasbility domain of the secondstage variables -> y

        # BESS
        # max charge cst
        self.model.addConstrs((y_charge[i] <= y_b[i] * self.charge_power for i in self.t_set), name='c_max_charge_' + str(iteration))
        # max discharge cst
        self.model.addConstrs((y_discharge[i] <= (1 - y_b[i]) * self.discharge_power for i in self.t_set),
                            name='c_max_discharge_' + str(iteration))
        # min soc cst
        self.model.addConstrs((y_s[i] >= self.soc_min for i in self.t_set), name='c_min_s_' + str(iteration))
        # min soc cst
        self.model.addConstrs((y_s[i] <= self.soc_max for i in self.t_set), name='c_max_s_' + str(iteration))

        # power balance equation
        self.model.addConstrs((y_prod[i] - y_PV[i] - (y_discharge[i] - y_charge[i]) == 0 for i in self.t_set),
                            name='c_power_balance_eq_' + str(iteration))
        # min production
        self.model.addConstrs((y_prod[i] >= self.min_production[i] for i in self.t_set),
                            name='c_min_y_prod_' + str(iteration))
        # max production
        self.model.addConstrs((y_prod[i] <= self.max_production[i] for i in self.t_set),
                            name='c_max_y_prod_' + str(iteration))
        if self.overproduction:
            # forbid overproduction: self.model.getVars() -> return all variables of the model, but the x variable are the first one from index 0 to 95
            self.model.addConstrs((y_prod[i] <= (self.model.getVars()[i] + self.deadband_penalty) for i in self.t_set),
                                name='c_over_prod_' + str(iteration))

        # BESS dynamics first period
        self.model.addConstr((y_s[0] - self.period_hours * (
                    self.charge_eff * y_charge[0] - y_discharge[0] / self.discharge_eff) == self.soc_ini),
                           name='c_BESS_first_period_' + str(iteration))
        # BESS dynamics from second to last periods
        self.model.addConstrs((y_s[i] - y_s[i - 1] - self.period_hours * (
                    self.charge_eff * y_charge[i] - y_discharge[i] / self.discharge_eff) == 0 for i in
                             range(1, self.nb_periods)), name='c_BESS_dynamics_' + str(iteration))
        # BESS dynamics last period
        self.model.addConstr((y_s[self.nb_periods - 1] == self.soc_end), name='c_BESS_last_period_' + str(iteration))

        # Short penalty cst: self.model.getVars() -> return all variables of the model, but the x variable are the first one from index 0 to 95
        self.model.addConstrs(
            (y_short_dev[i] >= (self.model.getVars()[i] - self.deadband_penalty) - y_prod[i] for i in self.t_set),
            name='c_short_penalty_' + str(iteration))
        # Long penalty cst: self.model.getVars() -> return all variables of the model, but the x variable are the first one from index 0 to 95
        self.model.addConstrs(
            (y_long_dev[i] >= y_prod[i] - (self.model.getVars()[i] + self.deadband_penalty) for i in self.t_set),
            name='c_long_penalty_' + str(iteration))

        # PV generation cst
        self.model.addConstrs((y_PV[i] <= pv_trajectory[i] for i in self.t_set), name='c_PV_generation_' + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 4. Store the added variables to the MP in a new dict
        self.allvar['var_' + str(iteration)] = dict()
        self.allvar['var_' + str(iteration)]['y_prod'] = y_prod
        self.allvar['var_' + str(iteration)]['y_short_dev'] = y_short_dev
        self.allvar['var_' + str(iteration)]['y_long_dev'] = y_long_dev
        self.allvar['var_' + str(iteration)]['y_s'] = y_s
        self.allvar['var_' + str(iteration)]['y_charge'] = y_charge
        self.allvar['var_' + str(iteration)]['y_discharge'] = y_discharge
        self.allvar['var_' + str(iteration)]['y_PV'] = y_PV
        self.allvar['var_' + str(iteration)]['y_b'] = y_b

        # -------------------------------------------------------------------------------------------------------------
        # 5. Update model to implement the modifications
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
        if solution['status'] == 2 or solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (subject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            # 0 dimensional variables
            solution['theta'] = self.allvar['theta'].X
            # 1D variable
            solution['x'] = [self.allvar['x'][t].X for t in self.t_set]
            solution['obj'] = m.objVal
        else:
            print('WARNING MP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            # solutionStatus = 3: Model was proven to be infeasible.
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['obj'] = float('nan')

        # 3. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution

    def update_sol(self, MP_sol:dict, i:int):
        """
        Add the solution of the 1 dimensional variables at iteration i.
        :param MP_sol: solution of the MP model at iteration i.
        :param i: index of interation.
        :return: update directly the dict.
        """
        MP_status = MP_sol['status']
        if MP_status == 2 or MP_status == 9:
            MP_sol['var_' + str(i)] = dict()
            # add the solution of the 1 dimensional variables at iteration
            for var in ['y_prod', 'y_short_dev', 'y_long_dev', 'y_s', 'y_charge', 'y_discharge', 'y_PV', 'y_b']:
                MP_sol['var_' + str(i)][var] = [self.allvar['var_' + str(i)][var][t].X for t in self.t_set]
        else:
            print('WARNING planner MP status %s -> problem not solved, cannot retrieve solution')

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