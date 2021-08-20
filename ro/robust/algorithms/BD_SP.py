# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt

from ro import read_file
from ro.determinist.algorithms import SP_primal_LP
from ro.ro_simulator_configuration import PARAMETERS
from root_project import ROOT_DIR

class BD_SP():
    """
    BD = Benders Dual.
    SP = Sub Problem of the Benders dual cutting plane algorithm.
    SP = Max-min problem that is reformulated as a single max problem by taking the dual.
    The resulting maximization problem is bilinear and is linearized using big-M's values.
    The final reformulated SP is a MILP due to binary variables related to the uncertainty set.

    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar deadband_penalty: deadband deviation between production and nomination, % of the total installed capacity (kW)
          constant for all market periods
    :ivar pv_forecast: PV point forecasts (kW)
    :ivar x: Nomination to the grid (injection > 0, withdrawal < 0) (kW)
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

    def __init__(self, pv_forecast:np.array, max_dev:np.array, engagement:np.array, gamma:float=0, heuristic:bool=False, M_neg:float=None):
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
        self.max_dev = max_dev  # (kW) The maximal deviation between the max and min PV uncertainty set bounds
        self.gamma = gamma # uncertainty budget <= self.nb_periods, gamma = 0 -> no uncertainty
        self.heuristic = heuristic
        self.M_neg = M_neg # big-M value

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
        model = gp.Model("SP_dual_MILP")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create dual variables -> phi

        # 2.1 Continuous variables
        phi_charge = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_cha")
        phi_discharge = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_discharge")
        phi_Smin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smin")
        phi_Smax = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smax")
        phi_y = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_y")  # free dual of power balance
        phi_ymin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=-0, name="phi_ymin")
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

        # 2.2 Binary variables related to the uncertainty set
        z_neg = model.addVars(self.nb_periods, vtype=GRB.BINARY, obj=0, name="z_neg")

        # 2.3 Continuous variables use for the linearization of the bilinear terms
        alpha_neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="alpha_neg")

        # -------------------------------------------------------------------------------------------------------------
        # 3. create objective
        obj_exp = 0
        for i in self.t_set:
            obj_exp += phi_charge[i] * self.charge_power + phi_discharge[i] * self.discharge_power - self.soc_min * phi_Smin[i] + self.soc_max * phi_Smax[i]
            obj_exp += self.max_production[i] * phi_ymax[i] - self.min_production[i] * phi_ymin[i]
            obj_exp += phi_devlong[i] * (self.x[i] + self.deadband_penalty) - phi_devshort[i] * (self.x[i] - self.deadband_penalty) + phi_PV[i] * self.pv_forecast[i]
            obj_exp += - alpha_neg[i] * self.max_dev[i] # uncertainty set
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
            model.addConstrs((phi_y[i] - phi_ymin[i] + phi_ymax[i] - phi_devshort[i] + phi_devlong[i] == -self.period_hours * self.selling_price[i] for i in self.t_set), name='c_y')
        model.addConstrs((- phi_devshort[i] <= self.penalty_factor *  self.period_hours * self.selling_price[i] for i in self.t_set), name='c_devshort')
        model.addConstrs((- phi_devlong[i] <= self.penalty_factor * self.period_hours * self.selling_price[i] for i in self.t_set), name='c_devlong')

        # bloc constraints related to the BESS primal variables
        model.addConstr(phi_discharge[0] - phi_y[0] + phi_ini * self.period_hours / self.discharge_eff <= 0, name='c_discharge_first')  # time period 1
        model.addConstrs((phi_discharge[i] - phi_y[i] + phi_s[i - 1] * self.period_hours / self.discharge_eff <= 0 for i in range(1, self.nb_periods)), name='c_discharge')  # time period 2 to nb_periods
        model.addConstr(phi_charge[0] + phi_y[0] - phi_ini * self.period_hours * self.charge_eff <= 0, name='c_charge_last')  # time period 1
        model.addConstrs((phi_charge[i] + phi_y[i] - phi_s[i - 1] * self.period_hours * self.charge_eff <= 0 for i in range(1, self.nb_periods)), name='c_charge')  # time period 2 to nb_periods
        model.addConstr(-phi_Smin[0] + phi_Smax[0] + phi_ini - phi_s[0] <= - self.high_soc_price, name='c_s_first')  # time period 1 for phi_Smin/phi_Smax and time period 2 for phi_s
        model.addConstrs((-phi_Smin[i] + phi_Smax[i] + phi_s[i - 2] - phi_s[i - 1] <= -self.high_soc_price for i in range(2, self.nb_periods - 1)), name='c_s')  # time period 3 to nb_periods - 1
        model.addConstr(-phi_Smin[self.nb_periods - 1] + phi_Smax[self.nb_periods - 1] + phi_end + phi_s[self.nb_periods - 2] <= -self.high_soc_price, name='c_s_last')  # Last time period

        model.addConstrs((- phi_y[i] + phi_PV[i] <= 0 for i in self.t_set), name='c_PV')


        # Bloc Constraints (20) from latex formulation -> constraints related to the uncertainty set
        model.addConstr((gp.quicksum(z_neg[i] for i in self.t_set) <= self.gamma ), name='c_gamma') # uncertainty budget

        # Bloc Constraints (24) from latex formulation -> constraints required to linearize bilinear terms
        # M_neg <= phi_PV <= M_pos = 0
        M_pos = 0
        model.addConstrs((alpha_neg[i]  <= M_pos * z_neg[i] for i in self.t_set), name='c_alpha_neg_1_max')
        model.addConstrs((alpha_neg[i]  >= -self.M_neg * z_neg[i] for i in self.t_set), name='c_alpha_neg_1_min')
        model.addConstrs((phi_PV[i] - alpha_neg[i]  <= M_pos * (1 - z_neg[i]) for i in self.t_set), name='c_alpha_neg_2_max')
        model.addConstrs((phi_PV[i] - alpha_neg[i]  >= -self.M_neg * (1 - z_neg[i]) for i in self.t_set), name='c_alpha_neg_2_min')

        # Get the index of the gamma max values of quantile_min
        ind_quantile_min_max_values = self.max_dev.reshape(-1).argsort()[-self.gamma:]

        # Set as heuristic solution the gamma max values of the min quantile
        # Populate 'start' attribute from an heuristic solution

        # if self.heuristic:
        #
        #     # compute the number of windows
        #     first_no_null = next(val for val in self.max_dev.tolist() if val)
        #     last_no_null = next(val for val in self.max_dev.tolist()[::-1] if val)
        #     i_first_no_null = np.where(self.max_dev == first_no_null)[0][0]
        #     i_last_no_null = np.where(self.max_dev == last_no_null)[0][0]
        #     nb_traj = i_last_no_null - (i_first_no_null + self.gamma - 1)
        #
        #     # total number of heuristic
        #     model.NumStart = 1 +  int(nb_traj) # Specifically, use the NumStart attribute to indicate how many start vectors you will supply.
        #
        #     # first heuristic
        #     model.setParam('StartNumber', 0)  # set the StartNumber parameter to a value between 0 and NumStart-1 to indicate which start you are supplying
        #     for ind in ind_quantile_min_max_values:
        #         z_neg[ind].start = 1
        #
        #     # compute the other heuristic with sliding window
        #     for i in range(1, nb_traj+1):
        #         model.setParam('StartNumber', i)  # set the StartNumber parameter to a value between 0 and NumStart-1 to indicate which start you are supplying
        #         for j in range(i_first_no_null + i, i_first_no_null + self.gamma + i):
        #             z_neg[j].start = 1
        #
        if self.heuristic:
            # total number of heuristic
            model.NumStart = 1  # Specifically, use the NumStart attribute to indicate how many start vectors you will supply.
            # first heuristic
            model.setParam('StartNumber', 0)  # set the StartNumber parameter to a value between 0 and NumStart-1 to indicate which start you are supplying
            for ind in ind_quantile_min_max_values:
                z_neg[ind].start = 1


        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['phi_charge'] = phi_charge
        self.allvar['phi_discharge'] = phi_discharge
        self.allvar['phi_Smin'] = phi_Smin
        self.allvar['phi_Smax'] = phi_Smax
        self.allvar['phi_y'] = phi_y
        self.allvar['phi_ymin'] = phi_ymin
        self.allvar['phi_ymax'] = phi_ymax
        self.allvar['phi_s'] = phi_s
        self.allvar['phi_devshort'] = phi_devshort
        self.allvar['phi_devlong'] = phi_devlong
        self.allvar['phi_PV'] = phi_PV
        self.allvar['alpha_neg'] = alpha_neg
        self.allvar['z_neg'] = z_neg
        self.allvar['phi_ini'] = phi_ini
        self.allvar['phi_end'] = phi_end
        if self.overproduction:
            self.allvar['phi_overprod'] = phi_overprod

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):
        """
        :param LogToConsole: no log in the console if set to False.
        :param logfile: no log in file if set to ""
        :param Threads: Default value = 0 -> use all threads
        :param MIPFocus: If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
                        If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
                        If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.
        :param TimeLimit: in seconds.
        """

        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        self.model.setParam('LogFile', logfile)
        self.model.setParam('Threads', Threads)

        self.model.optimize()

        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal


        # 0 dimensional variables
        for var in ['phi_ini', 'phi_end']:
            solution[var] = self.allvar[var].X

        # 1 dimensional variables
        for var in ['phi_charge', 'phi_discharge', 'phi_Smin', 'phi_Smax', 'phi_y', 'phi_ymin', 'phi_ymax', 'phi_devshort', 'phi_devlong', 'phi_PV', 'alpha_neg', 'z_neg']:
            solution[var] = [self.allvar[var][t].X for t in self.t_set]
        if self.overproduction:
            solution['phi_overprod'] = [self.allvar['phi_overprod'][t].X for t in self.t_set]

        solution['phi_s'] = [self.allvar['phi_s'][t].X for t in range(self.nb_periods-1)]

        solution['cut'] = solution['obj'] + sum(x * y for x, y in zip(solution['phi_devshort'], self.x)) - sum(x * y for x, y in zip(solution['phi_devlong'], self.x))

        # 3. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution

    def export_model(self, filename):
        """
        Export the model into a lp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)
        # self.model.write("%s.mps" % filename)


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = 'sandbox/ro/robust/export/'
    day = '2019-08-22' # sunny
    # day = '2019-08-18' # cloudy

    pv_solution = pd.read_csv('data1/uliege/uliege_15min.csv', index_col=0, parse_dates=True)['Pm'][day].values
    pv_forecast = pd.read_csv('data1/uliege/uliege_pv_prediction_python_15min.csv', parse_dates=True, index_col=0)[day].values

    # engagement = read_file(dir='sandbox/ro/determinist/export/', name='sol_LP_oracle'+day)
    engagement = read_file(dir='sandbox/ro/determinist/export/', name='sol_LP_PVUSA'+day)
    # engagement = [0]*96
    # engagement = read_file(dir='sandbox/ro/robust/export/24/', name='MP_benders_x_PVUSA_10')

    pv_dev_min = pv_forecast * 0.5
    pv_dev_max = pv_forecast * 1.1
    pv_dev_max[pv_dev_max > PARAMETERS['pv_capacity']] = PARAMETERS['pv_capacity']
    pv_dev_max = pv_dev_max - pv_forecast

    gamma = 24
    M_neg = 1
    # plt.figure()
    # plt.plot(pv_solution, label='Pm')
    # plt.plot(pv_forecast, label='Pp')
    # plt.plot(pv_forecast - pv_dev_min, label='quantile min')
    # plt.plot(pv_forecast + pv_dev_max, label='quantile max')
    # plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    # plt.legend()
    # plt.show()

    heuristic = True
    SP_dual = BD_SP(pv_forecast=pv_forecast, max_dev=pv_dev_min, engagement=engagement, gamma=gamma, heuristic=heuristic, M_neg=M_neg)
    SP_dual.export_model(dirname + 'SP_dual_MILP')
    MIPFocus = 0
    TimeLimit = 5
    logname = 'SP_dual_MILP_start_'+str(heuristic)+'_MIPFocus_'+str(MIPFocus)+'.log'
    SP_dual.solve(LogToConsole=True, logfile=dirname + logname, Threads=1, MIPFocus=MIPFocus, TimeLimit=TimeLimit)
    solution = SP_dual.store_solution()

    print('objective %.2f' %(solution['obj']))

    plt.figure()
    plt.plot(solution['phi_PV'], label='phi_PV')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['z_neg'], label='z_neg')
    plt.plot(solution['z_pos'], label='z_pos')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['alpha_neg'], label='alpha_neg')
    plt.plot(solution['alpha_pos'], label='alpha_pos')
    plt.legend()
    plt.show()

    worst_case = [pv_forecast[i] - solution['z_neg'][i] * pv_dev_min[i] + solution['z_pos'][i] * pv_dev_max[i] for i in range(96)]

    plt.figure()
    plt.plot(worst_case, 'k', label='worst_case')
    plt.plot(pv_solution, label='Pm')
    plt.plot(pv_forecast, label='Pp')
    plt.plot(pv_forecast - pv_dev_min, ':', label='quantile min')
    plt.plot(pv_forecast + pv_dev_max, ':', label='quantile max')
    plt.ylim(-0.05 * PARAMETERS['pv_capacity'], PARAMETERS['pv_capacity'])
    plt.legend()
    plt.show()

    # Get dispatch variables by solving the primal LP with the worst case PV generation trajectory
    SP_primal = SP_primal_LP(pv_forecast=worst_case, engagement=engagement)
    SP_primal.solve()
    SP_primal_sol = SP_primal.store_solution()

    print('objective %.2f' % (SP_primal_sol['obj']))

    plt.figure()
    plt.plot(SP_primal_sol['y_s'], label='s')
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
    plt.legend()
    plt.show()