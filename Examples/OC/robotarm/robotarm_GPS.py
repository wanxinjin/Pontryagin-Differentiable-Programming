import sys
from PDP import PDP
from JinEnv import JinEnv
from ControlTool import ControlTools
from casadi import *
import numpy as np
import time
import scipy.io as sio
import math
import matplotlib.pyplot as plt

# --------------------------- load environment ----------------------------------------
arm = JinEnv.RobotArm()
l1, m1, l2, m2, g = 1, 1, 1, 1, 0
arm.initDyn(l1=l1, m1=m1, l2=l2, m2=m2, g=0)
wq1, wq2, wdq1, wdq2, wu = 0.1, 0.1, 0.1, 0.1, 0.01
arm.initCost(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2, wu=wu)

# initial state and horizon
horizon = 20
ini_state = [pi/4, pi/2, 0, 0]
dt = 0.1

# --------------------------- create PDP true OC object ----------------------------------------
true_cartpoleoc = PDP.OCSys()
true_cartpoleoc.setStateVariable(arm.X)
true_cartpoleoc.setControlVariable(arm.U)
dyn = arm.X + dt * arm.f
true_cartpoleoc.setDyn(dyn)
true_cartpoleoc.setPathCost(arm.path_cost)
true_cartpoleoc.setFinalCost(arm.final_cost)
true_sol = true_cartpoleoc.ocSolver(ini_state=ini_state, horizon=horizon)
true_state_traj = true_sol['state_traj_opt']
true_control_traj = true_sol['control_traj_opt']
print(true_sol['cost'])

# --------------------------- set GPS object------------------------------
gps=ControlTools.GuidePS()
gps.setStateVariable(arm.X)
gps.setControlVariable(arm.U)
dyn = arm.X + dt * arm.f
gps.setDyn(dyn)
gps.setPathCost(arm.path_cost)
gps.setFinalCost(arm.final_cost)
gps.setNeuralPolicy([arm.X.numel()])

# --------------------------- do the iteration ----------------------------------------
for j in range(6,7):  # trial loop
    start_time = time.time()
    # learning rate
    lr = 1e-7
    # intialize lambda parameter
    lambda_auxvar_value=1*np.ones(horizon*gps.n_control)
    # lambda_auxvar_value=1
    # intialize policy parameter
    policy_auxvar_value=np.random.random(gps.n_policy_auxvar)
    # maximum iteration
    max_iter=5000
    # set penality
    rho=1000
    # trace storage vector
    loss_trace, parameter_trace = [], []
    # start iteration
    for k in range(int(max_iter)):
        # solve optimal control problem
        sol=gps.getTrajectoryOpt(ini_state=ini_state, horizon=horizon,
                                 lambda_auxvar_value=lambda_auxvar_value,
                                 policy_auxvar_value=policy_auxvar_value,
                                 rho=rho)
        state_traj = sol['state_traj_opt']
        control_traj = sol['control_traj_opt']

        # supervised to learn the policy
        policy_auxvar_value=gps.getSupervisedPolicy(state_traj=state_traj,
                                                    control_traj=control_traj,
                                                    lambda_auxvar_value=lambda_auxvar_value,
                                                    policy_auxvar_value=policy_auxvar_value,
                                                    rho=rho)
        # update lambda_auxvar_value
        grad_lambda_auxvar=gps.getGradLambda(state_traj=state_traj,
                                             control_traj=control_traj,
                                             policy_auxvar_value=policy_auxvar_value)
        lambda_auxvar_value=lambda_auxvar_value+lr*grad_lambda_auxvar

        # use the current policy to compute the cost
        cost, sol=gps.getPolicyCost(ini_state=ini_state, horizon=horizon,
                               policy_auxvar_value=policy_auxvar_value)
        loss_trace += [cost]
        parameter_trace += [policy_auxvar_value]
        # print
        if k % 1 == 0:
            print('trial:', j, 'Iter:', k, 'cost:', cost)

    # save the results
    cost, sol = gps.getPolicyCost(ini_state, horizon, policy_auxvar_value)

    save_data = {'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'learning_rate': lr,
                 'solved_solution':sol,
                 'time_passed': time.time() - start_time,
                 'robotarm': {'l1': l1,
                              'm1': m1,
                              'l2': l2,
                              'm2': m2,
                              'wq1': wq1,
                              'wq2': wq2,
                              'wdq1': wdq1,
                              'wdq2': wdq2,
                              'wu': wu},
                 'dt': dt,
                 'horizon': horizon}
    sio.savemat('./data/GPS_Neural_trial_' + str(j) + '.mat', {'results': save_data})
