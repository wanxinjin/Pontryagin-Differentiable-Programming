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
cartpole = JinEnv.CartPole()
mc, mp, l = 0.1, 0.1, 1
cartpole.initDyn(mc=mc, mp=mp, l=l)
wx, wq, wdx, wdq, wu = 0.1, 0.6, 0.1, 0.1, 0.3
cartpole.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu=wu)

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.05
horizon = 25
ini_state = [0, 0, 0, 0]

# --------------------------- create PDP true OC object ----------------------------------------
# compute the ground truth
true_cartpoleoc = PDP.OCSys()
true_cartpoleoc.setStateVariable(cartpole.X)
true_cartpoleoc.setControlVariable(cartpole.U)
dyn = cartpole.X + dt * cartpole.f
true_cartpoleoc.setDyn(dyn)
true_cartpoleoc.setPathCost(cartpole.path_cost)
true_cartpoleoc.setFinalCost(cartpole.final_cost)
true_sol = true_cartpoleoc.ocSolver(ini_state=ini_state, horizon=horizon)
true_state_traj = true_sol['state_traj_opt']
true_control_traj = true_sol['control_traj_opt']
print(true_sol['cost'])

# --------------------------- set GPS object------------------------------
gps=ControlTools.GuidePS()
gps.setStateVariable(cartpole.X)
gps.setControlVariable(cartpole.U)
dyn = cartpole.X + dt * cartpole.f
gps.setDyn(dyn)
gps.setPathCost(cartpole.path_cost)
gps.setFinalCost(cartpole.final_cost)
gps.setNeuralPolicy([cartpole.X.numel(),cartpole.X.numel()])

# --------------------------- do the iteration ----------------------------------------
for j in range(5):  # trial loop
    start_time = time.time()
    # learning rate
    lr = 1e-5
    # intialize lambda parameter
    lambda_auxvar_value=1*np.ones(horizon*gps.n_control)
    # intialize policy parameter
    policy_auxvar_value=np.random.random(gps.n_policy_auxvar)
    # maximum iteration
    max_iter=500
    # set penality
    rho=200
    # trace storage vector
    loss_trace, parameter_trace = [], []
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
        cost, traj=gps.getPolicyCost(ini_state=ini_state, horizon=horizon,
                                     policy_auxvar_value=policy_auxvar_value)
        loss_trace += [cost]
        parameter_trace += [policy_auxvar_value]
        # print
        if k % 1 == 0:
            print('trial:', j, 'Iter:', k, 'cost:', cost)

    # solve the trajectory
    cost, sol = gps.getPolicyCost(ini_state, horizon, policy_auxvar_value)

    # save the results
    save_data = {'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'learning_rate': lr,
                 'solved_solution': sol,
                 'time_passed': time.time() - start_time,
                 'cartpole': {'mc': mc,
                              'mp': mp,
                              'l': l,
                              'wx': wx,
                              'wq': wq,
                              'wdx': wdx,
                              'wdq': wdq,
                              'wu': wu},
                 'dt': dt,
                 'horizon': horizon}

    sio.savemat('./data/GPS_Neural_trial_' + str(j) + '.mat', {'results': save_data})