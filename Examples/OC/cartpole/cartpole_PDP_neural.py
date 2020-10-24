from PDP import PDP
from JinEnv import JinEnv
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
oc = PDP.ControlPlanning()
oc.setStateVariable(cartpole.X)
oc.setControlVariable(cartpole.U)
dyn = cartpole.X + dt * cartpole.f
oc.setDyn(dyn)
oc.setPathCost(cartpole.path_cost)
oc.setFinalCost(cartpole.final_cost)

# --------------------------- create PDP true OC object ----------------------------------------
# compute the ground truth
true_cartpoleoc = PDP.OCSys()
true_cartpoleoc.setStateVariable(cartpole.X)
true_cartpoleoc.setControlVariable(cartpole.U)
true_cartpoleoc.setDyn(dyn)
true_cartpoleoc.setPathCost(cartpole.path_cost)
true_cartpoleoc.setFinalCost(cartpole.final_cost)
true_sol = true_cartpoleoc.ocSolver(ini_state=ini_state, horizon=horizon)
true_state_traj = true_sol['state_traj_opt']
true_control_traj = true_sol['control_traj_opt']
print(true_sol['cost'])


# --------------------------- do the system control and planning ----------------------------------------
for j in range(5): # trial loop
    # learning rate
    lr = 1e-4
    loss_trace, parameter_trace = [], []
    oc.init_step_neural_policy(hidden_layers=[oc.n_state,oc.n_state])
    initial_parameter = np.random.randn(oc.n_auxvar)
    current_parameter = initial_parameter
    max_iter = 5000
    start_time = time.time()
    for k in range(int(max_iter)):
        # one iteration of PDP
        loss, dp = oc.step(ini_state, horizon, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter]
        # print
        if k % 100 == 0:
            print('trial:', j ,'Iter:', k, 'loss:', loss)

    # solve the trajectory
    sol = oc.integrateSys(ini_state, horizon, current_parameter)

    # save the results
    save_data = {'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'learning_rate': lr,
                 'solved_solution': sol,
                 'true_solution': true_sol,
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

    sio.savemat('./data/PDP_Neural_trial_' + str(j) + '.mat', {'results': save_data})