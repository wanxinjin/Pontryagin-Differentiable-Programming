from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import numpy as np
import time
import scipy.io as sio
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


# --------------------------- create PDP Control/Planning object ----------------------------------------
armoc = PDP.ControlPlanning()
armoc.setStateVariable(arm.X)
armoc.setControlVariable(arm.U)
dyn = arm.X + dt * arm.f
armoc.setDyn(dyn)
armoc.setPathCost(arm.path_cost)
armoc.setFinalCost(arm.final_cost)


# --------------------------- create PDP true OC object ----------------------------------------
true_cartpoleoc = PDP.OCSys()
true_cartpoleoc.setStateVariable(arm.X)
true_cartpoleoc.setControlVariable(arm.U)
true_cartpoleoc.setDyn(dyn)
true_cartpoleoc.setPathCost(arm.path_cost)
true_cartpoleoc.setFinalCost(arm.final_cost)
true_sol = true_cartpoleoc.ocSolver(ini_state=ini_state, horizon=horizon)
true_state_traj = true_sol['state_traj_opt']
true_control_traj = true_sol['control_traj_opt']
print(true_sol['cost'])



# --------------------------- do the system control and planning ----------------------------------------
for j in range(10):
    # learning rate
    lr = 1e-2
    loss_trace, parameter_trace = [], []
    armoc.init_step(horizon)
    initial_parameter = np.random.randn(armoc.n_auxvar)
    current_parameter = initial_parameter
    max_iter = 5000
    start_time = time.time()
    for k in range(int(max_iter)):
        # one iteration of PDP
        loss, dp = armoc.step(ini_state, horizon, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter]
        # print
        if k % 100 == 0:
            print('trial:', j ,'Iter:', k, 'loss:', loss)

    # solve the trajectory
    sol = armoc.integrateSys(ini_state, horizon, current_parameter)

    # save the results
    save_data = {'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'learning_rate': lr,
                 'solved_solution': sol,
                 'true_solution': true_sol,
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

    sio.savemat('./data/PDP_OC_results_trial_' + str(j) + '.mat', {'results': save_data})
