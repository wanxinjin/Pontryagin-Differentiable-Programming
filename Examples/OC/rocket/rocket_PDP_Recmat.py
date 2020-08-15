from PDP import PDP
from JinEnv import JinEnv
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import time

# --------------------------- load environment ----------------------------------------
rocket = JinEnv.Rocket()
rocket.initDyn(Jx=0.5, Jy=1., Jz=1., mass=1., l=1.)
rocket.initCost(wr=1, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust=0.4)

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.1
horizon = 50
rocketoc = PDP.ControlPlanning()
rocketoc.setStateVariable(rocket.X)
rocketoc.setControlVariable(rocket.U)
dyn = rocket.X + dt * rocket.f
rocketoc.setDyn(dyn)
rocketoc.setPathCost(rocket.path_cost)
rocketoc.setFinalCost(rocket.final_cost)
ini_r_I = [10, -8, 5.]
ini_v_I = [-.1, 0.0, -0.0]
ini_q = JinEnv.toQuaternion(1.5, [0, 0, 1])
ini_w = [0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w

# --------------------------- create PDP true oc solver ----------------------------------------
true_rocketoc = PDP.OCSys()
true_rocketoc.setStateVariable(rocket.X)
true_rocketoc.setControlVariable(rocket.U)
true_rocketoc.setDyn(dyn)
true_rocketoc.setPathCost(rocket.path_cost)
true_rocketoc.setFinalCost(rocket.final_cost)
true_sol = true_rocketoc.ocSolver(ini_state=ini_state, horizon=horizon)
print(true_sol['cost'])
# rocket.play_animation(rocket_len=2,state_traj=true_sol['state_traj_opt'],control_traj=true_sol['control_traj_opt'])
# ---------------------------- start learning the control policy -------------------------------------
for j in range(5):
    start_time = time.time()
    # learning rate
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    rocketoc.recmat_init_step(horizon, -1)
    # current_parameter = np.random.randn(rocketoc.n_auxvar)
    current_parameter = true_sol['control_traj_opt'].flatten()+5*np.random.randn(true_sol['control_traj_opt'].flatten().size)
    parameter_trace+=[current_parameter.flatten()]
    # iteration
    for k in range(int(5e4)):
        # one iteration of PDP
        loss, dp = rocketoc.recmat_step(ini_state, horizon, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter.flatten()]
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss)

    # solve the trajectory
    sol = rocketoc.recmat_unwarp(ini_state, horizon, current_parameter)
    # rocket.play_animation(rocket_len=2,state_traj=sol['state_traj'],control_traj=sol['control_traj'])

    # save the results
    save_data = {'trail_no': j,
                 'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'learning_rate': lr,
                 'solved_solution': sol,
                 'true_solution': true_sol,
                 'time_passed': time.time() - start_time,
                 'dt': dt,
                 'horizon': horizon
                 }
    sio.savemat('./data/PDP_OC_results_trial_' + str(j) + '.mat', {'results': save_data})
