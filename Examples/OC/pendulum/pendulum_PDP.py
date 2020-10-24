from PDP import PDP
from JinEnv import JinEnv
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt

# --------------------------- load environment ----------------------------------------
pendulum = JinEnv.SinglePendulum()
pendulum.initDyn(l=1, m=1, damping_ratio=0.05)
pendulum.initCost(wq=10, wdq=1, wu=0.1)

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.05
pendulumoc = PDP.ControlPlanning()
pendulumoc.setStateVariable(pendulum.X)
pendulumoc.setControlVariable(pendulum.U)
dyn = pendulum.X + dt * pendulum.f
pendulumoc.setDyn(dyn)
pendulumoc.setPathCost(pendulum.path_cost)
pendulumoc.setFinalCost(pendulum.final_cost)
# set the horizon and initial condition
horizon = 10
ini_state = [0, 0]

# --------------------------- create PDP true oc solver ----------------------------------------
ture_pendulumoc = PDP.OCSys()
ture_pendulumoc.setStateVariable(pendulum.X)
ture_pendulumoc.setControlVariable(pendulum.U)
ture_pendulumoc.setDyn(dyn)
ture_pendulumoc.setPathCost(pendulum.path_cost)
ture_pendulumoc.setFinalCost(pendulum.final_cost)
true_sol = ture_pendulumoc.ocSolver(ini_state=ini_state, horizon=horizon)
print(true_sol['cost'])

# true_state_traj = true_sol['state_traj_opt']
# true_control_traj = true_sol['control_traj_opt']
# plt.figure(1)
# plt.plot(true_control_traj)
# plt.show()
# pendulum.play_animation(len=1,dt=dt,state_traj=true_state_traj)

# ---------------------------- start learning the control policy -------------------------------------

for j in range(10):
    start_time = time.time()
    # learning rate
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    pendulumoc.init_step(horizon)
    current_parameter = np.random.randn(pendulumoc.n_auxvar)
    max_iter = 1e4
    # iteration
    for k in range(int(max_iter)):
        # one iteration of PDP
        loss, dp = pendulumoc.step(ini_state, horizon, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss)

    # solve the trajectory
    sol = pendulumoc.integrateSys(ini_state, horizon, current_parameter)

    # print(sol['cost'])
    # plt.figure(0)
    # plt.plot(true_sol['control_traj_opt'])
    # plt.plot(sol['control_traj'])
    # plt.legend(['true_control', 'pdp_control'])
    # plt.show()
    # pendulum.play_animation(len=2, dt=dt, state_traj=sol['state_traj'], state_traj_ref=true_sol['state_traj_opt'])


    # save the results
    save_data = {'trail_no': j,
                 'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'learning_rate': lr,
                 'solved_solution': sol,
                 'true_solution': true_sol,
                 'time_passed': time.time() - start_time,
                 'pendulum': {'l': 1,
                              'm': 1,
                              'damping_ratio': 0.05,
                              'wq': 10,
                              'wdq': 1,
                              'wu': 0.1},
                 'dt': dt,
                 'horizon': horizon
                 }
    sio.savemat('./data/PDP_OC_results_trial_' + str(j) + '.mat', {'results': save_data})
