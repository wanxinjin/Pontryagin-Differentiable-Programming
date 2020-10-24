from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import numpy as np
import time

# --------------------------- load environment ----------------------------------------
cartpole = JinEnv.CartPole()
cartpole.initDyn()
cartpole.initCost(wu=0.1)

# --------------------------- load demos data ----------------------------------------
data = sio.loadmat('data/cartpole_demos.mat')
trajectories = data['trajectories']
true_parameter = data['true_parameter'].flatten()
dt = data['dt']

# --------------------------- define PDP ----------------------------------------
cartpoleoc = PDP.OCSys()
cartpoleoc.setAuxvarVariable(vertcat(cartpole.dyn_auxvar, cartpole.cost_auxvar))
cartpoleoc.setControlVariable(cartpole.U)
cartpoleoc.setStateVariable(cartpole.X)
dyn = cartpole.X + dt * cartpole.f
cartpoleoc.setDyn(dyn)
cartpoleoc.setPathCost(cartpole.path_cost)
cartpoleoc.setFinalCost(cartpole.final_cost)
cartpoleoc.diffPMP()
lqr_solver = PDP.LQR()

# --------------------------- learn both the dynamics and objective function ----------------------------------------
for j in range(10):
    start_time = time.time()
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    sigma = 0.9
    initial_parameter = true_parameter + sigma * np.random.random(len(true_parameter)) - sigma / 2
    current_parameter = initial_parameter
    for k in range(int(1e4)):  # iteration loop (or epoch loop)
        loss = 0
        dp = np.zeros(current_parameter.shape)
        # loop for each demos trajectory
        n_demo = trajectories.shape[1]
        for i in range(n_demo):
            demo_state_traj = trajectories[0, i]['state_traj_opt'][0, 0]
            demo_control_traj = trajectories[0, i]['control_traj_opt'][0, 0]
            demo_ini_state = demo_state_traj[0, :]
            demo_horizon = demo_control_traj.shape[0]
            # learner's current trajectory based on current parameter guess
            traj = cartpoleoc.ocSolver(ini_state=demo_ini_state, horizon=demo_horizon, auxvar_value=current_parameter)
            # Establish the auxiliary control system
            aux_sys = cartpoleoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                           control_traj_opt=traj['control_traj_opt'],
                                           costate_traj_opt=traj['costate_traj_opt'],
                                           auxvar_value=current_parameter)
            lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
            lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                   Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
            lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
            aux_sol = lqr_solver.lqrSolver(numpy.zeros((cartpoleoc.n_state, cartpoleoc.n_auxvar)), demo_horizon)
            # take solution of the auxiliary control system
            dxdp_traj = aux_sol['state_traj_opt']
            dudp_traj = aux_sol['control_traj_opt']
            # evaluate the loss
            state_traj = traj['state_traj_opt']
            control_traj = traj['control_traj_opt']
            dldx_traj = state_traj - demo_state_traj
            dldu_traj = control_traj - demo_control_traj
            loss = loss + numpy.linalg.norm(dldx_traj) ** 2 + numpy.linalg.norm(dldu_traj) ** 2
            # chain rule
            for t in range(demo_horizon):
                dp = dp + np.matmul(dldx_traj[t, :], dxdp_traj[t]) + np.matmul(dldu_traj[t, :], dudp_traj[t])
            dp = dp + numpy.dot(dldx_traj[-1, :], dxdp_traj[-1])

        # take the expectation (average)
        dp = dp / n_demo
        loss = loss / n_demo
        # update
        current_parameter = current_parameter - lr * dp
        parameter_trace += [current_parameter]
        loss_trace += [loss]

        # print and terminal check
        if k % 1 == 0:
            print('trial:', j, 'iter:',k, 'loss:', loss_trace[-1].tolist())

    # save
    save_data = {'trail_no': j,
                 'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'learning_rate': lr,
                 'time_passed': time.time() - start_time}
    sio.savemat('./data/PDP_results_trial_' + str(j) + '.mat', {'results': save_data})
