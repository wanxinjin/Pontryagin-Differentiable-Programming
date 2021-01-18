from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time

# --------------------------- load environment ----------------------------------------
uav = JinEnv.Quadrotor()
uav.initDyn(c=0.01)
uav.initCost(wthrust=0.1)

# --------------------------- load demos data ----------------------------------------
data = sio.loadmat('../quadrotor/data/uav_demos.mat')
trajectories = data['trajectories']
true_parameter = data['true_parameter']
dt = data['dt']

# reorganize data into numpy-preferred
n_demo = trajectories.shape[1]
n_state = trajectories[0, 0]['state_traj_opt'][0, 0].shape[1]
n_control = trajectories[0, 0]['control_traj_opt'][0, 0].shape[1]

state_trajs = []
control_trajs = []
costate_trajs = []
for i in range(n_demo):
    state_traj = trajectories[0, i]['state_traj_opt'][0, 0]
    state_trajs += [state_traj]
    control_traj = trajectories[0, i]['control_traj_opt'][0, 0]
    control_trajs += [control_traj]
    costate_traj = trajectories[0, i]['costate_traj_opt'][0, 0]
    costate_trajs += [costate_traj]

# ---------------------------------------- define the KKT loss function -----------------------
# first establish a pdp object, because there are some utilities in pdp which can be used to define KKT loss
uavoc = PDP.OCSys()
uavoc.setAuxvarVariable(vertcat(uav.dyn_auxvar, uav.cost_auxvar))
uavoc.setControlVariable(uav.U)
uavoc.setStateVariable(uav.X)
dyn = uav.X + dt * uav.f
uavoc.setDyn(dyn)
uavoc.setPathCost(uav.path_cost)
uavoc.setFinalCost(uav.final_cost)
uavoc.diffPMP()
lqr_solver = PDP.LQR()

# define loss function, which is the norm of KKT residual, please find the related work for more details
KKT_loss = 0
parameter_theta = uavoc.auxvar
parameter_theta_guess = true_parameter.flatten().tolist()
parameter_lambda = []
parameter_lambda_guess = []  # in fact, this will store the true value of lambda (costates)
for i in range(n_demo):
    state_traj = state_trajs[i]
    control_traj = control_trajs[i]
    costates_traj = costate_trajs[i]
    horizon = control_traj.shape[0]
    # define costate variable
    LAM = SX.sym('lambda_1' + str(i), n_state)
    parameter_lambda += [LAM]
    parameter_lambda_guess += costates_traj[0, :].flatten().tolist()
    for t in range(0, horizon - 1):
        x_curr = state_traj[t + 1, :]
        u_curr = control_traj[t + 1, :]
        # define unknown costate
        LAM_next = SX.sym('lambda_' + str(t + 2) + str(i), n_state)
        parameter_lambda += [LAM_next]
        parameter_lambda_guess += costates_traj[t + 1, :].flatten().tolist()
        # compute the current loss
        loss_curr_1 = dot(LAM - uavoc.dHx_fn(x_curr, u_curr, LAM_next, parameter_theta),
                          LAM - uavoc.dHx_fn(x_curr, u_curr, LAM_next, parameter_theta))
        loss_curr_2 = dot(uavoc.dHu_fn(x_curr, u_curr, LAM_next, parameter_theta),
                          uavoc.dHu_fn(x_curr, u_curr, LAM_next, parameter_theta))
        KKT_loss = KKT_loss + loss_curr_1 + loss_curr_2
        # update
        LAM = LAM_next
    # final cost term
    KKT_loss = KKT_loss + dot(LAM - uavoc.dhx_fn(state_traj[-1, :], parameter_theta),
                              LAM - uavoc.dhx_fn(state_traj[-1, :], parameter_theta))

#  define KKT loss function
parameter_lambda = vcat(parameter_lambda)
parameter = vertcat(parameter_theta, parameter_lambda)
parameter_guess = parameter_theta_guess + parameter_lambda_guess
KKT_loss_fn = Function('loss_fn', [parameter], [KKT_loss])
grad_KKT_loss_fn = Function('grad_loss_fn', [parameter], [jacobian(KKT_loss, parameter)])

# ------------------------ inverse KKT: learn both the theta and costate (lambda) --------------------------

lr = 1e-7  # learning rate
for j in range(10):  # loop for trials
    start_time = time.time()
    traj_loss_trace, theta_trace, kkt_loss_trace = [], [], []

    sigma = 0.08
    initial_parameter = np.array(parameter_guess) # note we use the true parameter to initilize because otherwise the method is not working well.
    initial_parameter = initial_parameter + sigma * np.random.randn(np.size(initial_parameter)) - sigma / 2
    current_parameter = initial_parameter

    for k in range(int(2e4)):  # loop for iteration
        # update for parameter
        dp = grad_KKT_loss_fn(current_parameter).full().flatten()
        current_parameter = current_parameter - lr * dp
        current_theta = current_parameter[0:parameter_theta.numel()]
        theta_trace += [current_theta]  # only take theta
        kkt_loss = KKT_loss_fn(current_parameter).full().flatten().item()
        kkt_loss_trace += [kkt_loss]

        # solve the trajectory loss
        current_loss = 0
        for i in range(n_demo):
            # take out the demos i
            demo_state_traj = state_trajs[i]
            demo_control_traj = control_trajs[i]
            initial_state = demo_state_traj[0, :]
            horizon = demo_control_traj.shape[0]
            #  current trajectory based on current theta guess
            current_traj = uavoc.ocSolver(initial_state, horizon, current_theta)
            current_state_traj = current_traj['state_traj_opt']
            current_control_traj = current_traj['control_traj_opt']
            dldx_traj = current_state_traj - demo_state_traj
            dldu_traj = current_control_traj - demo_control_traj
            current_loss = current_loss + numpy.linalg.norm(dldx_traj) ** 2 + numpy.linalg.norm(dldu_traj) ** 2
        traj_loss_trace += [current_loss / n_demo]

        # print and terminal check
        if k % 1 == 0:
            print('trial:', j, 'iter:',k, ' loss: ', traj_loss_trace[-1])

    # save
    save_data = {'trail_no': j,
                 'loss_trace': traj_loss_trace,
                 'kkt_loss_trace': kkt_loss_trace,
                 'learning_rate': lr,
                 'parameter_trace': theta_trace,
                 'time_passed': time.time() - start_time}
    sio.savemat('./data/KKT_results_trial_' + str(j) + '.mat', {'results': save_data})
