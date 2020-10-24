import sys

sys.path.insert(1, '../../')
import PDP, JinEnv
from casadi import *
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from torch import nn


# --------------------------- load environment ----------------------------------------
env = JinEnv.Quadrotor()
env.initDyn(c=0.01)
env.initCost(wthrust=0.1)
true_parameter = [1, 1, 1, 1, 0.4, 1, 1, 5, 1]

# --------------------------- pdp object --------------------------------------
dt = 0.1
oc = PDP.OCSys()
oc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.cost_auxvar))
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
dyn = env.X + dt * env.f
oc.setDyn(dyn)
oc.setPathCost(env.path_cost)
oc.setFinalCost(env.final_cost)
lqr_solver = PDP.LQR()
print(oc.auxvar)
print(true_parameter)


# set initial state
ini_r_I=[-2, -8, 12.]
ini_v_I = [-9.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0.5,[1,-10,10])
ini_w = [-3.0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon = 40
true_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)


# # --------------------------- learn both the dynamics and objective function ----------------------------------------
# start_time = time.time()
# lr = 1e-4  # learning rate
# # initialize
# loss_trace, parameter_trace = [], []
# initial_parameter = numpy.array([1, 1, .1, 1, 0.2, .2, .1, .125, .1])
# current_parameter = initial_parameter
# parameter_trace+=[current_parameter]
# for k in range(int(10e4)):  # iteration loop (or epoch loop)
#     # demos information extraction
#     demo_state_traj = true_sol['state_traj_opt']
#     demo_control_traj = true_sol['control_traj_opt']
#     demo_ini_state = demo_state_traj[0, :]
#     demo_horizon = demo_control_traj.shape[0]
#     # learner's current trajectory based on current parameter guess
#     traj = oc.ocSolver(demo_ini_state, demo_horizon, current_parameter)
#     # Establish the auxiliary control system
#     aux_sys = oc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
#                               control_traj_opt=traj['control_traj_opt'],
#                               costate_traj_opt=traj['costate_traj_opt'],
#                               auxvar_value=current_parameter)
#     lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
#     lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
#                            Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
#     lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
#     aux_sol = lqr_solver.lqrSolver(numpy.zeros((oc.n_state, oc.n_auxvar)), demo_horizon)
#     # take solution of the auxiliary control system
#     dxdp_traj = aux_sol['state_traj_opt']
#     dudp_traj = aux_sol['control_traj_opt']
#     # evaluate the loss
#     state_traj = traj['state_traj_opt']
#     control_traj = traj['control_traj_opt']
#     dldx_traj = state_traj - demo_state_traj
#     dldu_traj = control_traj - demo_control_traj
#     loss = numpy.linalg.norm(dldx_traj) ** 2 + numpy.linalg.norm(dldu_traj) ** 2
#     # chain rule
#     dp = np.zeros(current_parameter.shape)
#     for t in range(demo_horizon):
#         dp = dp + np.matmul(dldx_traj[t, :], dxdp_traj[t]) + np.matmul(dldu_traj[t, :], dudp_traj[t])
#     dp = dp + numpy.dot(dldx_traj[-1, :], dxdp_traj[-1])
#
#     # update
#     current_parameter = current_parameter - lr * dp
#     parameter_trace += [current_parameter.flatten()]
#     loss_trace += [loss]
#
#     # print and terminal check
#     if k % 1 == 0:
#         print('iter:', k, ' loss: ', loss_trace[-1].tolist())
#
#
# save_data = {'loss_trace': loss_trace,
#              'parameter_trace': parameter_trace,
#              'learning_rate': lr,
#              'time_passed': time.time() - start_time}
# sio.savemat('uav_demo_data.mat', {'results': save_data})


# # --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('uav_demo_data.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
pdp_parameter_trace = np.squeeze(parameter_trace)
#
# pdp_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=pdp_parameter_trace[-1,:])
# env.play_animation(wing_len=1.5, state_traj=pdp_sol['state_traj_opt'],
#                    dt=dt, save_option=0, title='Imitation learning (# training iteration 0)')


# # --------------------------- planning using the learned model ----------------------------------------
# set initial state
ini_r_I=[5, 8, 6.]
ini_v_I = [-9.0, 0.0, 5.0]
ini_q = JinEnv.toQuaternion(1.5,[1,-.0,-0])
ini_w = [-3.0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon = 50
pdp_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=pdp_parameter_trace[-1,:])
env.play_animation(wing_len=1.5, state_traj=pdp_sol['state_traj_opt'],
                   dt=dt, save_option=1, title='Planning using learned dynamics and objective')