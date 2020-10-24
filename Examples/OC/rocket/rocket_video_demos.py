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
optimal_cost = true_sol['cost']
# rocket.play_animation(rocket_len=2,state_traj=true_sol['state_traj_opt'],control_traj=true_sol['control_traj_opt'])

# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('data/PDP_OC_results_trial_6.mat')
parameter_trace = load_data['results']['parameter_trace'][0, 0]
rocketoc.recmat_init_step(horizon, -1)
pdp_sol = rocketoc.recmat_unwarp(ini_state, horizon, parameter_trace[-1, :])

rocket.play_animation(rocket_len=2, state_traj=pdp_sol['state_traj'], control_traj=pdp_sol['control_traj'],
                      state_traj_ref=true_sol['state_traj_opt'], control_traj_ref=true_sol['control_traj_opt'],
                      dt=0.1,
                      save_option=0, title='Comparison with ground-truth from OC solver')
