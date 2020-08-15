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

# --------------------------- create PDP object ----------------------------------------
# create a pdp object
dt = 0.1
uavoc = PDP.OCSys()
uavoc.setAuxvarVariable(vertcat(uav.dyn_auxvar, uav.cost_auxvar))
uavoc.setStateVariable(uav.X)
uavoc.setControlVariable(uav.U)
dyn = uav.X + dt * uav.f
uavoc.setDyn(dyn)
uavoc.setPathCost(uav.path_cost)
uavoc.setFinalCost(uav.final_cost)

# --------------------------- demonstration generation ----------------------------------------
true_parameter = [1, 1, 1, 1, 0.4,  1, 1, 5, 1]
horizon = 50
demos = []
# set initial state
alternative_r_I=[[-8, -6, 9.],[8, 6, 9.]]
ini_v_I = [0.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0,[1,-1,1])
ini_w = [0.0, 0.0, 0.0]

for i in range(2):
    ini_state = alternative_r_I[i] + ini_v_I + ini_q + ini_w
    sol = uavoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)
    uav.play_animation(wing_len=1.5, state_traj=sol['state_traj_opt'])
    demos += [sol]
# save
sio.savemat('./data/uav_demos.mat', {'trajectories': demos,
                                     'dt': dt,
                                     'true_parameter': true_parameter})
