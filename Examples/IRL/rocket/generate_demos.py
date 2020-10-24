from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time

# --------------------------- load environment ----------------------------------------
env = JinEnv.Rocket()
env.initDyn()
env.initCost(wthrust=0.1)


# --------------------------- create PDP object ----------------------------------------
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

# --------------------------- demonstration generation ----------------------------------------
true_parameter = [0.5, 1, 1, 1, 1,
                  1, 1, 50, 1, 1]
demos = []
# set initial state
ini_r_I=[10, -8, 5.]
ini_v_I = [-.5, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(1.,[0,1,-1])
ini_w = [0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon = 40

for i in range(1):
    sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)
    env.play_animation(rocket_len=2,state_traj=sol['state_traj_opt'],control_traj=sol['control_traj_opt'])
    demos += [sol]
# save
sio.savemat('data/rocket_demos.mat', {'trajectories': demos,
                                     'dt': dt,
                                     'true_parameter': true_parameter})
