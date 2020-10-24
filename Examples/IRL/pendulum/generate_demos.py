from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt

# --------------------------- load environment ----------------------------------------
pendulum = JinEnv.SinglePendulum()
pendulum.initDyn()
pendulum.initCost()

# --------------------------- create PDP object ----------------------------------------
# create a pdp object
dt = 0.1
pendulumoc = PDP.OCSys()
pendulumoc.setAuxvarVariable(vertcat(pendulum.dyn_auxvar, pendulum.cost_auxvar))
pendulumoc.setStateVariable(pendulum.X)
pendulumoc.setControlVariable(pendulum.U)
dyn = pendulum.X + dt * pendulum.f
pendulumoc.setDyn(dyn)
pendulumoc.setPathCost(pendulum.path_cost)
pendulumoc.setFinalCost(pendulum.final_cost)
lqr_solver = PDP.LQR()

# --------------------------- demonstration generation ----------------------------------------
# generate demonstration using optimal control
true_parameter = [1, 1, 0.1, 10, 1]
horizon = 20
demos = []
ini_state = np.zeros(pendulumoc.n_state)
ini_q = [0, -1, -0.5, 0.5, 1]
for i in range(5):  # generate 5 dmonstrations with each with different initial q
    ini_state[0] = ini_q[i]
    sol = pendulumoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)
    pendulum.play_animation(len=1, dt=dt, state_traj=sol['state_traj_opt'])
    demos += [sol]
# save
sio.savemat('data/pendulum_demos.mat', {'trajectories': demos,
                                          'dt': dt,
                                          'true_parameter': true_parameter})
