from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------------- model specification ----------------------------------------
cartpole = JinEnv.CartPole()
cartpole.initDyn()
cartpole.initCost(wu=0.1)

# --------------------------- create PDP object ----------------------------------------
cartpoleoc = PDP.OCSys()
cartpoleoc.setAuxvarVariable(vertcat(cartpole.dyn_auxvar, cartpole.cost_auxvar))
cartpoleoc.setStateVariable(cartpole.X)
cartpoleoc.setControlVariable(cartpole.U)
dt = 0.1
dyn = dt * cartpole.f + cartpole.X
cartpoleoc.setDyn(dyn)
cartpoleoc.setPathCost(cartpole.path_cost)
cartpoleoc.setFinalCost(cartpole.final_cost)
lqr_solver = PDP.LQR()

# --------------------------- demonstration generation ----------------------------------------
true_parameter = [0.5, 0.5, 1, 1, 6, 1, 1]
horizon = 30
demos = []
ini_state = np.zeros(cartpoleoc.n_state)
ini_q = [0, -0.5, -0.25, 0.25, 0.5]
for i in range(5):  # generate 5 dmonstrations with each with different initial q
    ini_state[1] = ini_q[i]
    sol = cartpoleoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)
    cartpole.play_animation(pole_len=2, dt=dt, state_traj=sol['state_traj_opt'])
    # plt.plot(sol['control_traj_opt'])
    demos += [sol]
# save
sio.savemat('./data/cartpole_demos.mat', {'trajectories': demos,
                                          'dt': dt,
                                          'true_parameter': true_parameter})
