from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes

# --------------------------- load environment ----------------------------------------
env = JinEnv.CartPole()
mc, mp, l = 0.1, 0.1, 1
env.initDyn(mc=mc, mp=mp, l=l)
wx, wq, wdx, wdq, wu = 0.1, 0.6, 0.1, 0.1, 0.3
env.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu=wu)

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.05
oc = PDP.ControlPlanning()
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
dyn = env.X + dt * env.f
oc.setDyn(dyn)
oc.setPathCost(env.path_cost)
oc.setFinalCost(env.final_cost)
# initial state and horizon
horizon = 25
ini_state = [0, 0, 0, 0]

# --------------------------- create PDP true oc solver ----------------------------------------
true_uavoc = PDP.OCSys()
true_uavoc.setStateVariable(env.X)
true_uavoc.setControlVariable(env.U)
true_uavoc.setDyn(dyn)
true_uavoc.setPathCost(env.path_cost)
true_uavoc.setFinalCost(env.final_cost)
true_sol = true_uavoc.ocSolver(ini_state=ini_state, horizon=horizon)
print(true_sol['cost'])

# --------------------------- load PDP learned neural data ----------------------------------------
load_data=sio.loadmat('data/PDP_Neural_trial_5.mat')
pdp_neural_sol=load_data['results']['solved_solution'][0,0]

# --------------------------- load GPS learned neural data ----------------------------------------
load_data=sio.loadmat('data/GPS_Neural_trial_6.mat')
gps_neural_sol=load_data['results']['solved_solution'][0,0]

# --------------------------- plot ----------------------------------------
params = {'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize':25,
          'ytick.labelsize':25,
          'legend.fontsize':25}
plt.rcParams.update(params)
fig = plt.figure(figsize=(11, 9))
ax=fig.subplots()

line_gt,=ax.plot(true_sol['control_traj_opt'][:,0],color = '#0072BD', linewidth=7, linestyle='dashed', alpha=0.7)
line_pdp_neural,=ax.plot(pdp_neural_sol['control_traj'][0,0][:,0],color = '#A2142F', linewidth=5)
line_gps_neural,=ax.plot(gps_neural_sol['control_traj'][0,0][:,0],color = '#77AC30', linewidth=5)
ax.set_ylabel('Cart force')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_xlabel('Time')



ax.legend([line_pdp_neural,line_gps_neural, line_gt],[ 'Neural policy learned by PDP', 'Neural policy learned by GPS', 'Trajectory solved by OC solver'],facecolor='white',framealpha=0.5,
           loc='best',  ncol=1)
ax.set_position(pos=[0.15,0.1,0.80,0.8])

fig.suptitle('Cartpole optimal control', fontsize=40)

plt.show()

