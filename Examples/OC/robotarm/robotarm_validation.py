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
env = JinEnv.RobotArm()
l1, m1, l2, m2, g = 1, 1, 1, 1, 0
env.initDyn(l1=l1, m1=m1, l2=l2, m2=m2, g=0)
wq1, wq2, wdq1, wdq2, wu = 0.1, 0.1, 0.1, 0.1, 0.01
env.initCost(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2, wu=wu)

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.1
oc = PDP.ControlPlanning()
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
dyn = env.X + dt * env.f
oc.setDyn(dyn)
oc.setPathCost(env.path_cost)
oc.setFinalCost(env.final_cost)
# initial state and horizon
horizon = 20
ini_state = [pi/4, pi/2, 0, 0]

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
          'axes.titlesize': 25,
          'xtick.labelsize':25,
          'ytick.labelsize':25,
          'legend.fontsize':25}
plt.rcParams.update(params)
fig = plt.figure(figsize=(11, 9))
ax=fig.subplots(2,1)

line_gt,=ax[0].plot(true_sol['control_traj_opt'][:,0],color = '#0072BD', linewidth=7, linestyle='dashed', alpha=0.7)
line_pdp,=ax[0].plot(pdp_neural_sol['control_traj'][0,0][:,0],color = '#A2142F', linewidth=5)
line_gps_neural,=ax[0].plot(gps_neural_sol['control_traj'][0,0][:,0],color = '#77AC30', linewidth=5)
ax[0].set_ylabel('Joint torque 1')
ax[0].set_facecolor('#E6E6E6')
ax[0].grid()
ax[0].set_ylim([-2,4])

ax[1].plot(true_sol['control_traj_opt'][:,1],color = '#0072BD', linewidth=7, linestyle='dashed', alpha=0.7)
ax[1].plot(pdp_neural_sol['control_traj'][0,0][:,1],color = '#A2142F', linewidth=5)
ax[1].plot(gps_neural_sol['control_traj'][0,0][:,1],color = '#77AC30', linewidth=5)
ax[1].set_ylabel('Joint torque 2')
ax[1].set_facecolor('#E6E6E6')
ax[1].grid()
ax[1].set_xlabel('Time')

ax[0].legend([line_pdp, line_gps_neural, line_gt],[ 'Neural policy learned by PDP','Neural policy learned by GPS', 'Trajectory solved by OC solver'],facecolor='white',framealpha=0.5,
           loc='best',  ncol=1)

plt.subplots_adjust(wspace=0.3,right=0.95)

fig.suptitle('Robot arm optimal control', fontsize=40)

plt.show()

