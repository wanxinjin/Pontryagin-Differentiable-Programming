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
uav = JinEnv.Quadrotor()
Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
uav.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
wr, wv, wq, ww = 1, 1, 5, 1
uav.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.1
uavoc = PDP.ControlPlanning()
uavoc.setStateVariable(uav.X)
uavoc.setControlVariable(uav.U)
dyn = uav.X + dt * uav.f
uavoc.setDyn(dyn)
uavoc.setPathCost(uav.path_cost)
uavoc.setFinalCost(uav.final_cost)
# set the horizon and initial condition
horizon = 35
# set initial state
ini_r_I = [-8, -6, 9.]
ini_v_I = [0.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0, [1, -1, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w

# --------------------------- create PDP true oc solver ----------------------------------------
true_uavoc = PDP.OCSys()
true_uavoc.setStateVariable(uav.X)
true_uavoc.setControlVariable(uav.U)
true_uavoc.setDyn(dyn)
true_uavoc.setPathCost(uav.path_cost)
true_uavoc.setFinalCost(uav.final_cost)
true_sol = true_uavoc.ocSolver(ini_state=ini_state, horizon=horizon)
print(true_sol['cost'])

# --------------------------- load the iLQR data ----------------------------------------
load_data=sio.loadmat('data/iLQR_results_trial_0.mat')
ilqr_sol=load_data['results']['solved_solution'][0,0]

# --------------------------- load the PDP data ----------------------------------------
load_data=sio.loadmat('data/PDP_OC_results_trial_0.mat')
pdp_sol=load_data['results']['solved_solution'][0,0]


# --------------------------- load the PDP_RM data ----------------------------------------
load_data=sio.loadmat('data/PDP_Recmat_results_trial_5.mat')
rm_sol=load_data['results']['solved_solution'][0,0]


# --------------------------- plot ----------------------------------------
params = {'axes.labelsize': 30,
          'axes.titlesize': 20,
          'xtick.labelsize':20,
          'ytick.labelsize':20,
          'legend.fontsize':20}
plt.rcParams.update(params)
fig = plt.figure(figsize=(11, 9))
ax=fig.subplots(2,2)

line_gt,=ax[0,0].plot(true_sol['control_traj_opt'][:,1],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line_pdp,=ax[0,0].plot(pdp_sol['control_traj'][0,0][:,1],color = '#A2142F', linewidth=5)
line_ilqr,=ax[0,0].plot(ilqr_sol['control_traj'][0,0][:,1],color = '#EDB120', linewidth=7)
line_rm,=ax[0,0].plot(rm_sol['control_traj'][0,0][:,1],color=	'#7E2F8E', linewidth=5)
ax[0,0].set_ylabel('Thrust 2')
ax[0,0].set_facecolor('#E6E6E6')
ax[0,0].grid()

ax[0,1].plot(true_sol['control_traj_opt'][:,0],color = '#0072BD', linewidth=12, linestyle='dashed', alpha=0.7)
ax[0,1].plot(pdp_sol['control_traj'][0,0][:,0],color = '#A2142F', linewidth=5)
ax[0,1].plot(ilqr_sol['control_traj'][0,0][:,0],color = '#EDB120', linewidth=6)
ax[0,1].plot(rm_sol['control_traj'][0,0][:,0],color = '#7E2F8E', linewidth=4)
ax[0,1].yaxis.set_label_position("right")
ax[0,1].set_ylabel('Thrust 1')
ax[0,1].set_facecolor('#E6E6E6')
ax[0,1].grid()

ax[1,0].plot(true_sol['control_traj_opt'][:,2],color = '#0072BD', linewidth=12, linestyle='dashed', alpha=0.7)
ax[1,0].plot(pdp_sol['control_traj'][0,0][:,2],color = '#A2142F', linewidth=5)
ax[1,0].plot(ilqr_sol['control_traj'][0,0][:,2],color = '#EDB120', linewidth=8)
ax[1,0].plot(rm_sol['control_traj'][0,0][:,2],color = '#7E2F8E', linewidth=4)
ax[1,0].set_xlabel('Time')
ax[1,0].set_ylabel('Thrust 3')
ax[1,0].set_facecolor('#E6E6E6')
ax[1,0].grid()

ax[1,1].plot(true_sol['control_traj_opt'][:,3],color = '#0072BD', linewidth=12, linestyle='dashed', alpha=0.7)
ax[1,1].plot(pdp_sol['control_traj'][0,0][:,3],color = '#A2142F', linewidth=5)
ax[1,1].plot(ilqr_sol['control_traj'][0,0][:,3],color = '#EDB120', linewidth=8)
ax[1,1].plot(rm_sol['control_traj'][0,0][:,3],color = '#7E2F8E', linewidth=4)
ax[1,1].yaxis.set_label_position("right")
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Thrust 4')
ax[1,1].set_facecolor('#E6E6E6')
ax[1,1].grid()


ax[0,1].legend([line_ilqr,line_pdp, line_rm, line_gt],[ 'iLQR','PDP, N=5', 'PDP, N=35', 'by OC solver'],facecolor='white',framealpha=0.5,
           loc='best',  ncol=1)

plt.subplots_adjust(wspace=0.3)

fig.suptitle('UAV optimal control', fontsize=40)

plt.show()

