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
optimal_cost=true_sol['cost']
# rocket.play_animation(rocket_len=2,state_traj=true_sol['state_traj_opt'],control_traj=true_sol['control_traj_opt'])

# --------------------------- load the PDP_RM data ----------------------------------------
# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('./data/PDP_OC_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]
print(pdp_loss_list[0][-1])

params = {'axes.labelsize': 50,
          'axes.titlesize': 50,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize':30}
plt.rcParams.update(params)

fig = plt.figure(1, figsize=(12, 8))
ax = fig.subplots()
# ax.set_yscale('log')
ax.set_xlim(0,10000)
ax.set_xlabel('Iteration')
ax.set_ylabel('Control loss')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([.19,.15,.75,.7])
ax.set_title('Control for powered landing', pad=20)
# plot pdp results
for  pdp_loss in pdp_loss_list:
    ax.plot(pdp_loss[13:],  linewidth=4)
plt.show()


# --------------------------- load pdp results ----------------------------------------
load_data=sio.loadmat('data/PDP_OC_results_trial_0.mat')
parameter_trace=load_data['results']['parameter_trace'][0,0]
rocketoc.recmat_init_step(horizon, -1)
pdp_sol = rocketoc.recmat_unwarp(ini_state, horizon, parameter_trace[-1,:])


fig = plt.figure(2, figsize=(18, 8))
ax = fig.subplots(1,3)
ax[0].plot(true_sol['control_traj_opt'][:,0], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[0].plot(pdp_sol['control_traj'][:,0], color ='#A2142F', linewidth=5)
ax[0].set_ylabel(r'$T_x $ [N]' )
ax[0].set_facecolor('#E6E6E6')
ax[0].grid()
ax[0].set_xlabel('Time')

ax[1].plot(true_sol['control_traj_opt'][:,1], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1].plot(pdp_sol['control_traj'][:,1], color ='#A2142F', linewidth=5)
ax[1].set_ylabel(r'$T_y$ [N]')
ax[1].set_facecolor('#E6E6E6')
ax[1].grid()
ax[1].set_xlabel('Time')

line_gt,=ax[2].plot(true_sol['control_traj_opt'][:,2], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line_pdp,=ax[2].plot(pdp_sol['control_traj'][:,2], color ='#A2142F', linewidth=5)
ax[2].set_ylabel(r'$T_z$ [N]')
ax[2].set_facecolor('#E6E6E6')
ax[2].grid()
ax[2].set_xlabel('Time')
ax[2].legend([line_gt, line_pdp], ['OC solver', 'PDP'], facecolor='white', framealpha=0.5,loc='best')


# ax.set_position([.15,.15,.8,.7])
fig.suptitle('Learned control for powered landing', fontsize=50)
plt.subplots_adjust(wspace=0.6, bottom = 0.15, right = 0.95)

plt.show()


