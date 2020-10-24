from PDP import PDP
from JinEnv import JinEnv
import matplotlib.pyplot as plt
import scipy.io as sio
from casadi import *
import numpy as np

params = {'axes.labelsize': 50,
          'axes.titlesize': 50,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize':30}
plt.rcParams.update(params)

# --------------------------- plot the pdp loss results ----------------------------------------
# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('data/PDP_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]

fig = plt.figure(0, figsize=(12, 8))
ax = fig.subplots()
ax.set_xlim(0,1000)
ax.set_yscale('log')
# ax.set_ylim(bottom=1e-4,top=1e4)
ax.set_xlabel('Iteration')
ax.set_ylabel('Imitation Loss')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([.15,.15,.8,.7])
ax.set_title('Imitation of powered landing', pad=20)
# plot kkt results
for  pdp_loss in pdp_loss_list:
    ax.plot(pdp_loss,  linewidth=4)
plt.show()




# --------------------------- for validation ----------------------------------------
rocket = JinEnv.Rocket()
rocket.initDyn()
rocket.initCost(wthrust=0.1)

# --------------------------- load demos data ----------------------------------------
data = sio.loadmat('data/rocket_demos.mat')
trajectories = data['trajectories']
true_parameter = data['true_parameter']
dt = data['dt']
print(true_parameter)
# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('data/PDP_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
pdp_parameter_trace = np.squeeze(parameter_trace)

# --------------------------- define oc solver and initial condition ----------------------------------------
rocketoc = PDP.OCSys()
rocketoc.setAuxvarVariable(vertcat(rocket.dyn_auxvar, rocket.cost_auxvar))
rocketoc.setControlVariable(rocket.U)
rocketoc.setStateVariable(rocket.X)
dyn = rocket.X + dt * rocket.f
rocketoc.setDyn(dyn)
rocketoc.setPathCost(rocket.path_cost)
rocketoc.setFinalCost(rocket.final_cost)
# set initial state
ini_r_I=[10, -8, 5.]
ini_v_I = [-.5, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(1.,[0,-1,-1])
ini_w = [0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon = 40


# true results
true_sol = rocketoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)
pdp_sol = rocketoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=pdp_parameter_trace[-1, :])


fig = plt.figure(figsize=(18, 8))
ax = fig.subplots(1,3)
ax[0].plot(true_sol['control_traj_opt'][:,0], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[0].plot(pdp_sol['control_traj_opt'][:,0], color ='#A2142F', linewidth=5)
ax[0].set_ylabel(r'$T_x $ [N]' )
ax[0].set_facecolor('#E6E6E6')
ax[0].grid()
ax[0].set_xlabel('Time')

ax[1].plot(true_sol['control_traj_opt'][:,1], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1].plot(pdp_sol['control_traj_opt'][:,1], color ='#A2142F', linewidth=5)
ax[1].set_ylabel(r'$T_y$ [N]')
ax[1].set_facecolor('#E6E6E6')
ax[1].grid()
ax[1].set_xlabel('Time')

line_gt,=ax[2].plot(true_sol['control_traj_opt'][:,2], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line_pdp,=ax[2].plot(pdp_sol['control_traj_opt'][:,2], color ='#A2142F', linewidth=5)
ax[2].set_ylabel(r'$T_z$ [N]')
ax[2].set_facecolor('#E6E6E6')
ax[2].grid()
ax[2].set_xlabel('Time')
ax[2].legend([line_gt, line_pdp], ['truth', 'PDP'], facecolor='white', framealpha=0.5,loc='best')


# ax.set_position([.15,.15,.8,.7])
fig.suptitle('Planning for powered landing', fontsize=50)
plt.subplots_adjust(wspace=0.5, bottom = 0.15, right = 0.95)

plt.show()



