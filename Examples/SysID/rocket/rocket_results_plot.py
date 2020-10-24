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

# # --------------------------- plot the pdp loss results ----------------------------------------
# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('data/PDP_SysID_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]

fig = plt.figure(1, figsize=(12, 8))
ax = fig.subplots()
ax.set_xlim(0,2000)
ax.set_yscale('log')
# ax.set_ylim(bottom=1e-4,top=1e4)
ax.set_xlabel('Iteration')
ax.set_ylabel('SysID Loss')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([.18,.15,.75,.7])
ax.set_title('Dynamics identification', pad=20)
# plot kkt results
for  pdp_loss in pdp_loss_list:
    ax.plot(pdp_loss,  linewidth=4)
plt.show()


# --------------------------- load environment ----------------------------------------
rocket = JinEnv.Rocket()
rocket.initDyn()
true_parameter = [0.5, 1,1,1,1]
# --------------------------- create PDP SysID object ----------------------------------------
dt = 0.2
id = PDP.SysID()
id.setAuxvarVariable(rocket.dyn_auxvar)
id.setStateVariable(rocket.X)
id.setControlVariable(rocket.U)
dyn = rocket.X + dt * rocket.f
id.setDyn(dyn)
# set initial state
ini_r_I=[-10, -8, 5.]
ini_v_I = [.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0.5,[0,1,-1])
ini_w = [0, -0.1, 0.1]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon=10
inputs=np.random.rand(horizon, id.n_control)
# inputs=np.zeros((horizon, id.n_control))
true_sol = id.integrateDyn(ini_state=ini_state, inputs=inputs, auxvar_value=true_parameter)
print(id.state)

# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('data/PDP_SysID_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
pdp_parameter_trace = np.squeeze(parameter_trace)
pdp_sol = id.integrateDyn(ini_state=ini_state, inputs=inputs, auxvar_value=pdp_parameter_trace[-1, :])

fig = plt.figure(3,figsize=(18, 8))
ax = fig.subplots(2, 3)
line0,=ax[0,0].plot(true_sol[:,3],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line1,=ax[0,0].plot(pdp_sol[:,3],color = '#A2142F', linewidth=5)
ax[0,0].set_ylabel(r'${v}_x$')
ax[0,0].set_facecolor('#E6E6E6')
ax[0,1].plot(true_sol[:,4],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[0,1].plot(pdp_sol[:,4],color = '#A2142F', linewidth=5)
ax[0,1].set_ylabel(r'${v}_y$')
ax[0,1].set_facecolor('#E6E6E6')
ax[0,2].plot(true_sol[:,5],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[0,2].plot(pdp_sol[:,5],color = '#A2142F', linewidth=5)
ax[0,2].set_ylabel(r'${v}_z$')
ax[0,2].set_facecolor('#E6E6E6')

ax[1,0].plot(true_sol[:,10],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1,0].plot(pdp_sol[:,10],color = '#A2142F', linewidth=5)
ax[1,0].set_ylabel(r'${\omega}_x$')
ax[1,0].set_facecolor('#E6E6E6')
ax[1,0].set_ylim(-0.1,0.1)
ax[1,0].set_xlabel('Time')

ax[1,1].plot(true_sol[:,11],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1,1].plot(pdp_sol[:,11],color = '#A2142F', linewidth=5)
ax[1,1].set_ylabel(r'${\omega}_y$')
ax[1,1].set_facecolor('#E6E6E6')
ax[1,1].set_xlabel('Time')

ax[1,2].plot(true_sol[:,12],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1,2].plot(pdp_sol[:,12],color = '#A2142F', linewidth=5)
ax[1,2].set_facecolor('#E6E6E6')
ax[1,2].set_ylabel(r'${\omega}_z$')
ax[1,2].set_xlabel('Time')

ax[0,2].legend([line0, line1], ['truth', 'PDP'], facecolor='white', framealpha=0.5,loc='best')

plt.subplots_adjust(wspace=0.55, bottom = 0.15, right = 0.95)

fig.suptitle('Prediction using learned dynamics', fontsize=50)

plt.show()