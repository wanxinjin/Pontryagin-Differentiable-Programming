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
uav.initDyn(c=0.01)
uav.initCost(wthrust=0.1)
true_parameter = [1, 1, 1, 1, 0.4, 1, 1, 5, 1]


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

# set initial state
ini_r_I=[-8, -8, 9.]
ini_v_I = [0.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0,[1,0,0])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon = 50
true_sol = uavoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)


# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('data/PDP_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
pdp_parameter_trace = np.squeeze(parameter_trace)
pdp_sol = uavoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=pdp_parameter_trace[-1,:])
# uav.play_animation(wing_len=1.5, state_traj=pdp_sol['state_traj_opt'], state_traj_ref=true_sol['state_traj_opt'])


# --------------------------- load kkt results ----------------------------------------
load_data = sio.loadmat('data/KKT_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
kkt_parameter_trace = np.squeeze(parameter_trace)
kkt_sol = uavoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=kkt_parameter_trace[0,:])
# uav.play_animation(wing_len=1.5, state_traj=kkt_sol['state_traj_opt'], state_traj_ref=true_sol['state_traj_opt'])


# --------------------------- load nn results ----------------------------------------
# setup the neural network to train
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(uavoc.n_state, 10 * uavoc.n_state)
        self.output = nn.Linear(10 * uavoc.n_state, uavoc.n_control)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x
policy_model=PolicyNetwork()
policy_model.load_state_dict(torch.load('./data/NN_policy_model.pth'))
policy_model.eval()
nn_control_traj=numpy.zeros((horizon,uavoc.n_control))
nn_state_traj=numpy.zeros((horizon+1,uavoc.n_state))
nn_state_traj[0,:]=np.array(ini_state)
nn_state=torch.tensor(ini_state).float()
for t in range(horizon):
    nn_control=policy_model(nn_state)
    nn_control=nn_control.detach().numpy()
    nn_state=uavoc.dyn_fn(nn_state.numpy(),nn_control,true_parameter).full().flatten()
    nn_state=torch.from_numpy(nn_state).float()
    nn_control_traj[t,:]=nn_control
    nn_state_traj[t+1,:]=nn_state
# uav.play_animation(wing_len=1.5, state_traj=nn_state_traj, state_traj_ref=true_sol['state_traj_opt'])

# --------------------------- plot ----------------------------------------
params = {'axes.labelsize': 40,
          'axes.titlesize': 30,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize':30}
plt.rcParams.update(params)

fig = plt.figure(figsize=(11, 9))

h = [Size.Fixed(1.8), Size.Fixed(8.5)]
v = [Size.Fixed(1.2), Size.Fixed(7.5)]

divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)

ax = fig.subplots(2, 2)


line0,=ax[0,0].plot(true_sol['control_traj_opt'][:,1],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line1,=ax[0,0].plot(pdp_sol['control_traj_opt'][:,1],color = '#A2142F', linewidth=5)
line2,=ax[0,0].plot(kkt_sol['control_traj_opt'][:,1],color = '#EDB120', linewidth=5)
line3,=ax[0,0].plot(nn_control_traj[:,1],color = '#77AC30', linewidth=5)
ax[0,0].set_ylabel('Thrust 2')
ax[0,0].set_facecolor('#E6E6E6')
ax[0,0].grid()

ax[0,1].plot(true_sol['control_traj_opt'][:,0],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[0,1].plot(pdp_sol['control_traj_opt'][:,0],color = '#A2142F', linewidth=5)
ax[0,1].plot(kkt_sol['control_traj_opt'][:,0],color = '#EDB120', linewidth=5)
ax[0,1].plot(nn_control_traj[:,0],color = '#77AC30', linewidth=5)
ax[0,1].yaxis.set_label_position("right")
ax[0,1].set_ylabel('Thrust 1')
ax[0,1].set_facecolor('#E6E6E6')
ax[0,1].grid()

ax[1,0].plot(true_sol['control_traj_opt'][:,2],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1,0].plot(pdp_sol['control_traj_opt'][:,2],color = '#A2142F', linewidth=5)
ax[1,0].plot(kkt_sol['control_traj_opt'][:,2],color = '#EDB120', linewidth=5)
ax[1,0].plot(nn_control_traj[:,2],color = '#77AC30', linewidth=5)
ax[1,0].set_xlabel('Time')
ax[1,0].set_ylabel('Thrust 3')
ax[1,0].set_facecolor('#E6E6E6')
ax[1,0].grid()

ax[1,1].plot(true_sol['control_traj_opt'][:,3],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1,1].plot(pdp_sol['control_traj_opt'][:,3],color = '#A2142F', linewidth=5)
ax[1,1].plot(kkt_sol['control_traj_opt'][:,3],color = '#EDB120', linewidth=5)
ax[1,1].plot(nn_control_traj[:,3],color = '#77AC30', linewidth=5)
ax[1,1].yaxis.set_label_position("right")
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Thrust 4')
ax[1,1].set_facecolor('#E6E6E6')
ax[1,1].grid()

plt.legend([line0,line1,line2, line3],['Ground truth', 'PDP','Inverse KKT', 'Neural policy'],facecolor='white',framealpha=0.5,
           loc='best', bbox_to_anchor=(.61, 2.15, .5, 0.5), ncol=2)


plt.subplots_adjust(wspace=0.3)

plt.show()