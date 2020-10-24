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
cartpole = JinEnv.CartPole()
cartpole.initDyn()
true_parameter = [1, 1, 1]


# create a pdp object
dt = 0.05
cartpoleid = PDP.SysID()
cartpoleid.setAuxvarVariable(vertcat(cartpole.dyn_auxvar))
cartpoleid.setStateVariable(cartpole.X)
cartpoleid.setControlVariable(cartpole.U)
dyn = cartpole.X + dt * cartpole.f
cartpoleid.setDyn(dyn)

# set initial state
print(cartpoleid.state)
ini_state =[0,pi/2,0,0]
horizon = 40
true_sol = cartpoleid.integrateDyn(ini_state=ini_state, inputs=np.zeros((horizon, cartpoleid.n_control)), auxvar_value=true_parameter)


# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('data/PDP_SysID_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
pdp_parameter_trace = np.squeeze(parameter_trace)
pdp_sol = cartpoleid.integrateDyn(ini_state=ini_state, inputs=np.zeros((horizon, cartpoleid.n_control)), auxvar_value=pdp_parameter_trace[-1, :])
# cartpole.play_animation(pole_len=1.5, dt=0.05, state_traj=pdp_sol, state_traj_ref=true_sol)


# --------------------------- load dmd results ----------------------------------------
load_data = sio.loadmat('data/DMD_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
dmd_parameter_trace = np.squeeze(parameter_trace)

# linearID = ControlTools.SysID_DMD()
# linearID.setDimensions(n_state=cartpoleid.n_state, n_control=cartpoleid.n_control)
# dmd_state_traj=numpy.zeros((horizon+1,cartpoleid.n_state))
# dmd_state_traj[0,:]=numpy.array(ini_state)
# for t in range(horizon):
#     dmd_state_traj[t+1,:]=linearID.dyn_linear_fn(dmd_state_traj[t,:],np.zeros(cartpoleid.n_control),dmd_parameter_trace[-1,:]).full().flatten()
# # cartpole.play_animation(pole_len=1.5, dt=0.05, state_traj=dmd_state_traj, state_traj_ref=true_sol)


# --------------------------- load nn results ----------------------------------------
# setup the neural network to train
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(cartpoleid.n_state+cartpoleid.n_control, 10 * (cartpoleid.n_state+cartpoleid.n_control))
        self.output = nn.Linear(10 * (cartpoleid.n_state+cartpoleid.n_control), cartpoleid.n_state)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x
dyn_model=Network()
dyn_model.load_state_dict(torch.load('./data/NN_dyn_model.pth'))
dyn_model.eval()
nn_state_traj=numpy.zeros((horizon + 1, cartpoleid.n_state))
nn_state_traj[0,:]=np.array(ini_state)
for t in range(horizon):
    nn_state_control=np.concatenate([nn_state_traj[t, :],np.zeros(cartpoleid.n_control)])
    nn_state_control=torch.tensor(nn_state_control).float()
    nn_state=dyn_model(nn_state_control)
    nn_state_traj[t+1,:]=nn_state.detach().numpy()
# cartpole.play_animation(pole_len=1.5, dt=0.05, state_traj=nn_state_traj, state_traj_ref=true_sol)


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


line0,=ax[0,0].plot(true_sol[:,0],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line1,=ax[0,0].plot(pdp_sol[:,0],color = '#A2142F', linewidth=5)
# line2,=ax[0,0].plot(dmd_state_traj[:,0],color = '#EDB120', linewidth=5)
line3,=ax[0,0].plot(nn_state_traj[:,0],color = '#77AC30', linewidth=5)
ax[0,0].set_ylabel('Cart position')
ax[0,0].set_facecolor('#E6E6E6')
ax[0,0].grid()

ax[0,1].plot(true_sol[:,2],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[0,1].plot(pdp_sol[:,2],color = '#A2142F', linewidth=5)
# ax[0,1].plot(dmd_state_traj[:,2],color = '#EDB120', linewidth=5)
ax[0,1].plot(nn_state_traj[:,2],color = '#77AC30', linewidth=5)
ax[0,1].yaxis.set_label_position("right")
ax[0,1].set_ylabel('Cart velocity')
ax[0,1].set_facecolor('#E6E6E6')
ax[0,1].grid()

ax[1,0].plot(true_sol[:,1],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1,0].plot(pdp_sol[:,1],color = '#A2142F', linewidth=5)
# ax[1,0].plot(dmd_state_traj[:,1],color = '#EDB120', linewidth=5)
ax[1,0].plot(nn_state_traj[:,1],color = '#77AC30', linewidth=5)
ax[1,0].set_ylabel('Pole angle')
ax[1,0].set_facecolor('#E6E6E6')
ax[1,0].set_xlabel('Time')
ax[1,0].grid()

ax[1,1].plot(true_sol[:,3],color = '#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1,1].plot(pdp_sol[:,3],color = '#A2142F', linewidth=5)
# ax[1,1].plot(dmd_state_traj[:,3],color = '#EDB120', linewidth=5)
ax[1,1].plot(nn_state_traj[:,3],color = '#77AC30', linewidth=5)
ax[1,1].yaxis.set_label_position("right")
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Pole velocity')
ax[1,1].set_facecolor('#E6E6E6')
ax[1,1].grid()

# plt.legend([line0,line1,line2, line3],['Ground truth', 'PDP','DMDc', 'NN dyn'],facecolor='white',framealpha=0.5,
# loc='best', bbox_to_anchor=(.4, 2.15, .5, 0.5), ncol=2)
plt.legend([line0,line1, line3],['Ground truth', 'PDP', 'NN dyn'],facecolor='white',framealpha=0.5,
           loc='best', bbox_to_anchor=(.4, 2.15, .5, 0.5), ncol=2)
plt.subplots_adjust(wspace=0.30)

plt.show()

