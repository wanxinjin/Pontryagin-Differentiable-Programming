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
env.initDyn(g=0)
env.initCost(wu=0.01)
true_parameter = [1, 1, 1, 1,  1, 1, 0.5, 0.5]

# --------------------------- pdp object ----------------------------------------


# create a pdp object
dt = 0.1
oc = PDP.OCSys()
oc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.cost_auxvar))
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
dyn = env.X + dt * env.f
oc.setDyn(dyn)
oc.setPathCost(env.path_cost)
oc.setFinalCost(env.final_cost)

# set initial state
ini_state=np.array([-pi/3,2*pi/4,0,0])
horizon = 40
true_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)

# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('data/PDP_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
pdp_parameter_trace = np.squeeze(parameter_trace)
pdp_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=pdp_parameter_trace[-1, :])
# env.play_animation(l1=1, l2=1, dt=dt, state_traj=pdp_sol['state_traj_opt'], state_traj_ref=true_sol['state_traj_opt'])


# --------------------------- load kkt results ----------------------------------------
load_data = sio.loadmat('data/KKT_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
kkt_parameter_trace = np.squeeze(parameter_trace)
kkt_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=kkt_parameter_trace[0, :])
# env.play_animation(wing_len=1.5, state_traj=kkt_sol['state_traj_opt'], state_traj_ref=true_sol['state_traj_opt'])


# --------------------------- load nn results ----------------------------------------
# setup the neural network to train
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(oc.n_state, 10 * oc.n_state)
        self.output = nn.Linear(10 * oc.n_state, oc.n_control)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x
policy_model=PolicyNetwork()
policy_model.load_state_dict(torch.load('./data/NN_policy_model.pth'))
policy_model.eval()
nn_control_traj=numpy.zeros((horizon, oc.n_control))
nn_state_traj=numpy.zeros((horizon + 1, oc.n_state))
nn_state_traj[0,:]=np.array(ini_state)
nn_state=torch.tensor(ini_state).float()
for t in range(horizon):
    nn_control=policy_model(nn_state)
    nn_control=nn_control.detach().numpy()
    nn_state=oc.dyn_fn(nn_state.numpy(), nn_control, true_parameter).full().flatten()
    nn_state=torch.from_numpy(nn_state).float()
    nn_control_traj[t,:]=nn_control
    nn_state_traj[t+1,:]=nn_state
# env.play_animation(wing_len=1.5, state_traj=nn_state_traj, state_traj_ref=true_sol['state_traj_opt'])

# # --------------------------- plot ----------------------------------------
params = {'axes.labelsize': 30,
          'axes.titlesize': 20,
          'xtick.labelsize':20,
          'ytick.labelsize':20,
          'legend.fontsize':20}
plt.rcParams.update(params)
fig = plt.figure(figsize=(11, 8))
ax = fig.subplots(2,1)
line_gt,=ax[0].plot(true_sol['control_traj_opt'][:,0], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line_pdp,=ax[0].plot(pdp_sol['control_traj_opt'][:,0], color ='#A2142F', linewidth=5)
line_kkt,=ax[0].plot(kkt_sol['control_traj_opt'][:,0], color ='#EDB120', linewidth=5)
line_nn,=ax[0].plot(nn_control_traj[0:,0], color ='#77AC30', linewidth=5)
ax[0].set_ylabel('Torque 1')
ax[0].set_facecolor('#E6E6E6')
ax[0].grid()

ax[1].plot(true_sol['control_traj_opt'][:,1], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
ax[1].plot(pdp_sol['control_traj_opt'][:,1], color ='#A2142F', linewidth=5)
ax[1].plot(kkt_sol['control_traj_opt'][:,1], color ='#EDB120', linewidth=5)
ax[1].plot(nn_control_traj[:,1], color ='#77AC30', linewidth=5)
ax[1].set_ylabel('Torque 2')
ax[1].set_xlabel('Time')
ax[1].set_facecolor('#E6E6E6')
ax[1].grid()


ax[0].legend([line_gt, line_pdp, line_kkt, line_nn], ['Ground truth', 'PDP', 'Inverse KKT', 'Neural policy'], facecolor='white', framealpha=0.5,
           loc='best')

fig.suptitle('Robot arm planning', fontsize=40)
plt.show()