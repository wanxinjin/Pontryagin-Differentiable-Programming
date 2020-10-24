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
env.initDyn()
env.initCost(wu=0.1)
true_parameter = [0.5, 0.5, 1, 1, 6, 1, 1]

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
ini_state=np.array([-pi/3,0,0,0])
horizon = 20
true_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)


# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('data/PDP_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
pdp_parameter_trace = np.squeeze(parameter_trace)
pdp_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=pdp_parameter_trace[-1, :])
# uav.play_animation(wing_len=1.5, state_traj=pdp_sol['state_traj_opt'], state_traj_ref=true_sol['state_traj_opt'])


# --------------------------- load kkt results ----------------------------------------
load_data = sio.loadmat('data/KKT_results_trial_0.mat')
parameter_trace = load_data['results']['parameter_trace'][0,0]
kkt_parameter_trace = np.squeeze(parameter_trace)
kkt_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=kkt_parameter_trace[0, :])
# uav.play_animation(wing_len=1.5, state_traj=kkt_sol['state_traj_opt'], state_traj_ref=true_sol['state_traj_opt'])


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
# uav.play_animation(wing_len=1.5, state_traj=nn_state_traj, state_traj_ref=true_sol['state_traj_opt'])

# --------------------------- plot ----------------------------------------
params = {'axes.labelsize': 30,
          'axes.titlesize': 20,
          'xtick.labelsize':20,
          'ytick.labelsize':20,
          'legend.fontsize':20}
plt.rcParams.update(params)
fig = plt.figure(figsize=(11, 8))
ax = fig.subplots()
line_gt,=ax.plot(true_sol['control_traj_opt'], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
line_pdp,=ax.plot(pdp_sol['control_traj_opt'], color ='#A2142F', linewidth=5)
line_kkt,=ax.plot(kkt_sol['control_traj_opt'], color ='#EDB120', linewidth=5)
line_nn,=ax.plot(nn_control_traj, color ='#77AC30', linewidth=5)
ax.set_ylabel('Cart force')
ax.set_xlabel('Time')
ax.set_facecolor('#E6E6E6')
ax.grid()

fig.suptitle('Cartpole planning', fontsize=40)

plt.legend([line_gt, line_pdp, line_kkt, line_nn], ['Ground truth', 'PDP', 'Inverse KKT', 'Neural policy'], facecolor='white', framealpha=0.5,
           loc='best')
plt.show()