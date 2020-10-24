import sys

sys.path.insert(1, '../')
import PDP, JinEnv
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
from casadi import *

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


# --------------------------- load pdp results ----------------------------------------
load_data = sio.loadmat('PDP_OC_results_trial_6.mat')
parameter_trace = load_data['results']['parameter_trace'][0, 0]
rocketoc.recmat_init_step(horizon, -1)
print(parameter_trace.shape)
# rocket.play_animation(rocket_len=2, state_traj=pdp_sol['state_traj'], control_traj=pdp_sol['control_traj'],
#                       dt=0.1,
#                       save_option=0, title='Comparison with ground-truth from OC solver')



params = {'axes.labelsize': 20,
          'axes.titlesize': 25,
          'xtick.labelsize':15,
          'ytick.labelsize':15,
          'legend.fontsize':25}
plt.rcParams.update(params)


fig = plt.figure(0, figsize=(9, 5))


ax = fig.add_subplot(1,2,1, projection='3d')
ax.set_xlabel('East (m)', labelpad=15)
ax.set_ylabel('North (m)', labelpad=15)
ax.set_zlabel('Upward (m)', labelpad=10)
ax.set_zlim(0,12)
ax.set_zticks(np.arange(0,13,3))
ax.set_xlim(-8, 10)
ax.set_xticks(np.arange(-8,9,4))
ax.set_ylim(-6.5, 6.5)
ax.set_yticks(np.arange(-6,7,3))
ax.set_title('Learning to control \n rocket powered landing', pad=0, fontsize=25)
# target landing point
p = Circle((0, 0), 3, color='g', alpha=0.2, zorder=-1000)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=-.5, zdir="z")

# ax.view_init(elev=28, azim=-41)
ax.view_init(elev=23, azim=-49)

rocket_len=2

if True:
    pdp_sol = rocketoc.recmat_unwarp(ini_state, horizon, parameter_trace[-1, :])
    state_traj = pdp_sol['state_traj']
    control_traj = pdp_sol['control_traj']

    position = rocket.get_rocket_body_position(rocket_len, state_traj, control_traj)
    sim_horizon = np.size(position, 0) - 1
    for t in range(np.size(position, 0)):
        x = position[t, 0]
        if x < 0:
            sim_horizon = t
            break

    time_list=[t for t in range(18)]+[t for t in range(18,sim_horizon,4)]
    for t in time_list:
        ax.plot(position[:t, 1], position[:t, 2], position[:t, 0], color='#0000FF', alpha=1.0/sim_horizon*t)
        xg, yg, zg, xh, yh, zh, xf, yf, zf = position[t, 3:]
        ax.plot([yg, yh], [zg, zh], [xg, xh], linewidth=5, color='black', alpha=0.1+0.6/sim_horizon*t)
        ax.plot([yg, yf], [zg, zf], [xg, xf], linewidth=2, color='red', alpha=0.2+0.8/sim_horizon*t)

    ax.plot(position[:sim_horizon, 1], position[:sim_horizon, 2], position[:sim_horizon, 0], color='#0000FF')
    xg, yg, zg, xh, yh, zh, xf, yf, zf = position[sim_horizon, 3:]
    ax.plot([yg, yh], [zg, zh], [xg, xh], linewidth=5, color='black')


# ============================================================
# --------------------------- load environment ----------------------------------------
uav = JinEnv.Quadrotor()
uav.initDyn(c=0.01)
uav.initCost(wthrust=0.1)
true_parameter = [1, 1, 1, 1, 0.4, 1, 1, 5, 1]

# --------------------------- pdp object --------------------------------------
dt = 0.1
oc = PDP.OCSys()
oc.setAuxvarVariable(vertcat(uav.dyn_auxvar, uav.cost_auxvar))
oc.setStateVariable(uav.X)
oc.setControlVariable(uav.U)
dyn = uav.X + dt * uav.f
oc.setDyn(dyn)
oc.setPathCost(uav.path_cost)
oc.setFinalCost(uav.final_cost)


# set initial state
ini_r_I=[8, -7, 16.]
ini_v_I = [-9.0, -8.0, 6.0]
ini_q = JinEnv.toQuaternion(1.5,[0,-1,10])
ini_w = [-3.0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon = 30
true_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)


ax = fig.add_subplot(1,2,2, projection='3d')
ax.set_xlabel('X (m)', labelpad=15)
ax.set_ylabel('Y (m)', labelpad=15)
ax.set_zlabel('Z (m)', labelpad=10)
ax.set_zlim(0,12)
ax.set_zticks(np.arange(0,13,3))
ax.set_xlim(-8, 10)
ax.set_xticks(np.arange(-8,9,4))
ax.set_ylim(-3, 3)
ax.set_yticks(np.arange(-6,7,3))
ax.set_title('Imitation learning for \n UAV maneuvering', pad=5, fontsize=25)
ax.view_init(elev=23, azim=-49)

state_traj_ref=true_sol['state_traj_opt']
position_ref = uav.get_quadrotor_position(wing_len=1.5, state_traj=state_traj_ref)
sim_horizon_ref = np.size(position_ref, 0)-1

line_traj_ref, = ax.plot(position_ref[:sim_horizon_ref, 0], position_ref[:sim_horizon_ref, 1], position_ref[:sim_horizon_ref, 2], color='#4DBEEE', linewidth=7,
                         alpha=1)


# pdp_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=pdp_parameter_trace[-1,:])
pdp_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)
state_traj = pdp_sol['state_traj_opt']
position= uav.get_quadrotor_position(wing_len=2.5, state_traj=state_traj)
sim_horizon = np.size(position, 0) - 1
time_list=[t for t in range(18)]+[t for t in range(18,sim_horizon,2)]
line_traj, = ax.plot(position[:1, 0], position[:1, 1], position[:1, 2],color='#0000FF',linewidth=4)
for t in time_list:
    ax.plot(position[:t, 0], position[:t, 1], position[:t, 2],color='#0000FF',linewidth=2)
    c_x, c_y, c_z = position[t, 0:3]
    r1_x, r1_y, r1_z = position[t, 3:6]
    r2_x, r2_y, r2_z = position[t, 6:9]
    r3_x, r3_y, r3_z = position[t, 9:12]
    r4_x, r4_y, r4_z = position[t, 12:15]
    line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color='#A2142F', marker='o', markersize=3,)
    line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color='#7E2F8E', marker='o', markersize=3,)
    line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color='#D95319', marker='o', markersize=3, )
    line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color='#77AC30', marker='o', markersize=3, )

ax.legend([line_traj, line_traj_ref], ['learner', 'expert'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.45, 0.3, 0.3), fontsize=20)

plt.subplots_adjust(wspace=0.12, left=0.0, right=0.950)

plt.show()