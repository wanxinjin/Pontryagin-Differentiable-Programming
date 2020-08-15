from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt

# --------------------------- load environment ----------------------------------------
robotarm = JinEnv.RobotArm()
robotarm.initDyn(g=0)
robotarm.initCost(wu=0.01)

# --------------------------- define PDP ----------------------------------------
armoc = PDP.OCSys()
armoc.setAuxvarVariable(vertcat(robotarm.dyn_auxvar, robotarm.cost_auxvar))
armoc.setControlVariable(robotarm.U)
armoc.setStateVariable(robotarm.X)
dt = 0.1
dyn = robotarm.X + dt * robotarm.f
armoc.setDyn(dyn)
armoc.setPathCost(robotarm.path_cost)
armoc.setFinalCost(robotarm.final_cost)


# --------------------------- demonstration generation ----------------------------------------
# generate demonstration using optimal control
true_parameter = [1, 1, 1, 1,  1, 1, 0.5, 0.5]
horizon = 35
demos = []
ini_q1 = [-math.pi / 2, -3 * math.pi / 4, -math.pi / 4, -math.pi/2]
ini_q2 = [0, -math.pi / 2, math.pi / 2, math.pi]
for i in range(4):
    ini_state = [ini_q1[i], ini_q2[i], 0, 0]
    sol = armoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter, costate_option=0)
    # simulation
    robotarm.play_animation(l1=1, l2=1, dt=dt, state_traj=sol['state_traj_opt'], save_option=0)
    demos += [sol]

# save
sio.savemat('./data/robotarm_demos.mat',{'trajectories':demos,
                                         'dt':dt,
                                         'true_parameter': true_parameter})


