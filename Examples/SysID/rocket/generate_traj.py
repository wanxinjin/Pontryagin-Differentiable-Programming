from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time

# --------------------------- load environment ----------------------------------------
rocket = JinEnv.Rocket()
rocket.initDyn()

# --------------------------- create PDP SysID object ----------------------------------------
dt = 0.2
uavid = PDP.SysID()
uavid.setAuxvarVariable(rocket.dyn_auxvar)
uavid.setStateVariable(rocket.X)
uavid.setControlVariable(rocket.U)
dyn = rocket.X + dt * rocket.f
uavid.setDyn(dyn)

# --------------------------- generate experimental data ----------------------------------------
# true_parameter
true_parameter = [0.5, 1,1,1,1]
# generate the rand inputs
batch_inputs = uavid.getRandomInputs(n_batch=2, lb=[-10,-10, -10], ub=[10,10, 10], horizon=10)
batch_states = []
# set initial state
ini_r_I=[10, -8, 5.]
ini_v_I = [.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0.5,[0,1,-1])
ini_w = [0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
for i in range(len(batch_inputs)):
    states = uavid.integrateDyn(auxvar_value=true_parameter, ini_state=ini_state, inputs=batch_inputs[i])
    batch_states += [states]

# save the data
rocket_iodata = {'batch_inputs': batch_inputs,
                   'batch_states': batch_states,
                   'true_parameter': true_parameter}
sio.savemat('./data/rocket_iodata.mat', {'rocket_iodata': rocket_iodata})