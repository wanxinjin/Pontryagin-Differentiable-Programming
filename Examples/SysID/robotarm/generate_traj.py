from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time

# --------------------------- load environment ----------------------------------------
arm = JinEnv.RobotArm()
arm.initDyn(g=0)

# --------------------------- create PDP SysID object ----------------------------------------
# create a pdp object
dt = 0.1
armid = PDP.SysID()
armid.setAuxvarVariable(arm.dyn_auxvar)
armid.setStateVariable(arm.X)
armid.setControlVariable(arm.U)
dyn = arm.X + dt * arm.f
armid.setDyn(dyn)

# --------------------------- generate experimental data ----------------------------------------
# true_parameter
true_parameter = [1, 1, 1, 1]
# generate the rand inputs
batch_inputs = armid.getRandomInputs(n_batch=3, lb=[-5,-5], ub=[5,5], horizon=10)
batch_states = []
for i in range(len(batch_inputs)):
    ini_state = np.random.randn(armid.n_state)
    states = armid.integrateDyn(auxvar_value=true_parameter, ini_state=ini_state, inputs=batch_inputs[i])
    batch_states += [states]

# save the data
robotarm_iodata = {'batch_inputs': batch_inputs,
                   'batch_states': batch_states,
                   'true_parameter': true_parameter}
sio.savemat('./data/robotarm_iodata.mat', {'robotarm_iodata': robotarm_iodata})