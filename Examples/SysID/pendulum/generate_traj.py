from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time

# --------------------------- load environment ----------------------------------------
pendulum = JinEnv.SinglePendulum()
pendulum.initDyn()
pendulum.initCost()

# --------------------------- create PDP SysID object ----------------------------------------
# create a pdp object
dt = 0.05
pendulumid = PDP.SysID()
pendulumid.setAuxvarVariable(pendulum.dyn_auxvar)
pendulumid.setStateVariable(pendulum.X)
pendulumid.setControlVariable(pendulum.U)
dyn = pendulum.X + dt * pendulum.f
pendulumid.setDyn(dyn)

# --------------------------- generate experimental data ----------------------------------------
# true_parameter
true_parameter = [1, 1, 0.05]
# generate the rand inputs
batch_inputs = pendulumid.getRandomInputs(n_batch=3, lb=[-80], ub=[80], horizon=20)
batch_states = []
for i in range(len(batch_inputs)):
    ini_state = np.random.randn(pendulumid.n_state)
    states = pendulumid.integrateDyn(auxvar_value=true_parameter, ini_state=ini_state, inputs=batch_inputs[i])
    batch_states += [states]

# save the data
pendulum_iodata = {'batch_inputs': batch_inputs,
                   'batch_states': batch_states,
                   'true_parameter': true_parameter}
sio.savemat('data/pendulum_iodata.mat', {'pendulum_iodata': pendulum_iodata})
