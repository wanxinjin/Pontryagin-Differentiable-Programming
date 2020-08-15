from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time

# --------------------------- load environment ----------------------------------------
cartpole = JinEnv.CartPole()
cartpole.initDyn()
cartpole.initCost()

# --------------------------- create PDP SysID object ----------------------------------------
# create a pdp object
dt = 0.05
cartpoleid = PDP.SysID()
cartpoleid.setAuxvarVariable(cartpole.dyn_auxvar)
cartpoleid.setStateVariable(cartpole.X)
cartpoleid.setControlVariable(cartpole.U)
dyn = cartpole.X + dt * cartpole.f
cartpoleid.setDyn(dyn)

# --------------------------- generate experimental data ----------------------------------------
# true_parameter
true_parameter = [1, 1, 1]
# generate the rand inputs
batch_inputs = cartpoleid.getRandomInputs(n_batch=3, lb=[-10], ub=[10], horizon=20)
batch_states = []
for i in range(len(batch_inputs)):
    ini_state = np.random.randn(cartpoleid.n_state)
    states = cartpoleid.integrateDyn(auxvar_value=true_parameter, ini_state=ini_state, inputs=batch_inputs[i])
    batch_states += [states]

# save the data
cartpole_iodata = {'batch_inputs': batch_inputs,
                   'batch_states': batch_states,
                   'true_parameter': true_parameter}
sio.savemat('./data/cartpole_iodata.mat', {'cartpole_iodata': cartpole_iodata})