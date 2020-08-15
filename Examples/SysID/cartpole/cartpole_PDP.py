from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import numpy as np
import time
import scipy.io as sio

# --------------------------- load environment ----------------------------------------
cartpole = JinEnv.CartPole()
cartpole.initDyn()
cartpole.initCost()

# --------------------------- create PDP SysID object ----------------------------------------
dt = 0.05
cartpoleid = PDP.SysID()
cartpoleid.setAuxvarVariable(cartpole.dyn_auxvar)
cartpoleid.setStateVariable(cartpole.X)
cartpoleid.setControlVariable(cartpole.U)
dyn = cartpole.X + dt * cartpole.f
cartpoleid.setDyn(dyn)

# --------------------------- load the data ----------------------------------------
load_data = sio.loadmat('./data/cartpole_iodata.mat')
data = load_data['cartpole_iodata'][0, 0]
true_parameter = data['true_parameter']
n_batch = len(data['batch_inputs'])
batch_inputs = []
batch_states = []
for i in range(n_batch):
    batch_inputs += [data['batch_inputs'][i]]
    batch_states += [data['batch_states'][i]]

# --------------------------- load the data ----------------------------------------
for j in range(1):
    start_time = time.time()
    # learning rate
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    sigma = 2
    initial_parameter = np.array(true_parameter) + sigma * np.random.rand(len(true_parameter)) - sigma / 2
    current_parameter = initial_parameter
    for k in range(int(1e4)):
        # one iteration of PDP
        loss, dp = cartpoleid.step(batch_inputs, batch_states, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter]
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss,)

    # save
    save_data = {'trail_no': j,
                 'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'learning_rate': lr,
                 'time_passed': time.time() - start_time}
    sio.savemat('./PDP_SysID_results_trial_' + str(j) + '.mat', {'results': save_data})
