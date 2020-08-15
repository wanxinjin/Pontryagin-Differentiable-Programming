from PDP import PDP
from JinEnv import JinEnv
import numpy as np
import time
import scipy.io as sio

# --------------------------- load environment ----------------------------------------
pendulum = JinEnv.SinglePendulum()
pendulum.initDyn()
pendulum.initCost()

# --------------------------- create PDP SysID object ----------------------------------------
dt = 0.05
pendulumid = PDP.SysID()
pendulumid.setAuxvarVariable(pendulum.dyn_auxvar)
pendulumid.setStateVariable(pendulum.X)
pendulumid.setControlVariable(pendulum.U)
dyn = pendulum.X + dt * pendulum.f
pendulumid.setDyn(dyn)

# --------------------------- load the data ----------------------------------------
load_data = sio.loadmat('./data/pendulum_iodata.mat')
data = load_data['pendulum_iodata'][0, 0]
true_parameter = data['true_parameter']
n_batch = len(data['batch_inputs'])
batch_inputs = []
batch_states = []
for i in range(n_batch):
    batch_inputs += [data['batch_inputs'][i]]
    batch_states += [data['batch_states'][i]]

# --------------------------- learn the dynamics ----------------------------------------
for j in range(10):
    start_time = time.time()
    # learning rate
    lr = 1e-5
    # initialize
    loss_trace, parameter_trace = [], []
    sigma = 1.0
    initial_parameter = np.array(true_parameter) + sigma * np.random.rand(len(true_parameter)) - sigma / 2
    current_parameter = initial_parameter
    for k in range(int(2e4)):
        # one iteration of PDP
        loss, dp = pendulumid.step(batch_inputs, batch_states, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter]
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss)

    # save
    save_data = {'trail_no': j,
                 'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'learning_rate': lr,
                 'time_passed': time.time() - start_time}
    sio.savemat('./data/PDP_SysID_results_trial_' + str(j) + '.mat', {'results': save_data})
