from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import numpy as np
import time
import scipy.io as sio

# --------------------------- load environment ----------------------------------------
arm = JinEnv.RobotArm()
arm.initDyn(g=0)

# --------------------------- create PDP SysID object ----------------------------------------
dt = 0.1
armid = PDP.SysID()
armid.setAuxvarVariable(arm.dyn_auxvar)
armid.setStateVariable(arm.X)
armid.setControlVariable(arm.U)
dyn = arm.X + dt * arm.f
armid.setDyn(dyn)

# --------------------------- load the data ----------------------------------------
load_data = sio.loadmat('./data/robotarm_iodata.mat')
data = load_data['robotarm_iodata'][0, 0]
true_parameter = data['true_parameter']
n_batch = len(data['batch_inputs'])
batch_inputs = []
batch_states = []
for i in range(n_batch):
    batch_inputs += [data['batch_inputs'][i]]
    batch_states += [data['batch_states'][i]]

# --------------------------- load the data ----------------------------------------
for j in range(10):
    start_time = time.time()
    # learning rate
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    sigma = 0.8
    initial_parameter = np.array(true_parameter) + sigma * np.random.rand(len(true_parameter)) - sigma / 2
    current_parameter = initial_parameter
    for k in range(int(1e4)):
        # one iteration of PDP
        loss, dp = armid.step(batch_inputs, batch_states, current_parameter)
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
