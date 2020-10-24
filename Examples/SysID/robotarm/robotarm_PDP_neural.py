from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import numpy as np
import time
import scipy.io as sio
import torch
from torch import nn


# --------------------------- establish a dynamics neural network (difference equation) -------------------------
arm = JinEnv.RobotArm()
arm.initDyn(g=0)

dyn_net_para=[]
node=2
dyn_net=vertcat(arm.X, arm.U)
M1=SX.sym('M1', node*(arm.X.numel()+arm.U.numel()), arm.X.numel()+arm.U.numel())
b1=SX.sym('b1', node*(arm.X.numel()+arm.U.numel()))
dyn_net=tanh(mtimes(M1, dyn_net) + b1)
dyn_net_para+=[M1.reshape((-1, 1))]
dyn_net_para+=[b1.reshape((-1, 1))]
M2=SX.sym('M2', arm.X.numel(), node*(arm.X.numel()+arm.U.numel()))
b2=SX.sym('b2',arm.X.numel())
dyn_net=(mtimes(M2, dyn_net) + b2)
dyn_net_para+=[M2.reshape((-1, 1))]
dyn_net_para+=[b2.reshape((-1, 1))]
dyn_net_para=vcat(dyn_net_para)

# --------------------------- create PDP SysID object ----------------------------------------
id = PDP.SysID()
id.setAuxvarVariable(dyn_net_para)
id.setStateVariable(arm.X)
id.setControlVariable(arm.U)
id.setDyn(dyn_net)


# --------------------------- load the data ----------------------------------------
load_data = sio.loadmat('data/robotarm_iodata.mat')
data = load_data['robotarm_iodata'][0, 0]
true_parameter = data['true_parameter']
n_batch = len(data['batch_inputs'])
batch_inputs = []
batch_states = []
# just using one batch of data
for i in range(1):
    batch_inputs += [data['batch_inputs'][i]]
    batch_states += [data['batch_states'][i]]

# --------------------------- load the data ----------------------------------------
for j in range(10):
    start_time = time.time()
    # learning rate
    lr = 1e-5
    # initialize
    loss_trace, parameter_trace = [], []
    current_parameter = np.random.randn(id.n_auxvar)
    for k in range(int(1e4)):
        # one iteration of PDP
        loss, dp = id.step(batch_inputs, batch_states, current_parameter)
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
    sio.savemat('./data/PDP_SysID_Neural_results_trial_' + str(j) + '.mat', {'results': save_data})