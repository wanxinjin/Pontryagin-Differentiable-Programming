import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes

# load inverse kkt results
kkt_loss_list = []
for i in range(5):
    load = sio.loadmat('data/KKT_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    index = np.argwhere(loss_trace > 1000)
    loss_trace[index] = loss_trace[
        index - 1]  # remove the spikes inside the data (only for kkt results, because it is too bad)
    kkt_loss_list += [loss_trace]

# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('data/PDP_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]

# load nn results
nn_loss_list = []
for i in range(5):
    load = sio.loadmat('data/NN_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    nn_loss_list += [loss_trace]


# Yes
params = {'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize':20,
          'ytick.labelsize':20,
          'legend.fontsize':16}
plt.rcParams.update(params)



# plot
fig = plt.figure( figsize=(28, 8))
ax = fig.subplots(1,5)

k=0
for kkt_loss, pdp_loss, nn_loss in zip(kkt_loss_list, pdp_loss_list, nn_loss_list):
    ax[k].set_yscale('log')
    ax[k].set_xlim(-500, 10000)

    ax[k].set_xlabel('Iteration')
    ax[k].tick_params(axis='both', which='major')
    ax[k].set_facecolor('#E6E6E6')
    ax[k].grid()
    line_kkt,=ax[k].plot(kkt_loss[0:], marker='v', markevery=1000, color=[0.9290, 0.6940, 0.1250], linewidth=4,markersize=10)
    line_nn,=ax[k].plot(nn_loss[0::3], marker='P', markevery=1000, color=[0.4660, 0.6740, 0.1880], linewidth=4,markersize=10)
    line_pdp,=ax[k].plot(pdp_loss[1:], marker='o', markevery=1000, color = [0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
    ax[k].set_title('Trial #'+str(k+1))
    ax[k].legend([line_pdp,line_kkt,line_nn],['PDP','Inv. KKT','NN policy'],facecolor='white',framealpha=0.5)
    k+=1
ax[0].set_ylabel('UAV imitation loss')
plt.subplots_adjust(wspace=0.35, left=0.08, right=0.975)
plt.show()
