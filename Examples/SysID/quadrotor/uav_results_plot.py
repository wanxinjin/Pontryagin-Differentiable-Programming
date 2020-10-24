import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes

# load dmd results
dmd_loss_list = []
for i in range(5):
    load = sio.loadmat('data/DMD_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    dmd_loss_list += [loss_trace]

# load pdp results
pdp_loss_list = []
for i in range(5,10):
    load = sio.loadmat('data/PDP_SysID_results_trial_' + str(i))
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
fig = plt.figure( figsize=(28, 7))
ax = fig.subplots(1,5)

k=0
for dmd_loss, pdp_loss, nn_loss in zip(dmd_loss_list, pdp_loss_list, nn_loss_list):
    ax[k].set_yscale('log')
    ax[k].set_xlim(-500, 10000)
    ax[k].set_xlabel('Iteration')
    ax[k].tick_params(axis='both', which='major')
    ax[k].set_facecolor('#E6E6E6')
    ax[k].grid()
    line_dmd,=ax[k].plot(dmd_loss[0:], marker='v', markevery=1000, color=[0.9290, 0.6940, 0.1250], linewidth=4, markersize=10)
    line_nn,=ax[k].plot(nn_loss[0::3], marker='P', markevery=1000, color=[0.4660, 0.6740, 0.1880], linewidth=4,markersize=10)
    line_pdp,=ax[k].plot(pdp_loss, marker='o', markevery=1000, color = [0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
    ax[k].set_title('#Trial:'+str(k+1))
    ax[k].legend([line_pdp,line_dmd,line_nn],['PDP','DMDc','NN dyn'],facecolor='white',framealpha=0.5, loc='lower left')
    k+=1
ax[0].set_ylabel('UAV SysID loss')
plt.subplots_adjust(wspace=0.4, left=0.08, right=0.975)
plt.show()
