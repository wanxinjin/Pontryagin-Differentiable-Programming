import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from matplotlib.ticker import ScalarFormatter


# load gps results
gps_loss_list = []
for i in range(5):
    load = sio.loadmat('./data/GPS_Neural_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    gps_loss_list += [loss_trace]

# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('./data/PDP_Neural_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]

# plot
params = {'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize':20,
          'ytick.labelsize':20,
          'legend.fontsize':16}
plt.rcParams.update(params)

fig = plt.figure( figsize=(28, 7))
ax = fig.subplots(1,5)

k=0
for gps_loss, pdp_loss in zip(gps_loss_list, pdp_loss_list):
    ax[k].set_xlim(-10, 400)
    ax[k].set_ylim(0,50)
    ax[k].set_xlabel('Iteration')
    ax[k].tick_params(axis='both', which='major')
    ax[k].set_facecolor('#E6E6E6')
    ax[k].grid()
    line_gps,=ax[k].plot(gps_loss[0:], marker='v', markevery=50, color='#77AC30', linewidth=4, markersize=10)
    line_pdp,=ax[k].plot(pdp_loss, marker='o', markevery=50, color = [0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
    ax[k].set_title('Trial#'+str(k+1))
    ax[k].legend([line_pdp, line_gps], ['PDP', 'GPS'], facecolor='white', framealpha=0.5)

    k+=1
ax[0].set_ylabel('Robot arm control loss')
plt.subplots_adjust(wspace=0.35, left=0.08, right=0.975)
plt.show()