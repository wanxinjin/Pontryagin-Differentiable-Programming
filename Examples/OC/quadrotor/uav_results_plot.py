import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from matplotlib.ticker import ScalarFormatter


# load inverse ilqr results
ilqr_loss_list = []
for i in range(5):
    load = sio.loadmat('./data/iLQR_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    ilqr_loss_list += [loss_trace]
optimal_cost=sio.loadmat('./data/iLQR_results_trial_0')['results']['solved_solution'][0,0]['cost'][0,0].item()


# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('./data/PDP_OC_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]

# load rm results
rm_loss_list = []
for i in range(5):
    load = sio.loadmat('./data/PDP_Recmat_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    rm_loss_list += [loss_trace]


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
for ilqr_loss, pdp_loss, rm_loss in zip(ilqr_loss_list, pdp_loss_list, rm_loss_list):
    ax[k].set_xlim(-10, 150)
    ax[k].set_ylim(1000,45000)
    ax[k].set_xlabel('Iteration')
    ax[k].tick_params(axis='both', which='major')
    ax[k].set_facecolor('#E6E6E6')
    ax[k].grid()
    line_ilqr,=ax[k].plot(ilqr_loss[0::1], marker='v', markevery=20, color=[0.9290, 0.6940, 0.1250], linewidth=4, markersize=10)
    line_rm,=ax[k].plot(rm_loss[0:], marker='P', markevery=20, color='#77AC30', linewidth=4, markersize=10)
    line_pdp,=ax[k].plot(pdp_loss, marker='o', markevery=20, color = [0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
    line_gt, = ax[k].plot(optimal_cost * np.ones(30000), color='#0072BD', linewidth=4, linestyle='dashed', alpha=0.7)
    ax[k].ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    ax[k].set_title('      Trial#'+str(k+1))
    ax[k].legend([line_pdp,  line_rm, line_ilqr, line_gt], ['PDP, N=5', 'PDP, N=35', 'iLQR', 'by OC solver'], facecolor='white', framealpha=0.5)

    k+=1
ax[0].set_ylabel('UAV control loss')
plt.subplots_adjust(wspace=0.35, left=0.08, right=0.975)
plt.show()