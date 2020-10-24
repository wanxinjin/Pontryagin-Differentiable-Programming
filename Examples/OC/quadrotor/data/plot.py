import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from matplotlib.ticker import ScalarFormatter


params = {'axes.labelsize': 50,
          'axes.titlesize': 30,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize':30}
plt.rcParams.update(params)

fig = plt.figure(figsize=(11, 9))

h = [Size.Fixed(1.8), Size.Fixed(8.5)]
v = [Size.Fixed(1.2), Size.Fixed(7.3)]

divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
# the width and height of the rectangle is ignored.
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
ax.set_xlim(-10,150)
ax.set_xlabel('Iteration')
ax.set_ylabel('Control loss')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([1,1,10,6])


# load ilqr results
ilqr_loss_list = []
for i in range(5):
    load = sio.loadmat('./iLQR_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    ilqr_loss_list += [loss_trace]

optimal_cost=sio.loadmat('./iLQR_results_trial_0')['results']['solved_solution'][0,0]['cost'][0,0].item()
print(optimal_cost)

# load pdp N=5 results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('./PDP_OC_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]

# load pdp N=25 results
rm_loss_list = []
for i in range(5):
    load = sio.loadmat('./PDP_Recmat_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    rm_loss_list += [loss_trace]


# plot  results
for ilqr_loss, pdp_loss, rm_loss in zip(ilqr_loss_list, pdp_loss_list, rm_loss_list):
    ax.plot(ilqr_loss[1::1], marker='v', markevery=10, color=[0.9290, 0.6940, 0.1250], linewidth=4,markersize=10)
    ax.plot(rm_loss, marker='P', markevery=10, color=	'#7E2F8E', linewidth=4,markersize=10)
    ax.plot(pdp_loss, marker='o', markevery=10, color = [0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)

# show legend
line_pdp,=ax.plot(pdp_loss_list[0], marker='o', markevery=10, color = [0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
line_ilqr,=ax.plot(ilqr_loss_list[0][1::1], marker='v', markevery=10, color=[0.9290, 0.6940, 0.1250], linewidth=4,markersize=10)
line_rm,=ax.plot(rm_loss_list[0], marker='P', markevery=10, color=	'#7E2F8E', linewidth=4,markersize=10)
line_gt,=ax.plot(optimal_cost*np.ones(30000), color='#0072BD', linewidth=4,linestyle='dashed', alpha=0.7)
ax.legend([line_ilqr,line_pdp,line_rm, line_gt],['iLQR','PDP with poly policy (N=5)','PDP with poly policy (N=35)','solved by OC solver'],facecolor='white',framealpha=0.5)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
plt.show()
