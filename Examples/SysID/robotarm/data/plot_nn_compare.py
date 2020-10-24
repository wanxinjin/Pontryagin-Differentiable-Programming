import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


params = {'axes.labelsize': 50,
          'axes.titlesize': 30,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize':30}
plt.rcParams.update(params)

fig = plt.figure(figsize=(11, 9))

h = [Size.Fixed(2.0), Size.Fixed(8.3)]
v = [Size.Fixed(1.2), Size.Fixed(7.5)]

divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

fig.add_axes(ax)

# ax.set_yscale('log')
ax.set_xlim(-200,10000)
ax.set_ylim(bottom=1e1,top=1e3)
ax.set_xlabel('Iteration')
ax.set_ylabel('SysID Loss')
ax.tick_params(axis='both', which='major')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([1,1,10,6])


# load adam results
adam_loss_list = []
for i in range(5):
    load = sio.loadmat('./Adam_NN_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    adam_loss_list += [loss_trace]

# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('./PDP_SysID_Neural_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]


# plot kkt results
for adam_loss, pdp_loss in zip(adam_loss_list, pdp_loss_list):
    ax.plot(pdp_loss, marker='o', markevery=1000, color = [0.6350, 0.0780, 0.1840], linewidth=4, markersize=10)
    ax.plot(adam_loss, marker='v', markevery=1000, color=[0.4660, 0.6740, 0.1880], linewidth=4, markersize=10)
# show legend
line_pdp,=ax.plot(pdp_loss_list[0], marker='o', markevery=1000, color = [0.6350, 0.0780, 0.1840], linewidth=4, markersize=10)
line_adam,=ax.plot(adam_loss_list[0][1:], marker='v', markevery=1000, color=[0.4660, 0.6740, 0.1880], linewidth=4, markersize=10)
ax.legend([line_pdp,line_adam],['PDP','Pytorch Adam'],facecolor='white',framealpha=0.5)
plt.show()
