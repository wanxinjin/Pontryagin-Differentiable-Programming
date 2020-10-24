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
v = [Size.Fixed(1.2), Size.Fixed(7.5)]

divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
# the width and height of the rectangle is ignored.
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
# ax.set_yscale('log')
ax.set_xlim(-20,400)
ax.set_ylim(130,200)
ax.set_xlabel('Iteration')
ax.set_ylabel('Control loss')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([1,1,10,6])


# load pdp neural results
pdp_neural_loss_list = []
for i in range(5):
    load = sio.loadmat('./PDP_Neural_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_neural_loss_list += [loss_trace]

# load gps neural results
gps_loss_list = []
for i in range(5):
    load = sio.loadmat('./GPS_Neural_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    if i==3:
        loss_trace=loss_trace[100:]
    gps_loss_list += [loss_trace]


# plot  results
for pdp_neural_loss, gps_loss in zip( pdp_neural_loss_list,gps_loss_list):
    ax.plot(pdp_neural_loss, marker='P', markevery=25, color=[0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
    ax.plot(gps_loss, marker='o', markevery=25, color = '#77AC30', linewidth=4,markersize=10,)
# show legend
line_gps,=ax.plot(gps_loss_list[0], marker='o', markevery=25, color = '#77AC30', linewidth=4,markersize=10)
line_pdp_neural,=ax.plot(pdp_neural_loss_list[0], marker='P', markevery=200, color=[0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
ax.legend([line_pdp_neural, line_gps],['PDP with neural policy', 'GPS with neural policy'],facecolor='white',framealpha=0.5)
plt.show()
