import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

nX = 10  # number of grid points
domainWidth = 1e6  # meters
dX = domainWidth / nX
timeStep = 100  # years
nYears = 20000  # years
nStep = int(nYears / timeStep)
flowParam = 1e4  # m horizontal / yr
snowFall = 0.5  # m / y
plotlimit = 4000

elevation = np.zeros(nX + 2)  # consider the 'ghost cells' with elevation = 0
flow = np.zeros(nX + 1)  # flows between each grid cell are calculated

fig, ax = plt.subplots()
ax.plot(elevation)
ax.set_ylim([0, plotlimit])
plt.show()

for itime in range(nStep):
    for ix in range(0, nX + 1):
        flow[ix] = (elevation[ix] - elevation[ix + 1]) / dX * flowParam * (elevation[ix] + elevation[ix + 1]) / 2 / dX

    for ix in range(1, nX + 1):
        elevation[ix] = elevation[ix] + (snowFall + flow[ix - 1] - flow[ix]) * timeStep

    print("year:", itime * timeStep)
    ax.clear()
    ax.plot(elevation)
    ax.set_ylim([0, plotlimit])
    clear_output(wait=True)
    display(fig)
    plt.pause(0.01)

ax.clear()
ax.plot(elevation)
ax.set_ylim([0, plotlimit])
plt.show()
