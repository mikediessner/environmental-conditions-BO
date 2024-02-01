import numpy as np
import torch
import matplotlib.pyplot as plt
from nubo.test_functions import Levy


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# function
levy = Levy(2, minimise=True)

n = 100
x = np.linspace(-7.5, 7.5, n)
y = np.linspace(-10., 10., n)

X, Y = np.meshgrid(x, y)
X, Y = np.ravel(X).reshape(-1, 1), np.ravel(Y).reshape(-1, 1)
XY = np.hstack([X, Y])
Z = levy(torch.from_numpy(XY))

# make figure
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X.reshape(n, n), Y.reshape(n, n), Z.reshape(n, n), cmap='viridis')
ax.set_xlabel("Controllable parameter")
ax.set_ylabel("Uncontrollable parameter")
ax.set_zlabel("Output")
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-10., 10.)
ax.set_xticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5])
ax.set_yticks([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])


#################
## Save figure ##
#################

plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("empirical_analysis/figures/Figure3.png")
plt.savefig("empirical_analysis/figures/Figure3.eps", format="eps")
plt.clf()
