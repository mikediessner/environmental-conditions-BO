import numpy as np
import matplotlib.pyplot as plt
from windfarm_simulator import MIN_X, MAX_X, MIN_Y, MAX_Y, SPACING, Bastankhah_PorteAgel_2014
from py_wake.examples.data.ParqueFicticio import ParqueFicticioSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake import HorizontalGrid
from scipy.stats import binned_statistic


# Plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# Directories
LOAD_DIR = "wind_farm_simulator/results/"
FIG_DIR = "wind_farm_simulator/figures/"

# Parameters
N_WT = 4                  # Number of wind turbines
N_EVALS = 200             # Number of simulator evaluations for ENVBO
MIN_WD, MAX_WD = 90, 135  # Lower and upper bounds of wind direction
MIN_WS, MAX_WS = 4, 10    # Lower and upper bounds of wind speed
WS = 6.0                  # Fixed wind speed
WDS = [90, 105, 120, 135] # Wind directions to benchmark


##############
## Figure 8 ##
##############

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7*0.5))
axs = axs.ravel()

# Specify wind farm simulator
site = ParqueFicticioSite()
wind_turbines = V80()
wf_model = Bastankhah_PorteAgel_2014(site, wind_turbines, groundModel=None)

# Load wind turbine positions for wake plots
envbo_results = np.loadtxt(f"{LOAD_DIR}envbo_results_WT{N_WT}.txt", 
                            delimiter=",",
                            skiprows=1)
wt_x = envbo_results[2, :N_WT].reshape(N_WT)
wt_y = envbo_results[2, N_WT:2*N_WT].reshape(N_WT)

# Resolution of contour plots
x = np.linspace(MIN_X, MAX_X, 100)
y = np.linspace(MIN_Y, MAX_Y, 100)
X,Y = np.meshgrid(x, y)

for i, wd in enumerate([0, 120]):

    # Plot first row (terrain)
    lw = site.local_wind(X.flatten(), Y.flatten(), 70, ws=WS, wd=wd)
    Z = lw.WS_ilk.reshape(X.shape)

    c = axs[i].contourf(X, Y, Z, levels=100)
    plt.colorbar(c, label='Local wind speed [m/s]', ax=axs[i], format="{x:.2f}")

    axs[i].set_xlabel("X location")
    axs[i].set_ylabel("Y location")
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_aspect('auto')
    axs[i].set_xlim([MIN_X, MAX_X])
    axs[i].set_xbound(lower=MIN_X, upper=MAX_X)
    axs[i].set_title(f"Wind direction = {wd} degrees, wind speed = {WS} m/s")

    # Plot second row (wake)
    sim_res = wf_model(wt_x, wt_y, h=70, type=0, wd=wd, ws=WS)

    flow_map = sim_res.flow_map(grid=HorizontalGrid(x = x, y = y), wd=wd, ws=WS)
    flow_map.plot_wake_map(ax=axs[i+2])

    axs[i+2].set_xlabel("X location")
    axs[i+2].set_ylabel("Y location")
    axs[i+2].set_xticks([])
    axs[i+2].set_yticks([])
    axs[i+2].set_aspect('auto')
    axs[i+2].set_xlim(MIN_X, MAX_X)
    axs[i+2].set_title(f"Wind direction = {wd} degrees, wind speed = {WS} m/s")
    axs[i+2].get_legend().remove()


# Save plot
plt.tight_layout()
fig.set_rasterized(True)
plt.savefig(f"{FIG_DIR}Figure8.png")
plt.savefig(f"{FIG_DIR}Figure8.eps", format="eps")
plt.clf()


##############
## Figure 9 ##
##############

# Load results
envbo_results = np.loadtxt(f"{LOAD_DIR}envbo_results_WT{N_WT}.txt",
                           delimiter=",",
                           skiprows=1)
slsqp_results = np.loadtxt(f"{LOAD_DIR}slsqp_results_WT{N_WT}.txt",
                           delimiter=",",
                           skiprows=1)
bo_results = np.loadtxt(f"{LOAD_DIR}bo_results_WT{N_WT}.txt",
                        delimiter=",",
                        skiprows=1)

# Site
site = ParqueFicticioSite()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9.0, 11.7*0.6))
axs = axs.ravel()

# Resolution of contour plots
x = np.linspace(MIN_X, MAX_X, 100)
y = np.linspace(MIN_Y, MAX_Y, 100)
X, Y = np.meshgrid(x, y)

for i, wd in enumerate(WDS):

    # Get wind turbine positions and AEP
    envbo_x = envbo_results[i, :N_WT].reshape(N_WT)
    envbo_y = envbo_results[i, N_WT:2*N_WT].reshape(N_WT)
    envbo_aep = envbo_results[i, -1]

    slsqp_x = slsqp_results[i, :N_WT].reshape(N_WT)
    slsqp_y = slsqp_results[i, N_WT:2*N_WT].reshape(N_WT)
    slsqp_aep = slsqp_results[i, -2]

    bo_x = bo_results[i, :N_WT].reshape(N_WT)
    bo_y = bo_results[i, N_WT:2*N_WT].reshape(N_WT)
    bo_aep = bo_results[i, -2]

    # Plot local wind speed
    lw = site.local_wind(X.flatten(), Y.flatten(), h=70, ws=WS, wd=wd)
    Z = lw.WS_ilk.reshape(X.shape)
    c = axs[i].contourf(X, Y, Z, levels=100)
    plt.colorbar(c, label='Local wind speed [m/s]', ax=axs[i], format="{x:.2f}")

    # Plot wind turbines
    axs[i].plot(envbo_x, envbo_y, label="ENVBO", c="red", linestyle="", marker="x")
    axs[i].plot(slsqp_x, slsqp_y, label="SLSQP", c="black", linestyle="", marker="x")
    axs[i].plot(bo_x, bo_y, label="BO", c="white", linestyle="", marker="x")

    # Plot constraints
    for k in range(N_WT):
        axs[i].add_patch(plt.Circle((envbo_x[k], envbo_y[k]), SPACING, ec="red", fill=False, linestyle="--"))
        axs[i].add_patch(plt.Circle((slsqp_x[k], slsqp_y[k]), SPACING, ec="black", fill=False, linestyle="--"))
        axs[i].add_patch(plt.Circle((bo_x[k], bo_y[k]), SPACING, ec="white", fill=False, linestyle="--"))

    axs[i].set_xlim(MIN_X, MAX_X)
    axs[i].set_ylim(MIN_Y, MAX_Y)
    if i == 0: axs[i].legend()
    axs[i].set_xlabel("X location")
    axs[i].set_ylabel("Y location")
    axs[i].set_title(f"Wind direction {wd} degrees\nENVBO: {float(envbo_aep):.2f} GWh vs SLSQP: {float(slsqp_aep):.2f} GWh vs BO: {float(bo_aep):.2f}")
    axs[i].set_xticks([])
    axs[i].set_yticks([])

# Save plot
plt.tight_layout()
fig.set_rasterized(True)
plt.savefig(f"{FIG_DIR}Figure9.png")
plt.savefig(f"{FIG_DIR}Figure9.eps", format="eps")
plt.clf()


###############
## Figure 10 ##
###############

# Load results
envbo_results = np.loadtxt(f"{LOAD_DIR}envbo_long_results_WT{N_WT}.txt",
                           delimiter=",",
                           skiprows=1)
slsqp_results = np.loadtxt(f"{LOAD_DIR}slsqp_results_WT{N_WT}.txt",
                           delimiter=",",
                           skiprows=1)
bo_results = np.loadtxt(f"{LOAD_DIR}bo_results_WT{N_WT}.txt",
                        delimiter=",",
                        skiprows=1)
envbo_experiment = np.loadtxt(f"{LOAD_DIR}envbo_experiment_WT{N_WT}.txt",
                              delimiter=",",
                              skiprows=1)

# Extract wind speeds for histogram
envbo_aep = envbo_results[:, -1].reshape(-1)
envbo_wds = envbo_results[:, -3].reshape(-1)
envbo_random_walk = envbo_experiment[:, -3].reshape(-1)

slsqp_aep = slsqp_results[:, -2].reshape(-1)
slsqp_evals = slsqp_results[:, -1].reshape(-1)
slsqp_evals = np.repeat(WDS, slsqp_evals.astype(int))

bo_aep = bo_results[:, -2].reshape(-1)
bo_evals = bo_results[:, -1].reshape(-1)
bo_evals = np.repeat(WDS, bo_evals.astype(int))

# Compute counts for wind speed bins
envbo_bins = binned_statistic(envbo_random_walk, envbo_random_walk, bins=20)[2]
unique, counts = np.unique(envbo_bins, return_counts=True)
print(dict(zip(unique, counts)))
print(slsqp_results[:, -1].reshape(-1))
print(bo_results[:, -1].reshape(-1))

fig, ax = plt.subplots(nrows=2, ncols=1)

# AEP vs wind direction plot
ax[0].scatter(envbo_wds, envbo_aep, c="tab:blue")
ax[0].scatter(WDS, slsqp_aep, c="tab:orange")
ax[0].scatter(WDS, bo_aep, c="tab:green")
ax[0].set_ylabel("AEP [in GWh]")
ax[0].set_ylim(0, 6)

# Number of evaluations vs wind direction plot
ax[1].hist(envbo_random_walk, bins=20, color="tab:blue", label="ENVBO", zorder=2)
ax[1].hist(slsqp_evals, bins=20, color="tab:orange", label="SLSQP", zorder=0)
ax[1].hist(bo_evals, bins=20, color="tab:green", label="BO", zorder=1)
ax[1].set_xlabel("Wind direction [in degrees]")
ax[1].set_ylabel("Evaluations")
ax[1].legend(ncols=3)
ax[1].set_ylim(0, 400)

# Save plot
plt.tight_layout()
fig.set_rasterized(True)
plt.savefig(f"{FIG_DIR}Figure10.png")
plt.savefig(f"{FIG_DIR}Figure10.eps", format="eps")
plt.clf()
