import numpy as np
import matplotlib.pyplot as plt


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7*0.5))

runs = 30
steps = 11

# results directory
DIR = "empirical_analysis/results"

###################
## PLOT 1: Noise ##
###################

dynamic_dim = 5

directories = [f"{DIR}/hartmann_ei/", f"{DIR}/hartmann_noisy_0.025_ei/", f"{DIR}/hartmann_noisy_0.05_ei/", f"{DIR}/hartmann_noisy_0.1_ei/"]
filenames = ["hartmann_ei", "hartmann_noisy_0.025_ei", "hartmann_noisy_0.05_ei", "hartmann_noisy_0.1_ei"]
labels = [f"$\sigma = 0.000$",f"$\sigma = 0.025$", f"$\sigma = 0.050$", f"$\sigma = 0.100$"]
n = len(labels)

all_maes = np.zeros((runs*n, steps))
all_axis_ranges = np.zeros(runs*n)
for i in range(len(labels)):

    axis_range = np.zeros(runs)
    maes = np.zeros((runs, steps))
    for run in range(runs):
        data = np.loadtxt(f"{directories[i]+filenames[i]}_run{run+1}.csv", skiprows=1, delimiter=",")
        lower = np.min(data[:, dynamic_dim])
        upper = np.max(data[:, dynamic_dim])
        axis_range[run] = upper - lower
        
        mae = np.loadtxt(f"{directories[i]+filenames[i]}_mae_run{run+1}.csv", skiprows=1, delimiter=",")
        maes[run, :] = mae[:, -1]
        
    all_maes[i*runs:(i+1)*runs, :] = maes
    all_axis_ranges[i*runs:(i+1)*runs] = axis_range
    axs[0, 0].scatter(axis_range, maes[:, -1], label=labels[i])

# compute trendline
x = np.polyfit(all_axis_ranges, all_maes[:, -1], 1)
trend = np.poly1d(x)

# plot
axs[0, 0].plot(all_axis_ranges, trend(all_axis_ranges), color="black", label="Trend")
axs[0, 0].set_xlabel("Searched space")
axs[0, 0].set_ylabel("Mean absolute percentage error")
axs[0, 0].set_xlim(0, 1)
axs[0, 0].set_ylim(0, )
axs[0, 0].legend(loc="upper left")


#########################################
## PLOT 2: Multiple dynamic parameters ##
#########################################

dynamic_dim = [[5], [3, 5], [0, 3, 5]]
runs = 30

directories = [f"{DIR}/hartmann_ei/", f"{DIR}/hartmann_multi_2_ei/", f"{DIR}/hartmann_multi_3_ei/"]
filenames = ["hartmann_ei", "hartmann_multi_2_ei","hartmann_multi_3_ei"]
labels = [f"$n_d = 1$", f"$n_d = 2$", f"$n_d = 3$"]
n = len(labels)

all_maes = np.zeros((runs*n, steps))
all_axis_ranges = np.zeros(runs*n)
for i in range(len(labels)):

    axis_range = np.zeros(runs)
    maes = np.zeros((runs, steps))
    for run in range(runs):
        data = np.loadtxt(f"{directories[i]+filenames[i]}_run{run+1}.csv", skiprows=1, delimiter=",")

        lower = np.min(data[:, dynamic_dim[i][0]])
        upper = np.max(data[:, dynamic_dim[i][0]])
        axis_range_1 = upper - lower

        if len(dynamic_dim[i]) >= 3:
            lower = np.min(data[:, dynamic_dim[i][1]])
            upper = np.max(data[:, dynamic_dim[i][1]])
            axis_range_2 = upper - lower
        else:
            axis_range_2 = 1

        if len(dynamic_dim[i]) == 3:
            lower = np.min(data[:, dynamic_dim[i][2]])
            upper = np.max(data[:, dynamic_dim[i][2]])
            axis_range_3 = upper - lower
        else:
            axis_range_3 = 1
        
        axis_range[run] = axis_range_1 * axis_range_2 * axis_range_3
        
        mae = np.loadtxt(f"{directories[i]+filenames[i]}_mae_run{run+1}.csv", skiprows=1, delimiter=",")
        maes[run, :] = mae[:, -1]
        
    all_maes[i*runs:(i+1)*runs, :] = maes
    all_axis_ranges[i*runs:(i+1)*runs] = axis_range
    axs[0, 1].scatter(axis_range, maes[:, -1], label=labels[i])

# compute trendline
x = np.polyfit(all_axis_ranges, all_maes[:, -1], 1)
trend = np.poly1d(x)

# plot
axs[0, 1].plot(all_axis_ranges, trend(all_axis_ranges), color="black", label="Trend")
axs[0, 1].set_xlabel("Searched space")
axs[0, 1].set_ylabel("Mean absolute percentage error")
axs[0, 1].set_xlim(0, 1)
axs[0, 1].set_ylim(0, )
axs[0, 1].legend(loc="upper right")


############################
## PLOT 3: Rate of change ##
############################

dynamic_dim = 5
runs = 30

directories = [f"{DIR}/hartmann_ei/", f"{DIR}/hartmann_stepsize_0.1_ei/", f"{DIR}/hartmann_stepsize_0.25_ei/", f"{DIR}/hartmann_stepsize_0.5_ei/", f"{DIR}/hartmann_stepsize_1.0_ei/"]
filenames = ["hartmann_ei", "hartmann_stepsize_0.1_ei", "hartmann_stepsize_0.25_ei", "hartmann_stepsize_0.5_ei", "hartmann_stepsize_1.0_ei"]
labels = [f"$a = 0.05$", f"$a = 0.10$", f"$a = 0.25$", f"$a = 0.50$", f"$a = 1.00$"]
n = len(labels)

all_maes = np.zeros((runs*n, steps))
all_axis_ranges = np.zeros(runs*n)
for i in range(len(labels)):

    axis_range = np.zeros(runs)
    maes = np.zeros((runs, steps))
    for run in range(runs):
        data = np.loadtxt(f"{directories[i]+filenames[i]}_run{run+1}.csv", skiprows=1, delimiter=",")
        lower = np.min(data[:, dynamic_dim])
        upper = np.max(data[:, dynamic_dim])
        axis_range[run] = upper - lower
        
        mae = np.loadtxt(f"{directories[i]+filenames[i]}_mae_run{run+1}.csv", skiprows=1, delimiter=",")
        maes[run, :] = mae[:, -1]
        
    all_maes[i*runs:(i+1)*runs, :] = maes
    all_axis_ranges[i*runs:(i+1)*runs] = axis_range
    axs[1, 0].scatter(axis_range, maes[:, -1], label=labels[i])

# compute trendline
x = np.polyfit(all_axis_ranges, all_maes[:, -1], 1)
trend = np.poly1d(x)

# plot
axs[1, 0].plot(all_axis_ranges, trend(all_axis_ranges), color="black", label="Trend")
axs[1, 0].set_xlabel("Searched space")
axs[1, 0].set_ylabel("Mean absolute precentage error")
axs[1, 0].set_xlim(0, 1)
axs[1, 0].set_ylim(0, )
axs[1, 0].legend(loc="upper left")


#######################################
## PLOT 4: Variability of parameters ##
#######################################

dynamic_dim = [2, 5, 0]
runs = 30

directories = [f"{DIR}/hartmann_variability_low_ei/", f"{DIR}/hartmann_ei/", f"{DIR}/hartmann_variability_high_ei/"]
filenames = ["hartmann_variability_low_ei", "hartmann_ei", "hartmann_variability_high_ei"]
labels = ["Low", "Medium", "High"]
n = len(labels)

all_maes = np.zeros((runs*n, steps))
all_axis_ranges = np.zeros(runs*n)
for i in range(len(labels)):

    axis_range = np.zeros(runs)
    maes = np.zeros((runs, steps))
    for run in range(runs):
        data = np.loadtxt(f"{directories[i]+filenames[i]}_run{run+1}.csv", skiprows=1, delimiter=",")
        lower = np.min(data[:, dynamic_dim[i]])
        upper = np.max(data[:, dynamic_dim[i]])
        axis_range[run] = upper - lower
        
        mae = np.loadtxt(f"{directories[i]+filenames[i]}_mae_run{run+1}.csv", skiprows=1, delimiter=",")
        maes[run, :] = mae[:, -1]
        
    all_maes[i*runs:(i+1)*runs, :] = maes
    all_axis_ranges[i*runs:(i+1)*runs] = axis_range
    axs[1, 1].scatter(axis_range, maes[:, -1], label=labels[i])

# compute trendline
x = np.polyfit(all_axis_ranges, all_maes[:, -1], 1)
trend = np.poly1d(x)

# plot
axs[1, 1].plot(all_axis_ranges, trend(all_axis_ranges), color="black", label="Trend")
axs[1, 1].set_xlabel("Searched space")
axs[1, 1].set_ylabel("Mean absolute percentage error")
axs[1, 1].set_xlim(0, 1)
axs[1, 1].set_ylim(0, )
axs[1, 1].legend(loc="upper right")


#################
## Save figure ##
#################

plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("empirical_analysis/figures/Figure7.png")
plt.savefig("empirical_analysis/figures/Figure7.eps", format="eps")
