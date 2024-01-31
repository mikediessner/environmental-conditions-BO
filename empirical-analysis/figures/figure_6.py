import matplotlib.pyplot as plt
from source.plot import give_scores, aggregate_mae


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7*0.5))

# set parameters
evals = 100
eval_stepsize = 10
runs = 30
ylim = 4

basic_maes = aggregate_mae(runs, "data/hartmann/ei/ei-hartmann")


###################
## PLOT 1: Noise ##
###################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/noisy_0.025/hartmann_noisy_0.025")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/noisy_0.05/hartmann_noisy_0.05")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/noisy_0.1/hartmann_noisy_0.1")

# compute average and standard deviation of mean absolute error over all runs
low_maes = aggregate_mae(runs, "data/hartmann/analysis/noisy_0.025/hartmann_noisy_0.025")
medium_maes = aggregate_mae(runs, "data/hartmann/analysis/noisy_0.05/hartmann_noisy_0.05")
high_maes = aggregate_mae(runs, "data/hartmann/analysis/noisy_0.1/hartmann_noisy_0.1")

# plot
axs[0, 0].plot(basic_maes[:, 0], basic_maes[:, 1], label=f"$\sigma = 0.000$", zorder=8)
axs[0, 0].fill_between(basic_maes[:, 0], basic_maes[:, 1]-1.96*basic_maes[:, 2], basic_maes[:, 1]+1.96*basic_maes[:, 2], alpha=0.25, zorder=4)
axs[0, 0].plot(low_maes[:, 0], low_maes[:, 1], label=f"$\sigma = 0.025$", zorder=7)
axs[0, 0].fill_between(low_maes[:, 0], low_maes[:, 1]-1.96*low_maes[:, 2], low_maes[:, 1]+1.96*low_maes[:, 2], alpha=0.25, zorder=3)
axs[0, 0].plot(medium_maes[:, 0], medium_maes[:, 1], label=f"$\sigma = 0.050$", zorder=6)
axs[0, 0].fill_between(medium_maes[:, 0], medium_maes[:, 1]-1.96*medium_maes[:, 2], medium_maes[:, 1]+1.96*medium_maes[:, 2], alpha=0.25, zorder=2)
axs[0, 0].plot(high_maes[:, 0], high_maes[:, 1], label=f"$\sigma = 0.100$", zorder=5)
axs[0, 0].fill_between(high_maes[:, 0], high_maes[:, 1]-1.96*high_maes[:, 2], high_maes[:, 1]+1.96*high_maes[:, 2], alpha=0.25, zorder=1)
axs[0, 0].set_xlim(0, low_maes[-1, 0])
axs[0, 0].set_ylim(0, ylim)
axs[0, 0].set_xlabel("Evaluations")
axs[0, 0].set_ylabel("Mean absolute error")
axs[0, 0].legend()


#########################################
## PLOT 2: Multiple dynamic parameters ##
#########################################

filename = "data/hartmann/multi/hartmann_multi"

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/multi_2/hartmann_multi_2")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/multi_3/hartmann_multi_3")

# compute average and standard deviation of mean absolute error over all runs
two_maes = aggregate_mae(runs, "data/hartmann/analysis/multi_2/hartmann_multi_2")
three_maes = aggregate_mae(runs, "data/hartmann/analysis/multi_3/hartmann_multi_3")

# plot
axs[0, 1].plot(basic_maes[:, 0], basic_maes[:, 1], label=f"$n_d = 1$", zorder=6)
axs[0, 1].fill_between(basic_maes[:, 0], basic_maes[:, 1]-1.96*basic_maes[:, 2], basic_maes[:, 1]+1.96*basic_maes[:, 2], alpha=0.25, zorder=3)
axs[0, 1].plot(two_maes[:, 0], two_maes[:, 1], label=f"$n_d = 2$", zorder=5)
axs[0, 1].fill_between(two_maes[:, 0], two_maes[:, 1]-1.96*two_maes[:, 2], two_maes[:, 1]+1.96*two_maes[:, 2], alpha=0.25, zorder=2)
axs[0, 1].plot(three_maes[:, 0], three_maes[:, 1], label=f"$n_d = 3$", zorder=4)
axs[0, 1].fill_between(three_maes[:, 0], three_maes[:, 1]-1.96*three_maes[:, 2], three_maes[:, 1]+1.96*three_maes[:, 2], alpha=0.25, zorder=1)
axs[0, 1].set_xlim(0, two_maes[-1, 0])
axs[0, 1].set_ylim(0, ylim)
axs[0, 1].set_xlabel("Evaluations")
axs[0, 1].set_ylabel("Mean absolute error")
axs[0, 1].legend()


############################
## PLOT 3: Rate of change ##
############################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/stepsize_0.1/hartmann_stepsize_0.1")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/stepsize_0.25/hartmann_stepsize_0.25")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/stepsize_0.5/hartmann_stepsize_0.5")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/stepsize_1.0/hartmann_stepsize_1.0")

# compute average and standard deviation of mean absolute error over all runs
maes_010 = aggregate_mae(runs, "data/hartmann/analysis/stepsize_0.1/hartmann_stepsize_0.1")
maes_025 = aggregate_mae(runs, "data/hartmann/analysis/stepsize_0.25/hartmann_stepsize_0.25")
maes_50 = aggregate_mae(runs, "data/hartmann/analysis/stepsize_0.5/hartmann_stepsize_0.5")
maes_100 = aggregate_mae(runs, "data/hartmann/analysis/stepsize_1.0/hartmann_stepsize_1.0")

# plot
axs[1, 0].plot(basic_maes[:, 0], basic_maes[:, 1], label=f"$a = 0.05$", zorder=10)
axs[1, 0].fill_between(basic_maes[:, 0], basic_maes[:, 1]-1.96*basic_maes[:, 2], basic_maes[:, 1]+1.96*basic_maes[:, 2], alpha=0.25, zorder=5)
axs[1, 0].plot(maes_010[:, 0], maes_010[:, 1], label=f"$a = 0.10$", zorder=9)
axs[1, 0].fill_between(maes_010[:, 0], maes_010[:, 1]-1.96*maes_010[:, 2], maes_010[:, 1]+1.96*maes_010[:, 2], alpha=0.25, zorder=4)
axs[1, 0].plot(maes_025[:, 0], maes_025[:, 1], label=f"$a = 0.25$", zorder=8)
axs[1, 0].fill_between(maes_025[:, 0], maes_025[:, 1]-1.96*maes_025[:, 2], maes_025[:, 1]+1.96*maes_025[:, 2], alpha=0.25, zorder=3)
axs[1, 0].plot(maes_50[:, 0], maes_50[:, 1], label=f"$a = 0.50$", zorder=7)
axs[1, 0].fill_between(maes_50[:, 0], maes_50[:, 1]-1.96*maes_50[:, 2], maes_50[:, 1]+1.96*maes_50[:, 2], alpha=0.25, zorder=2)
axs[1, 0].plot(maes_100[:, 0], maes_100[:, 1], label=f"$a = 1.00$", zorder=6)
axs[1, 0].fill_between(maes_100[:, 0], maes_100[:, 1]-1.96*maes_100[:, 2], maes_100[:, 1]+1.96*maes_100[:, 2], alpha=0.25, zorder=1)
axs[1, 0].set_xlim(0, basic_maes[-1, 0])
axs[1, 0].set_ylim(0, ylim)
axs[1, 0].set_xlabel("Evaluations")
axs[1, 0].set_ylabel("Mean absolute error")
axs[1, 0].legend()


#######################################
## PLOT 4: Variability of parameters ##
#######################################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/variability_low/hartmann_variability_low")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/analysis/variability_high/hartmann_variability_high")

# compute average and standard deviation of mean absolute error over all runs
low_maes = aggregate_mae(runs, "data/hartmann/analysis/variability_low/hartmann_variability_low")
high_maes = aggregate_mae(runs, "data/hartmann/analysis/variability_high/hartmann_variability_high")

# plot
axs[1, 1].plot(low_maes[:, 0], low_maes[:, 1], label="Low", zorder=6)
axs[1, 1].fill_between(low_maes[:, 0], low_maes[:, 1]-1.96*low_maes[:, 2], low_maes[:, 1]+1.96*low_maes[:, 2], alpha=0.25, zorder=3)
axs[1, 1].plot(basic_maes[:, 0], basic_maes[:, 1], label="Medium", zorder=5)
axs[1, 1].fill_between(basic_maes[:, 0], basic_maes[:, 1]-1.96*basic_maes[:, 2], basic_maes[:, 1]+1.96*basic_maes[:, 2], alpha=0.25, zorder=2)
axs[1, 1].plot(high_maes[:, 0], high_maes[:, 1], label="High", zorder=4)
axs[1, 1].fill_between(high_maes[:, 0], high_maes[:, 1]-1.96*high_maes[:, 2], high_maes[:, 1]+1.96*high_maes[:, 2], alpha=0.25, zorder=1)
axs[1, 1].set_xlim(0, low_maes[-1, 0])
axs[1, 1].set_ylim(0, ylim)
axs[1, 1].set_xlabel("Evaluations")
axs[1, 1].set_ylabel("Mean absolute error")
axs[1, 1].legend()


#################
## Save figure ##
#################

plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("Figure6.png")
plt.savefig("Figure6.eps", format="eps")
