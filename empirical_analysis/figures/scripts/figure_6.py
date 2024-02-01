import matplotlib.pyplot as plt
from helper import give_scores, aggregate_mape


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7*0.5))

# set parameters
evals = 100
eval_stepsize = 10
runs = 30
ylim = 4

# results directory
DIR = "empirical_analysis/results"

basic_mapes = aggregate_mape(runs, f"{DIR}/hartmann_ei/hartmann_ei")


###################
## PLOT 1: Noise ##
###################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_noisy_0.025_ei/hartmann_hartmann_noisy_0.025_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_noisy_0.05_ei/hartmann_hartmann_noisy_0.05_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_noisy_0.1_ei/hartmann_hartmann_noisy_0.1_ei")

# compute average and standard deviation of mean absolute error over all runs
low_mapes = aggregate_mape(runs, f"{DIR}/hartmann_noisy_0.025_ei/hartmann_hartmann_noisy_0.025_ei")
medium_mapes = aggregate_mape(runs, f"{DIR}/hartmann_noisy_0.05_ei/hartmann_hartmann_noisy_0.05_ei")
high_mapes = aggregate_mape(runs, f"{DIR}/hartmann_noisy_0.1_ei/hartmann_hartmann_noisy_0.1_ei")

# plot
axs[0, 0].plot(basic_mapes[:, 0], basic_mapes[:, 1], label=f"$\sigma = 0.000$", zorder=8)
axs[0, 0].fill_between(basic_mapes[:, 0], basic_mapes[:, 1]-1.96*basic_mapes[:, 2], basic_mapes[:, 1]+1.96*basic_mapes[:, 2], alpha=0.25, zorder=4)
axs[0, 0].plot(low_mapes[:, 0], low_mapes[:, 1], label=f"$\sigma = 0.025$", zorder=7)
axs[0, 0].fill_between(low_mapes[:, 0], low_mapes[:, 1]-1.96*low_mapes[:, 2], low_mapes[:, 1]+1.96*low_mapes[:, 2], alpha=0.25, zorder=3)
axs[0, 0].plot(medium_mapes[:, 0], medium_mapes[:, 1], label=f"$\sigma = 0.050$", zorder=6)
axs[0, 0].fill_between(medium_mapes[:, 0], medium_mapes[:, 1]-1.96*medium_mapes[:, 2], medium_mapes[:, 1]+1.96*medium_mapes[:, 2], alpha=0.25, zorder=2)
axs[0, 0].plot(high_mapes[:, 0], high_mapes[:, 1], label=f"$\sigma = 0.100$", zorder=5)
axs[0, 0].fill_between(high_mapes[:, 0], high_mapes[:, 1]-1.96*high_mapes[:, 2], high_mapes[:, 1]+1.96*high_mapes[:, 2], alpha=0.25, zorder=1)
axs[0, 0].set_xlim(0, low_mapes[-1, 0])
axs[0, 0].set_ylim(0, ylim)
axs[0, 0].set_xlabel("Evaluations")
axs[0, 0].set_ylabel("Mean absolute percentage error")
axs[0, 0].legend()


#########################################
## PLOT 2: Multiple dynamic parameters ##
#########################################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_multi_2_ei/hartmann_multi_2_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_multi_3_ei/hartmann_multi_3_ei")

# compute average and standard deviation of mean absolute error over all runs
two_mapes = aggregate_mape(runs, f"{DIR}/hartmann_multi_2_ei/hartmann_multi_2_ei")
three_mapes = aggregate_mape(runs, f"{DIR}/hartmann_multi_3_ei/hartmann_multi_3_ei")

# plot
axs[0, 1].plot(basic_mapes[:, 0], basic_mapes[:, 1], label=f"$n_d = 1$", zorder=6)
axs[0, 1].fill_between(basic_mapes[:, 0], basic_mapes[:, 1]-1.96*basic_mapes[:, 2], basic_mapes[:, 1]+1.96*basic_mapes[:, 2], alpha=0.25, zorder=3)
axs[0, 1].plot(two_mapes[:, 0], two_mapes[:, 1], label=f"$n_d = 2$", zorder=5)
axs[0, 1].fill_between(two_mapes[:, 0], two_mapes[:, 1]-1.96*two_mapes[:, 2], two_mapes[:, 1]+1.96*two_mapes[:, 2], alpha=0.25, zorder=2)
axs[0, 1].plot(three_mapes[:, 0], three_mapes[:, 1], label=f"$n_d = 3$", zorder=4)
axs[0, 1].fill_between(three_mapes[:, 0], three_mapes[:, 1]-1.96*three_mapes[:, 2], three_mapes[:, 1]+1.96*three_mapes[:, 2], alpha=0.25, zorder=1)
axs[0, 1].set_xlim(0, two_mapes[-1, 0])
axs[0, 1].set_ylim(0, ylim)
axs[0, 1].set_xlabel("Evaluations")
axs[0, 1].set_ylabel("Mean absolute percentage error")
axs[0, 1].legend()


############################
## PLOT 3: Rate of change ##
############################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_stepsize_0.1_ei/hartmann_stepsize_0.1_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_stepsize_0.25_ei/hartmann_stepsize_0.25_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_stepsize_0.5_ei/hartmann_stepsize_0.5_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_stepsize_1.0_ei/hartmann_stepsize_1.0_ei")

# compute average and standard deviation of mean absolute error over all runs
mapes_010 = aggregate_mape(runs, f"{DIR}/hartmann_stepsize_0.1_ei/hartmann_stepsize_0.1_ei")
mapes_025 = aggregate_mape(runs, f"{DIR}/hartmann_stepsize_0.25_ei/hartmann_stepsize_0.25_ei")
mapes_50 = aggregate_mape(runs, f"{DIR}/hartmann_stepsize_0.5_ei/hartmann_stepsize_0.5_ei")
mapes_100 = aggregate_mape(runs, f"{DIR}/hartmann_stepsize_1.0_ei/hartmann_stepsize_1.0_ei")

# plot
axs[1, 0].plot(basic_mapes[:, 0], basic_mapes[:, 1], label=f"$a = 0.05$", zorder=10)
axs[1, 0].fill_between(basic_mapes[:, 0], basic_mapes[:, 1]-1.96*basic_mapes[:, 2], basic_mapes[:, 1]+1.96*basic_mapes[:, 2], alpha=0.25, zorder=5)
axs[1, 0].plot(mapes_010[:, 0], mapes_010[:, 1], label=f"$a = 0.10$", zorder=9)
axs[1, 0].fill_between(mapes_010[:, 0], mapes_010[:, 1]-1.96*mapes_010[:, 2], mapes_010[:, 1]+1.96*mapes_010[:, 2], alpha=0.25, zorder=4)
axs[1, 0].plot(mapes_025[:, 0], mapes_025[:, 1], label=f"$a = 0.25$", zorder=8)
axs[1, 0].fill_between(mapes_025[:, 0], mapes_025[:, 1]-1.96*mapes_025[:, 2], mapes_025[:, 1]+1.96*mapes_025[:, 2], alpha=0.25, zorder=3)
axs[1, 0].plot(mapes_50[:, 0], mapes_50[:, 1], label=f"$a = 0.50$", zorder=7)
axs[1, 0].fill_between(mapes_50[:, 0], mapes_50[:, 1]-1.96*mapes_50[:, 2], mapes_50[:, 1]+1.96*mapes_50[:, 2], alpha=0.25, zorder=2)
axs[1, 0].plot(mapes_100[:, 0], mapes_100[:, 1], label=f"$a = 1.00$", zorder=6)
axs[1, 0].fill_between(mapes_100[:, 0], mapes_100[:, 1]-1.96*mapes_100[:, 2], mapes_100[:, 1]+1.96*mapes_100[:, 2], alpha=0.25, zorder=1)
axs[1, 0].set_xlim(0, basic_mapes[-1, 0])
axs[1, 0].set_ylim(0, ylim)
axs[1, 0].set_xlabel("Evaluations")
axs[1, 0].set_ylabel("Mean absolute percentage error")
axs[1, 0].legend()


#######################################
## PLOT 4: Variability of parameters ##
#######################################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_variability_low_ei/hartmann_variability_low_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_variability_high_ei/hartmann_variability_high_ei")

# compute average and standard deviation of mean absolute error over all runs
low_mapes = aggregate_mape(runs, f"{DIR}/hartmann_variability_low_ei/hartmann_variability_low_ei")
high_mapes = aggregate_mape(runs, f"{DIR}/hartmann_variability_high_ei/hartmann_variability_high_ei")

# plot
axs[1, 1].plot(low_mapes[:, 0], low_mapes[:, 1], label="Low", zorder=6)
axs[1, 1].fill_between(low_mapes[:, 0], low_mapes[:, 1]-1.96*low_mapes[:, 2], low_mapes[:, 1]+1.96*low_mapes[:, 2], alpha=0.25, zorder=3)
axs[1, 1].plot(basic_mapes[:, 0], basic_mapes[:, 1], label="Medium", zorder=5)
axs[1, 1].fill_between(basic_mapes[:, 0], basic_mapes[:, 1]-1.96*basic_mapes[:, 2], basic_mapes[:, 1]+1.96*basic_mapes[:, 2], alpha=0.25, zorder=2)
axs[1, 1].plot(high_mapes[:, 0], high_mapes[:, 1], label="High", zorder=4)
axs[1, 1].fill_between(high_mapes[:, 0], high_mapes[:, 1]-1.96*high_mapes[:, 2], high_mapes[:, 1]+1.96*high_mapes[:, 2], alpha=0.25, zorder=1)
axs[1, 1].set_xlim(0, low_mapes[-1, 0])
axs[1, 1].set_ylim(0, ylim)
axs[1, 1].set_xlabel("Evaluations")
axs[1, 1].set_ylabel("Mean absolute percentage error")
axs[1, 1].legend()


#################
## Save figure ##
#################

plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("empirical_analysis/figures/Figure6.png")
plt.savefig("empirical_analysis/figures/Figure6.eps", format="eps")
