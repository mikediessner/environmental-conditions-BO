import matplotlib.pyplot as plt
from helper import give_scores, aggregate_mape


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8.3, 11.7*0.25))

# set up experiment
runs = 30          # number of replicaions
evals = 100        # number of evaluations
eval_stepsize = 10 # increments of predictions

# results directory
DIR = "empirical-analysis/results"


##################
## PLOT 1: Levy ##
##################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/levy_ucb4/levy_ucb4")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/levy_ucb8/levy_ucb8")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/levy_ucb16/levy_ucb16")

# compute average and standard deviation of mean absolute error over all runs
mapes_bm = aggregate_mape(runs, f"{DIR}/levy_benchmark/levy_benchmark")
mapes_ucb4 = aggregate_mape(runs, f"{DIR}/levy_ucb4/levy_ucb4")
mapes_ucb8 = aggregate_mape(runs, f"{DIR}/levy_ucb8/levy_ucb8")
mapes_ucb16 = aggregate_mape(runs, f"{DIR}/levy_ucb16/levy_ucb16")

# plot
axs[0].plot(mapes_bm[:, 0], mapes_bm[:, 1], c="black", label="Benchmark", zorder=7)

axs[0].plot(mapes_ucb4[:, 0], mapes_ucb4[:, 1], c="tab:blue", label=r"UCB $\beta$ = 4", zorder=6)
axs[0].fill_between(mapes_ucb4[:, 0], mapes_ucb4[:, 1]-1.96*mapes_ucb4[:, 2], mapes_ucb4[:, 1]+1.96*mapes_ucb4[:, 2], color="tab:blue", alpha=0.25, zorder=3)

axs[0].plot(mapes_ucb8[:, 0], mapes_ucb8[:, 1], c="tab:green", label=r"UCB $\beta$ = 8", zorder=5)
axs[0].fill_between(mapes_ucb8[:, 0], mapes_ucb8[:, 1]-1.96*mapes_ucb8[:, 2], mapes_ucb8[:, 1]+1.96*mapes_ucb8[:, 2], color="tab:green", alpha=0.25, zorder=2)

axs[0].plot(mapes_ucb16[:, 0], mapes_ucb16[:, 1], c="tab:orange", label=r"UCB $\beta$ = 16", zorder=4)
axs[0].fill_between(mapes_ucb16[:, 0], mapes_ucb16[:, 1]-1.96*mapes_ucb16[:, 2], mapes_ucb16[:, 1]+1.96*mapes_ucb16[:, 2], color="tab:orange", alpha=0.25, zorder=1)

# set axis limits
axs[0].set_xlim(0, mapes_bm[-1, 0])
axs[0].set_ylim(0, )

# set axis labels
axs[0].set_xlabel("Evaluations")
axs[0].set_ylabel("Mean absolute percentage error")

# make legend
axs[0].legend()


######################
## PLOT 2: Hartmann ##
######################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_ucb4/hartmann_ucb4")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_ucb8/hartmann_ucb8")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_ucb16/hartmann_ucb16")

# compute average and standard deviation of mean absolute error over all runs
mapes_bm = aggregate_mape(runs, f"{DIR}/hartmann_benchmark/hartmann_benchmark")
mapes_ucb4 = aggregate_mape(runs, f"{DIR}/hartmann_ucb4/hartmann_ucb4")
mapes_ucb8 = aggregate_mape(runs, f"{DIR}/hartmann_ucb8/hartmann_ucb8")
mapes_ucb16 = aggregate_mape(runs, f"{DIR}/hartmann_ucb16/hartmann_ucb16")

# plot
axs[1].plot(mapes_bm[:, 0], mapes_bm[:, 1], c="black", label="Benchmark", zorder=7)

axs[1].plot(mapes_ucb4[:, 0], mapes_ucb4[:, 1], c="tab:blue", label=r"UCB $\beta$ = 4", zorder=6)
axs[1].fill_between(mapes_ucb4[:, 0], mapes_ucb4[:, 1]-1.96*mapes_ucb4[:, 2], mapes_ucb4[:, 1]+1.96*mapes_ucb4[:, 2], color="tab:blue", alpha=0.25, zorder=3)

axs[1].plot(mapes_ucb8[:, 0], mapes_ucb8[:, 1], c="tab:green", label=r"UCB $\beta$ = 8", zorder=5)
axs[1].fill_between(mapes_ucb8[:, 0], mapes_ucb8[:, 1]-1.96*mapes_ucb8[:, 2], mapes_ucb8[:, 1]+1.96*mapes_ucb8[:, 2], color="tab:green", alpha=0.25, zorder=2)

axs[1].plot(mapes_ucb16[:, 0], mapes_ucb16[:, 1], c="tab:orange", label=r"UCB $\beta$ = 16", zorder=4)
axs[1].fill_between(mapes_ucb16[:, 0], mapes_ucb16[:, 1]-1.96*mapes_ucb16[:, 2], mapes_ucb16[:, 1]+1.96*mapes_ucb16[:, 2], color="tab:orange", alpha=0.25, zorder=1)

# set axis limits
axs[1].set_xlim(0, mapes_bm[-1, 0])
axs[1].set_ylim(0, )

# set axis labels
axs[1].set_xlabel("Evaluations")
axs[1].set_ylabel("Mean absolute percentage error")

# make legend
axs[1].legend()


#################
## Save figure ##
#################

plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("empirical-analysis/figures/Figure5.png")
plt.savefig("empirical-analysis/figures/Figure5.eps", format="eps")
