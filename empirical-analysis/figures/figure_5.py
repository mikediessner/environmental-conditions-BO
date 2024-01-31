from source.plot import give_scores, aggregate_mae
import matplotlib.pyplot as plt


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8.3, 11.7*0.25))

# set up experiment
runs = 30          # number of replicaions
evals = 100        # number of evaluations
eval_stepsize = 10 # increments of predictions


##################
## PLOT 1: Levy ##
##################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/levy/ucb-4/ucb-4-levy")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/levy/ucb-8/ucb-8-levy")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/levy/ucb-16/ucb-16-levy")

# compute average and standard deviation of mean absolute error over all runs
maes_bm = aggregate_mae(runs, "data/levy/benchmark/benchmark-levy")
maes_ucb4 = aggregate_mae(runs, "data/levy/ucb-4/ucb-4-levy")
maes_ucb8 = aggregate_mae(runs, "data/levy/ucb-8/ucb-8-levy")
maes_ucb16 = aggregate_mae(runs, "data/levy/ucb-16/ucb-16-levy")

# plot
axs[0].plot(maes_bm[:, 0], maes_bm[:, 1], c="black", label="Benchmark", zorder=7)

axs[0].plot(maes_ucb4[:, 0], maes_ucb4[:, 1], c="tab:blue", label=r"UCB $\beta$ = 4", zorder=6)
axs[0].fill_between(maes_ucb4[:, 0], maes_ucb4[:, 1]-1.96*maes_ucb4[:, 2], maes_ucb4[:, 1]+1.96*maes_ucb4[:, 2], color="tab:blue", alpha=0.25, zorder=3)

axs[0].plot(maes_ucb8[:, 0], maes_ucb8[:, 1], c="tab:green", label=r"UCB $\beta$ = 8", zorder=5)
axs[0].fill_between(maes_ucb8[:, 0], maes_ucb8[:, 1]-1.96*maes_ucb8[:, 2], maes_ucb8[:, 1]+1.96*maes_ucb8[:, 2], color="tab:green", alpha=0.25, zorder=2)

axs[0].plot(maes_ucb16[:, 0], maes_ucb16[:, 1], c="tab:orange", label=r"UCB $\beta$ = 16", zorder=4)
axs[0].fill_between(maes_ucb16[:, 0], maes_ucb16[:, 1]-1.96*maes_ucb16[:, 2], maes_ucb16[:, 1]+1.96*maes_ucb16[:, 2], color="tab:orange", alpha=0.25, zorder=1)

# set axis limits
axs[0].set_xlim(0, maes_bm[-1, 0])
axs[0].set_ylim(0, )

# set axis labels
axs[0].set_xlabel("Evaluations")
axs[0].set_ylabel("Mean absolute error")

# make legend
axs[0].legend()


######################
## PLOT 2: Hartmann ##
######################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/ucb-4/ucb-4-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/ucb-8/ucb-8-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/ucb-16/ucb-16-hartmann")

# compute average and standard deviation of mean absolute error over all runs
maes_bm = aggregate_mae(runs, "data/hartmann/benchmark/benchmark-hartmann")
maes_ucb4 = aggregate_mae(runs, "data/hartmann/ucb-4/ucb-4-hartmann")
maes_ucb8 = aggregate_mae(runs, "data/hartmann/ucb-8/ucb-8-hartmann")
maes_ucb16 = aggregate_mae(runs, "data/hartmann/ucb-16/ucb-16-hartmann")

# plot
axs[1].plot(maes_bm[:, 0], maes_bm[:, 1], c="black", label="Benchmark", zorder=7)

axs[1].plot(maes_ucb4[:, 0], maes_ucb4[:, 1], c="tab:blue", label=r"UCB $\beta$ = 4", zorder=6)
axs[1].fill_between(maes_ucb4[:, 0], maes_ucb4[:, 1]-1.96*maes_ucb4[:, 2], maes_ucb4[:, 1]+1.96*maes_ucb4[:, 2], color="tab:blue", alpha=0.25, zorder=3)

axs[1].plot(maes_ucb8[:, 0], maes_ucb8[:, 1], c="tab:green", label=r"UCB $\beta$ = 8", zorder=5)
axs[1].fill_between(maes_ucb8[:, 0], maes_ucb8[:, 1]-1.96*maes_ucb8[:, 2], maes_ucb8[:, 1]+1.96*maes_ucb8[:, 2], color="tab:green", alpha=0.25, zorder=2)

axs[1].plot(maes_ucb16[:, 0], maes_ucb16[:, 1], c="tab:orange", label=r"UCB $\beta$ = 16", zorder=4)
axs[1].fill_between(maes_ucb16[:, 0], maes_ucb16[:, 1]-1.96*maes_ucb16[:, 2], maes_ucb16[:, 1]+1.96*maes_ucb16[:, 2], color="tab:orange", alpha=0.25, zorder=1)

# set axis limits
axs[1].set_xlim(0, maes_bm[-1, 0])
axs[1].set_ylim(0, )

# set axis labels
axs[1].set_xlabel("Evaluations")
axs[1].set_ylabel("Mean absolute error")

# make legend
axs[1].legend()


# save plot
plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("Figure5.png")
plt.savefig("Figure5.eps", format="eps")
