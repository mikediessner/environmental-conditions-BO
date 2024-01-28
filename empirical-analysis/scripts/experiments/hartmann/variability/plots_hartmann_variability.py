from source.plot import give_scores, aggregate_mae
import matplotlib.pyplot as plt


# set up experiment
runs = 30          # number of replicaions
evals = 100        # number of evaluations
eval_stepsize = 10 # increments of predictions

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/variability_low/hartmann_variability_low")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/variability_high/hartmann_variability_high")

# compute average and standard deviation of mean absolute error over all runs
low_maes = aggregate_mae(runs, "data/hartmann/variability_low/hartmann_variability_low")
high_maes = aggregate_mae(runs, "data/hartmann/variability_high/hartmann_variability_high")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(low_maes[:, 0], low_maes[:, 1], label="Low")
plt.fill_between(low_maes[:, 0], low_maes[:, 1]-1.96*low_maes[:, 2], low_maes[:, 1]+1.96*low_maes[:, 2], alpha=0.25)
plt.plot(high_maes[:, 0], high_maes[:, 1], label="High")
plt.fill_between(high_maes[:, 0], high_maes[:, 1]-1.96*high_maes[:, 2], high_maes[:, 1]+1.96*high_maes[:, 2], alpha=0.25)

# set axis limits
plt.xlim(0, low_maes[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("plots/hartmann-variability-mae.png", dpi=1000)
plt.savefig("plots/hartmann-variability-mae.eps", format="eps")
plt.clf()
