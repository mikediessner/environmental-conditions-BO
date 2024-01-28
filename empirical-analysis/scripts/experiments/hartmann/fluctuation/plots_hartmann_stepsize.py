from source.plot import give_scores, aggregate_mae
import matplotlib.pyplot as plt


# set up experiment
runs = 30          # number of replicaions
evals = 100        # number of evaluations
eval_stepsize = 10 # increments of predictions

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/stepsize_0.1/hartmann_stepsize_0.1")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/stepsize_0.25/hartmann_stepsize_0.25")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/stepsize_0.5/hartmann_stepsize_0.5")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/stepsize_1.0/hartmann_stepsize_1.0")

# compute average and standard deviation of mean absolute error over all runs
maes_005 = aggregate_mae(runs, "data/hartmann/basic/hartmann")
maes_010 = aggregate_mae(runs, "data/hartmann/stepsize_0.1/hartmann_stepsize_0.1")
maes_025 = aggregate_mae(runs, "data/hartmann/stepsize_0.25/hartmann_stepsize_0.25")
maes_50 = aggregate_mae(runs, "data/hartmann/stepsize_0.5/hartmann_stepsize_0.5")
maes_100 = aggregate_mae(runs, "data/hartmann/stepsize_1.0/hartmann_stepsize_1.0")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(maes_005[:, 0], maes_005[:, 1], label="0.05")
plt.fill_between(maes_005[:, 0], maes_005[:, 1]-1.96*maes_005[:, 2], maes_005[:, 1]+1.96*maes_005[:, 2], alpha=0.25)
plt.plot(maes_010[:, 0], maes_010[:, 1], label="0.10")
plt.fill_between(maes_010[:, 0], maes_010[:, 1]-1.96*maes_010[:, 2], maes_010[:, 1]+1.96*maes_010[:, 2], alpha=0.25)
plt.plot(maes_025[:, 0], maes_025[:, 1], label="0.25")
plt.fill_between(maes_025[:, 0], maes_025[:, 1]-1.96*maes_025[:, 2], maes_025[:, 1]+1.96*maes_025[:, 2], alpha=0.25)
plt.plot(maes_50[:, 0], maes_50[:, 1], label="0.50")
plt.fill_between(maes_50[:, 0], maes_50[:, 1]-1.96*maes_50[:, 2], maes_50[:, 1]+1.96*maes_50[:, 2], alpha=0.25)
plt.plot(maes_100[:, 0], maes_100[:, 1], label="1.00")
plt.fill_between(maes_100[:, 0], maes_100[:, 1]-1.96*maes_100[:, 2], maes_100[:, 1]+1.96*maes_100[:, 2], alpha=0.25)

# set axis limits
plt.xlim(0, maes_005[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("plots/hartmann-stepsize-mae.png", dpi=1000)
plt.savefig("plots/hartmann-stepsize-mae.eps", format="eps")
plt.clf()
