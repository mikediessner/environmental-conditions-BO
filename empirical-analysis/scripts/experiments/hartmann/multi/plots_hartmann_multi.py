from source.plot import give_scores, aggregate_mae
import matplotlib.pyplot as plt


# set up experiment
runs = 30          # number of replicaions
evals = 100        # number of evaluations
eval_stepsize = 10 # increments of predictions

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/multi_2/hartmann_multi_2")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/multi_3/hartmann_multi_3")

# compute average and standard deviation of mean absolute error over all runs
two_maes = aggregate_mae(runs, "data/hartmann/multi_2/hartmann_multi_2")
three_maes = aggregate_mae(runs, "data/hartmann/multi_3/hartmann_multi_3")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(two_maes[:, 0], two_maes[:, 1], label=f"$n_d = 2$")
plt.fill_between(two_maes[:, 0], two_maes[:, 1]-1.96*two_maes[:, 2], two_maes[:, 1]+1.96*two_maes[:, 2], alpha=0.25)
plt.plot(three_maes[:, 0], three_maes[:, 1], label=f"$n_d = 3$")
plt.fill_between(three_maes[:, 0], three_maes[:, 1]-1.96*three_maes[:, 2], three_maes[:, 1]+1.96*three_maes[:, 2], alpha=0.25)

# set axis limits
plt.xlim(0, two_maes[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("plots/hartmann-multi-mae.png", dpi=1000)
plt.savefig("plots/hartmann-multi-mae.eps", format="eps")
plt.clf()
