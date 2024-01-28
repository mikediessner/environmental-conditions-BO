from source.plot import give_scores, aggregate_mae
import matplotlib.pyplot as plt


# set up experiment
runs = 30          # number of replicaions
evals = 100        # number of evaluations
eval_stepsize = 10 # increments of predictions

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/benchmark/benchmark-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/observation-ei/observation-ei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/observation-logei/observation-logei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/ucb-4/ucb-4-hartmann")

# compute average and standard deviation of mean absolute error over all runs
maes_bm = aggregate_mae(runs, "results/hartmann/benchmark/benchmark-hartmann")
maes_ei = aggregate_mae(runs, "results/hartmann/observation-ei/observation-ei-hartmann")
maes_logei = aggregate_mae(runs, "results/hartmann/observation-logei/observation-logei-hartmann")
maes_ucb = aggregate_mae(runs, "results/hartmann/ucb-4/ucb-4-hartmann")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(maes_bm[:, 0], maes_bm[:, 1], c="black", label="Benchmark")

plt.plot(maes_ei[:, 0], maes_ei[:, 1], c="tab:orange", label="EI")
plt.plot(maes_ei[:, 0], maes_ei[:, 1]-1.96*maes_ei[:, 2], c="tab:orange", linestyle="dotted")
plt.plot(maes_ei[:, 0], maes_ei[:, 1]+1.96*maes_ei[:, 2], c="tab:orange", linestyle="dotted")

plt.plot(maes_logei[:, 0], maes_logei[:, 1], c="tab:blue", label="Log EI")
plt.plot(maes_logei[:, 0], maes_logei[:, 1]-1.96*maes_logei[:, 2], c="tab:blue", linestyle="dotted")
plt.plot(maes_logei[:, 0], maes_logei[:, 1]+1.96*maes_logei[:, 2], c="tab:blue", linestyle="dotted")

plt.plot(maes_ucb[:, 0], maes_ucb[:, 1], c="tab:green", label="UCB")
plt.plot(maes_ucb[:, 0], maes_ucb[:, 1]-1.96*maes_ucb[:, 2], c="tab:green", linestyle="dotted")
plt.plot(maes_ucb[:, 0], maes_ucb[:, 1]+1.96*maes_ucb[:, 2], c="tab:green", linestyle="dotted")


# set axis limits
plt.xlim(0, maes_bm[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("tests/plots/comparison-hartmann.png")
plt.savefig("tests/plots/comparison-levy-2d-1.5.eps", format="eps")
plt.clf()
