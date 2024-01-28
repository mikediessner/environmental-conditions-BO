from source.plot import give_scores, aggregate_mae
import matplotlib.pyplot as plt


# set up experiment
runs = 30          # number of replicaions
evals = 100        # number of evaluations
eval_stepsize = 10 # increments of predictions

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/observation-ei/observation-ei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/prediction-ei/prediction-ei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/sampling-ei/sampling-ei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/montecarlo-ei/montecarlo-ei-hartmann")

# compute average and standard deviation of mean absolute error over all runs
maes_obs = aggregate_mae(runs, "results/hartmann/observation-ei/observation-ei-hartmann")
maes_pred = aggregate_mae(runs, "results/hartmann/prediction-ei/prediction-ei-hartmann")
maes_samp = aggregate_mae(runs, "results/hartmann/sampling-ei/sampling-ei-hartmann")
maes_mc = aggregate_mae(runs, "results/hartmann/montecarlo-ei/montecarlo-ei-hartmann")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(maes_obs[:, 0], maes_obs[:, 1], c="tab:blue", label="Observation")
plt.plot(maes_obs[:, 0], maes_obs[:, 1]-1.96*maes_obs[:, 2], c="tab:blue", linestyle="dotted")
plt.plot(maes_obs[:, 0], maes_obs[:, 1]+1.96*maes_obs[:, 2], c="tab:blue", linestyle="dotted")

plt.plot(maes_pred[:, 0], maes_pred[:, 1], c="tab:orange", label="Prediction")
plt.plot(maes_pred[:, 0], maes_pred[:, 1]-1.96*maes_pred[:, 2], c="tab:orange", linestyle="dotted")
plt.plot(maes_pred[:, 0], maes_pred[:, 1]+1.96*maes_pred[:, 2], c="tab:orange", linestyle="dotted")

plt.plot(maes_samp[:, 0], maes_samp[:, 1], c="tab:red", label="Sampling")
plt.plot(maes_samp[:, 0], maes_samp[:, 1]-1.96*maes_samp[:, 2], c="tab:red", linestyle="dotted")
plt.plot(maes_samp[:, 0], maes_samp[:, 1]+1.96*maes_samp[:, 2], c="tab:red", linestyle="dotted")

plt.plot(maes_mc[:, 0], maes_mc[:, 1], c="tab:green", label="Monte Carlo")
plt.plot(maes_mc[:, 0], maes_mc[:, 1]-1.96*maes_mc[:, 2], c="tab:green", linestyle="dotted")
plt.plot(maes_mc[:, 0], maes_mc[:, 1]+1.96*maes_mc[:, 2], c="tab:green", linestyle="dotted")


# set axis limits
plt.xlim(0, maes_obs[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")
plt.title("Expected improvement")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("results/plots/comparison-hartmann-ei.png", dpi=1000)
plt.savefig("results/plots/comparison-hartmann-ei.eps", format="eps")
plt.clf()



# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/ucb-4/ucb-4-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/ucb-8/ucb-8-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/ucb-16/ucb-16-hartmann")

# compute average and standard deviation of mean absolute error over all runs
maes_ucb4 = aggregate_mae(runs, "results/hartmann/ucb-4/ucb-4-hartmann")
maes_ucb8 = aggregate_mae(runs, "results/hartmann/ucb-8/ucb-8-hartmann")
maes_ucb16 = aggregate_mae(runs, "results/hartmann/ucb-16/ucb-16-hartmann")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(maes_ucb4[:, 0], maes_ucb4[:, 1], c="tab:blue", label="4.0")
plt.plot(maes_ucb4[:, 0], maes_ucb4[:, 1]-1.96*maes_ucb4[:, 2], c="tab:blue", linestyle="dotted")
plt.plot(maes_ucb4[:, 0], maes_ucb4[:, 1]+1.96*maes_ucb4[:, 2], c="tab:blue", linestyle="dotted")

plt.plot(maes_ucb8[:, 0], maes_ucb8[:, 1], c="tab:orange", label="8.0")
plt.plot(maes_ucb8[:, 0], maes_ucb8[:, 1]-1.96*maes_ucb8[:, 2], c="tab:orange", linestyle="dotted")
plt.plot(maes_ucb8[:, 0], maes_ucb8[:, 1]+1.96*maes_ucb8[:, 2], c="tab:orange", linestyle="dotted")

plt.plot(maes_ucb16[:, 0], maes_ucb16[:, 1], c="tab:red", label="16.0")
plt.plot(maes_ucb16[:, 0], maes_ucb16[:, 1]-1.96*maes_ucb16[:, 2], c="tab:red", linestyle="dotted")
plt.plot(maes_ucb16[:, 0], maes_ucb16[:, 1]+1.96*maes_ucb16[:, 2], c="tab:red", linestyle="dotted")

# set axis limits
plt.xlim(0, maes_ucb4[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")
plt.title("Upper confidence bound")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("results/plots/comparison-hartmann-ucb.png", dpi=1000)
plt.savefig("results/plots/comparison-hartmann-ucb.eps", format="eps")
plt.clf()




# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/observation-logei/observation-logei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/prediction-logei/prediction-logei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/sampling-logei/sampling-logei-hartmann")

# compute average and standard deviation of mean absolute error over all runs
maes_obs = aggregate_mae(runs, "results/hartmann/observation-logei/observation-logei-hartmann")
maes_pred = aggregate_mae(runs, "results/hartmann/prediction-logei/prediction-logei-hartmann")
maes_samp = aggregate_mae(runs, "results/hartmann/sampling-logei/sampling-logei-hartmann")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(maes_obs[:, 0], maes_obs[:, 1], c="tab:blue", label="Observation")
plt.plot(maes_obs[:, 0], maes_obs[:, 1]-1.96*maes_obs[:, 2], c="tab:blue", linestyle="dotted")
plt.plot(maes_obs[:, 0], maes_obs[:, 1]+1.96*maes_obs[:, 2], c="tab:blue", linestyle="dotted")

plt.plot(maes_pred[:, 0], maes_pred[:, 1], c="tab:orange", label="Prediction")
plt.plot(maes_pred[:, 0], maes_pred[:, 1]-1.96*maes_pred[:, 2], c="tab:orange", linestyle="dotted")
plt.plot(maes_pred[:, 0], maes_pred[:, 1]+1.96*maes_pred[:, 2], c="tab:orange", linestyle="dotted")

plt.plot(maes_samp[:, 0], maes_samp[:, 1], c="tab:red", label="Sampling")
plt.plot(maes_samp[:, 0], maes_samp[:, 1]-1.96*maes_samp[:, 2], c="tab:red", linestyle="dotted")
plt.plot(maes_samp[:, 0], maes_samp[:, 1]+1.96*maes_samp[:, 2], c="tab:red", linestyle="dotted")

# set axis limits
plt.xlim(0, maes_obs[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")
plt.title("Log expected improvement")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("results/plots/comparison-hartmann-logei.png", dpi=1000)
plt.savefig("results/plots/comparison-hartmann-logei.eps", format="eps")
plt.clf()



# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/observation-ei/observation-ei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/ucb-4/ucb-4-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "results/hartmann/sampling-logei/sampling-logei-hartmann")

# compute average and standard deviation of mean absolute error over all runs
maes_ei = aggregate_mae(runs, "results/hartmann/observation-ei/observation-ei-hartmann")
maes_ucb = aggregate_mae(runs, "results/hartmann/ucb-4/ucb-4-hartmann")
maes_logei = aggregate_mae(runs, "results/hartmann/sampling-logei/sampling-logei-hartmann")

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
plt.tight_layout()

# make figure
fig = plt.figure()

# plot
plt.plot(maes_ei[:, 0], maes_ei[:, 1], c="tab:blue", label="EI (observation)")
plt.plot(maes_ei[:, 0], maes_ei[:, 1]-1.96*maes_ei[:, 2], c="tab:blue", linestyle="dotted")
plt.plot(maes_ei[:, 0], maes_ei[:, 1]+1.96*maes_ei[:, 2], c="tab:blue", linestyle="dotted")

plt.plot(maes_ucb[:, 0], maes_ucb[:, 1], c="tab:orange", label="Log EI (sampling)")
plt.plot(maes_ucb[:, 0], maes_ucb[:, 1]-1.96*maes_ucb[:, 2], c="tab:orange", linestyle="dotted")
plt.plot(maes_ucb[:, 0], maes_ucb[:, 1]+1.96*maes_ucb[:, 2], c="tab:orange", linestyle="dotted")

plt.plot(maes_logei[:, 0], maes_logei[:, 1], c="tab:red", label="UCB (beta=4.0)")
plt.plot(maes_logei[:, 0], maes_logei[:, 1]-1.96*maes_logei[:, 2], c="tab:red", linestyle="dotted")
plt.plot(maes_logei[:, 0], maes_logei[:, 1]+1.96*maes_logei[:, 2], c="tab:red", linestyle="dotted")

# set axis limits
plt.xlim(0, maes_ei[-1, 0])
plt.ylim(0, )

# set axis labels
plt.xlabel("Evaluations")
plt.ylabel("Mean absolute error")
plt.title("Log expected improvement")

# make legend
plt.legend()

# save plot
fig.set_rasterized(True)
plt.savefig("results/plots/comparison-hartmann-best.png", dpi=1000)
plt.savefig("results/plots/comparison-hartmann-best.eps", format="eps")
plt.clf()