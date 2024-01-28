from source.plot import give_scores, aggregate_mae, make_plot


# set up experiment
runs = 30                                                            # number of replicaions
evals = 100                                                          # number of evaluations
eval_stepsize = 10                                                   # increments of predictions
filename = "results/hartmann/observation-ei/observation-ei-hartmann" # filename for output csv files

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, filename)

# compute average and standard deviation of mean absolute error over all runs
maes = aggregate_mae(runs, filename)

# make plot
make_plot(maes, "results/plots/observation-ei-hartmann")
