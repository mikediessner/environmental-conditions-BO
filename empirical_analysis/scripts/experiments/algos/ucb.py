import os
import torch
import numpy as np
from source.ucb import envbo_experiment
from source.predict import make_predictions
from source.plot import give_scores, aggregate_mape, make_plot
from joblib import Parallel, delayed


def ucb(func, noise, bounds, env_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str, beta):
    # paths
    data_dir = f"empirical_analysis/results/{name}_ucb{beta}/"
    plot_dir = "empirical_analysis/plots/"
    filename = f"{name}_ucb{beta}"

    # check if directory exists and create
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # EXPERIMENTS
    # load all random walks and initial starting points
    random_walks, starts = [], []
    for run in range(runs):
        walk = np.loadtxt(f"empirical_analysis/data/random_walks/{walk_str}_random_walk_run{run+1}.csv", delimiter=",")
        walk = torch.from_numpy(walk).reshape((-1, len(env_dims)))
        random_walks.append(walk)

        start = np.loadtxt(f"empirical_analysis/data/starts/{start_str}_starts_run{run+1}.csv", delimiter=",")
        start = torch.from_numpy(start).reshape((1, bounds.size(1)+1))
        starts.append(start)

    # run experiments
    Parallel(n_jobs=cores)(delayed(envbo_experiment)(func=func,
                                                     noise=noise,
                                                     bounds=bounds,
                                                     beta=beta,
                                                     random_walk=random_walks[run],
                                                     env_dims=env_dims,
                                                     num_starts=num_starts,
                                                     num_samples=num_samples,
                                                     evals=evals,
                                                     run=run,
                                                     filename=data_dir+filename,
                                                     stepsize=stepsize,
                                                     starts=starts[run]
                                                     ) for run in range(runs))


    # PREDICTIONS
    Parallel(n_jobs=cores)(delayed(make_predictions)(func=func,
                                                     num_tests=num_tests,
                                                     eval_stepsize=eval_stepsize,
                                                     filename=data_dir+filename,
                                                     run=run
                                                     ) for run in range(runs))


    # PLOTS
    # compute mean absolute error
    for run in range(runs):
        give_scores(evals, eval_stepsize, run, data_dir+filename)

    # compute average and standard deviation of mean absolute error over all runs
    maes = aggregate_mape(runs, data_dir+filename)

    # make plot
    make_plot(maes, plot_dir+filename)
