import os
import torch
import numpy as np
from source.ucb import envbo_experiment
from nubo.test_functions import Hartmann6D
from joblib import Parallel, delayed


# set up experiment
func = Hartmann6D(minimise=False)     # objective function
noise = 0.0                           # std of Gaussian noise added to func
bounds = func.bounds                  # upper and lower bounds for parameters
dynamic_dims = [5]                    # list of indices for dynamic dimensions
num_starts = 20                       # number of initial starts for the optimiser
num_samples = 500                     # number of samples for num_starts
evals = 100                           # number of evaluations
runs = 30                             # number of replicaions
directory = "results/hartmann/ucb-4/" # directory path
filename = "ucb-4-hartmann"           # filename for output csv files
stepsize = [0.05]                     # stepsize of random walk
cores = 6                             # number of parallel jobs
beta = 4.0

# check if directory exists and create
if not os.path.isdir(directory):
    os.mkdir(directory)

# load all random walks and initial starting points
random_walks, starts = [], []
for run in range(runs):
    walk = np.loadtxt(f"data/random_walks/1d_random_walk_run{run+1}.csv", delimiter=",")
    walk = torch.from_numpy(walk).reshape((-1, len(dynamic_dims)))
    random_walks.append(walk)

    start = np.loadtxt(f"data/starts/hartmann_starts_run{run+1}.csv", delimiter=",")
    start = torch.from_numpy(start).reshape((1, bounds.size(1)+1))
    starts.append(start)

# run experiments in parallel
Parallel(n_jobs=cores)(delayed(envbo_experiment)(func=func,
                                                 noise=noise,
                                                 bounds=bounds,
                                                 beta=beta,
                                                 random_walk=random_walks[run],
                                                 dynamic_dims=dynamic_dims,
                                                 num_starts=num_starts,
                                                 num_samples=num_samples,
                                                 evals=evals,
                                                 run=run,
                                                 filename=directory+filename,
                                                 stepsize=stepsize,
                                                 starts=starts[run]
                                                 ) for run in range(runs))
