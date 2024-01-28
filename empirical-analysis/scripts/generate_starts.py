import torch
import numpy as np
from nubo.utils import unnormalise
from nubo.test_functions import Hartmann6D, Levy


# set up variables
directory = "data/starts/"
evaluations = 100
runs = 30


##############
## Hartmann ##
##############

func = Hartmann6D(noise_std=0.0, minimise=False)
bounds = func.bounds
dims = func.dims

# sample starts for all runs
for run in range(runs):

    # sample random start inputs
    x = unnormalise(torch.rand((1, dims), dtype=torch.double), bounds)

    # evaluate random start inputs
    y = func(x)

    # combine => full start
    starts = torch.hstack([x, y.reshape((1, 1))])

    # save start
    np.savetxt(f"{directory}hartmann_starts_run{run+1}.csv", starts, delimiter=",")


##############
## Levy ##
##############

func = Levy(dims=3, noise_std=0.0, minimise=False)
bounds = func.bounds
dims = func.dims

# sample starts for all runs
for run in range(runs):

    # sample random start inputs
    x = unnormalise(torch.rand((1, dims), dtype=torch.double), bounds)

    # evaluate random start inputs
    y = func(x)

    # combine => full start
    starts = torch.hstack([x, y.reshape((1, 1))])

    # save start
    np.savetxt(f"{directory}levy_starts_run{run+1}.csv", starts, delimiter=",")
