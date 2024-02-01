import numpy as np


# set up variables
directory = "data/random_walks/"
evaluations = 100
runs = 30

# sample random walks for all runs
for run in range(runs):

    # sample random walks for 3 parameters
    walk_3d = np.random.uniform(low=-1.0, high=1.0, size=(evaluations, 3))

    # drop last two / last column for random walks with 2 / 1 parameter(s)
    walk_2d = walk_3d[:, :2]
    walk_1d = walk_3d[:, :1].reshape((evaluations, 1))

    # save random walks
    np.savetxt(f"{directory}3d_random_walk_run{run+1}.csv", walk_3d, delimiter=",")
    np.savetxt(f"{directory}2d_random_walk_run{run+1}.csv", walk_2d, delimiter=",")
    np.savetxt(f"{directory}1d_random_walk_run{run+1}.csv", walk_1d, delimiter=",")
