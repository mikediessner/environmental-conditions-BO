import numpy as np


runs = 30
evals = 100

sname = "data/starts/hartmann_starts"

wname = ["data/random_walks/1d_random_walk",
         "data/random_walks/2d_random_walk",
         "data/random_walks/3d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk",
         "data/random_walks/1d_random_walk"]
         
rnames = ["data/hartmann/basic/hartmann",
          "data/hartmann/multi_2/hartmann_multi_2",
          "data/hartmann/multi_3/hartmann_multi_3",
          "data/hartmann/noisy_0.025/hartmann_noisy_0.025",
          "data/hartmann/noisy_0.05/hartmann_noisy_0.05",
          "data/hartmann/noisy_0.1/hartmann_noisy_0.1",
          "data/hartmann/stepsize_0.1/hartmann_stepsize_0.1",
          "data/hartmann/stepsize_0.25/hartmann_stepsize_0.25",
          "data/hartmann/stepsize_0.5/hartmann_stepsize_0.5",
          "data/hartmann/stepsize_1.0/hartmann_stepsize_1.0",
          "data/hartmann/variability_low/hartmann_variability_low",
          "data/hartmann/variability_high/hartmann_variability_high"]

dynamic_dims = [[5], [3, 5], [0, 3, 5], [5], [5], [5], [5], [5], [5], [5], [2], [0]]

stepsize = [[0.05], [0.1, 0.05], [0.1, 0.1, 0.05], [0.05], [0.05], [0.05], [0.1], [0.25], [0.5], [1.0], [0.05], [0.05]]


for i, rname in enumerate(rnames):
    print("========================================================")
    print(rname)
    print("========================================================")

    dyn_dims = len(dynamic_dims[i])

    for run in range(runs):

        starts = np.loadtxt(f"{sname}_run{run+1}.csv", delimiter=",")
        results = np.loadtxt(f"{rname}_run{run+1}.csv", skiprows=1, delimiter=",")

        if any(starts != results[0, :]):
            raise ValueError("Starts are incorrect.")
        
        dyn_values = np.zeros((evals, dyn_dims))
        dyn_values[0, :] = starts[dynamic_dims[i]]
        random_walk = np.loadtxt(f"{wname[i]}_run{run+1}.csv", delimiter=",").reshape((-1, dyn_dims))

        for eval in range(1, evals):
            dyn_values[eval, :] = dyn_values[eval-1, :] + random_walk[eval-1, :]*stepsize[i]
            dyn_values[eval, :] = np.clip(dyn_values[eval, :], a_min=0.0, a_max=1.0)

        dyn_results = np.loadtxt(f"{rname}_run{run+1}.csv", skiprows=1, delimiter=",")[:, dynamic_dims[i]]

        if not all(np.sum(np.isclose(dyn_values, dyn_results, atol=0.0001), axis=1) == dyn_dims):
            print(np.hstack([dyn_results, dyn_values]))
            raise ValueError("Random walks are incorrect.")

    print("Correct starts used.")
    print("Correct random walk used.")
    print("\n")
