import torch
import numpy as np
from source.helper import make_header, update_env
import json


def bm_experiment(func, noise, bounds, random_walk, env_dims, num_starts, num_samples, evals, run, filename, stepsize, starts=None):

    # get number of parameters
    dims = bounds.size((1))

    # make training data
    if isinstance(starts, torch.Tensor):
        x_train = starts[:, :-1]
        y_train = starts[:, -1]
    elif isinstance(starts, np.ndarray):
        x_train = torch.from_numpy(starts[:, :-1])
        y_train = torch.from_numpy(starts[:, -1])

    # gather all algorithm parameters
    params = {"method": "Uniform",
              "noise": noise,
              "bounds": bounds.tolist(),
              "random_walk": random_walk.flatten().tolist(),
              "env_dims": env_dims,
              "num_starts": num_starts,
              "num_samples": num_samples,
              "evaluations": evals,
              "run": run,
              "filename": filename,
              "stepsize": stepsize,
              "starts": starts.flatten().tolist()}
    
    # save parameters
    with open(f"{filename}_params_run{run+1}.txt", "w") as convert_file:
        convert_file.write(json.dumps(params, indent=4))


    # Bayesian optimisation loop
    for iter in range(evals-1):

        # compute next random walk step
        env_values = update_env(x=x_train[-1, :],
                                bounds=bounds,
                                stepsize=stepsize,
                                random_walk=random_walk[iter, :],
                                env_dims=env_dims)
        
        # compute next candidate point
        x_new = torch.tensor(np.random.uniform(bounds[0, :], bounds[1, :], (len(bounds[0, :]),)), dtype=torch.float)
        x_new = x_new.reshape((1, -1))
        x_new[0, env_dims] = env_values

        # evaluate new point
        y_new = func(x_new) + torch.normal(mean=0.0, std=noise, size=(1,))

        # add to training data
        x_train = torch.vstack((x_train, x_new))
        y_train = torch.hstack((y_train, y_new))

        # save results
        np.savetxt(f"{filename}_run{run+1}.csv",
                   torch.hstack([x_train, y_train.reshape(-1, 1)]).numpy(),
                   delimiter=",",
                   header=make_header(dims, env_dims),
                   comments="")
