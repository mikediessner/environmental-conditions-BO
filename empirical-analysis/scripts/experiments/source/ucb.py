import torch
import numpy as np
from nubo.acquisition import UpperConfidenceBound
from nubo.models import GaussianProcess, fit_gp
from gpytorch.likelihoods import GaussianLikelihood
from source.helper import make_header, update_env
from nubo.algorithms import _cond_optim
import json


def envbo_step(x_train, y_train, bounds, beta, env_values, env_dims, num_starts, num_samples):

    dims = range(bounds.size(1))

    # specify Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)

    # fit Gaussian process
    fit_gp(x_train, y_train,
           gp=gp,
           likelihood=likelihood,
           lr=0.1,
           steps=200)

    # specify acquisition function
    acq = UpperConfidenceBound(gp=gp, beta=beta)

    # optimise acquisition function conditional on dynamic parameter
    x_new, ei = _cond_optim(func=acq,
                            env_dims=env_dims,
                            env_values=env_values,
                            bounds=bounds,
                            num_starts=num_starts,
                            num_samples=num_samples)

    return x_new


def envbo_experiment(func, noise, bounds, beta, random_walk, env_dims, num_starts, num_samples, evals, run, filename, stepsize, starts=None):

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
    params = {"method": "UCB",
              "beta": beta,
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
        x_new = envbo_step(x_train=x_train,
                           y_train=y_train,
                           bounds=bounds,
                           beta=beta,
                           env_values=env_values.tolist(),
                           env_dims=env_dims,
                           num_starts=num_starts,
                           num_samples=num_samples)

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
    