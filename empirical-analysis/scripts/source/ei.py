import torch
import numpy as np
from nubo.acquisition import ExpectedImprovement
from nubo.models import GaussianProcess, fit_gp
from gpytorch.likelihoods import GaussianLikelihood
from source.optim import optim
from source.helper import make_header, update_dynamics, reduce_dims
import json


def envbo_step(x_train, y_train, bounds, x_dynamic, dynamic_dims, num_starts, num_samples):

    dims = range(bounds.size(1))

    # bounds
    control_dims = list(set(dims).difference(set(dynamic_dims)))
    control_bounds = bounds[:, control_dims]

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
    acq = ExpectedImprovement(gp=gp, y_best=torch.max(y_train))

    # optimise acquisition function conditional on dynamic parameter
    x_new, ei = optim(func=reduce_dims,
                      bounds=control_bounds,
                      num_starts=num_starts,
                      num_samples=num_samples,
                      func_args=(acq, dynamic_dims, x_dynamic, False))

    for i, dim in enumerate(dynamic_dims):
        x_new = torch.hstack([x_new[:, :dim], x_dynamic[i].reshape(1, 1), x_new[:, dim:]])

    return x_new


def envbo_experiment(func, noise, bounds, random_walk, dynamic_dims, num_starts, num_samples, evals, run, filename, stepsize, starts=None):

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
    params = {"method": "EI",
              "noise": noise,
              "bounds": bounds.tolist(),
              "random_walk": random_walk.flatten().tolist(),
              "dynamic_dims": dynamic_dims,
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
        x_dynamic = update_dynamics(x=x_train[-1, :],
                                    bounds=bounds,
                                    stepsize=stepsize,
                                    random_walk=random_walk[iter, :],
                                    dynamic_dims=dynamic_dims)
        
        # compute next candidate point
        x_new = envbo_step(x_train=x_train,
                           y_train=y_train,
                           bounds=bounds,
                           x_dynamic=x_dynamic,
                           dynamic_dims=dynamic_dims,
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
                   header=make_header(dims, dynamic_dims),
                   comments="")
    