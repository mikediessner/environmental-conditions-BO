import torch
import numpy as np
from nubo.acquisition import ExpectedImprovement
from nubo.models import GaussianProcess, fit_gp
from gpytorch.likelihoods import GaussianLikelihood
from helper import cond_optim
from nubo.utils import normalise, unnormalise
from scipy.spatial.distance import pdist
from windfarm_simulator import SPACING, simulate_aep
from typing import List, Optional, Tuple


def step(x_train: torch.Tensor,
         y_train: torch.Tensor,
         env_dims: List[float],
         env_values: List[int],
         bounds: torch.Tensor,
         num_starts: Optional[int]=10,
         num_samples: Optional[int]=100) -> torch.Tensor:
    """
    Optimisation step of regular Bayesian optimisation algorithm.

    Parameters
    ----------
    x_train : ``torch.Tensor``
        (size n x d) Training inputs.
    y_train : ``torch.Tensor``
        (size n) Training outputs.
    env_dims : ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.

    Returns
    -------
    x_new : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    """

    # Get number of parameters
    dims = bounds.size((1))

    # Specify Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)

    # Fit Gaussian process
    fit_gp(x_train, y_train,
           gp=gp,
           likelihood=likelihood,
           lr=0.1,
           steps=200)

    # Specify acquisition function
    acq = ExpectedImprovement(gp=gp, y_best=torch.max(y_train))

    # Placement constraint
    def check_distance(x, bounds, spacing):
        "Spacing constraint of wind turbines"
        x = unnormalise(x, bounds.numpy())
        distances = pdist(x[:-len(env_dims)].reshape(2, -1).T, "euclidean")
        min_distance = min(distances)
        return min_distance - spacing
    
    cons = ({"type": "ineq", "fun": lambda x: check_distance(x, bounds, SPACING)},)

    # Optimise acquisition function conditional on dynamic parameter
    x_new, _ = cond_optim(func=acq,
                          env_dims=env_dims,
                          env_values=env_values,
                          bounds=torch.tensor([[0,]*dims, [1,]*dims]),
                          constraints=cons,
                          num_starts=num_starts,
                          num_samples=num_samples,)

    return x_new


def bo_benchmark(env_dims: List[float],
                 env_values: List[int],
                 bounds: torch.Tensor,
                 evals: int,
                 starts: torch.Tensor,
                 num_starts: int,
                 num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Algorithm to run regular Bayesian optimisation benchmark.

    Parameters
    ----------
    env_dims : ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    evals : int
        Number of function evaluations.
    starts : ``torch.Tensor``
        (size 1 x d+1) Initial training inputs and corresponding output.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.

    Returns
    -------
    x_new : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    """

    # Objective function
    def sim(x):

        x = x.reshape((1, -1))
        torch_out = False

        if isinstance(x, torch.Tensor):
            torch_out = True
            x = x.numpy()

        aep = simulate_aep(x)

        if torch_out:
            aep = torch.from_numpy(aep)

        return aep

    # Make training data
    if isinstance(starts, torch.Tensor):
        x_train = starts[:, :-1]
        y_train = starts[:, -1]
    elif isinstance(starts, np.ndarray):
        x_train = torch.from_numpy(starts[:, :-1])
        y_train = torch.from_numpy(starts[:, -1])

    x_train = normalise(x_train, bounds)
    env_values = normalise(env_values, bounds[:, env_dims])

    # Bayesian optimisation loop
    for iter in range(evals-1):

        # Compute next candidate point
        x_new = step(x_train=x_train,
                     y_train=y_train,
                     env_dims=env_dims,
                     env_values=env_values.tolist(),
                     bounds=bounds,
                     num_starts=num_starts,
                     num_samples=num_samples)
        
        # Evaluate new point
        y_new = sim(unnormalise(x_new, bounds))

        # Add to training data
        x_train = torch.vstack((x_train, x_new))
        y_train = torch.hstack((y_train, y_new))

    # Select best candidate
    best_i = torch.argmax(y_train)
    best_x =  torch.reshape(x_train[best_i, :], (1, -1))
    best_y = torch.reshape(y_train[best_i], (1,))
    results = torch.hstack([unnormalise(x_train, bounds), y_train.reshape((-1, 1))])
    
    return unnormalise(best_x, bounds), best_y, evals, results
