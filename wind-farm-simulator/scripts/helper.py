import torch
import numpy as np
from torch import Tensor
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable, List, Any 
from nubo.optimisation import gen_candidates
from nubo.utils import LatinHypercubeSampling, unnormalise


def random_walk(min: int,
                max: int,
                step: int,
                N: int,
                seed: Optional[int]=None) -> np.ndarray:
    """
    Sample random walk for environmental variables.

    Parameters
    ----------
    min : int
        Lower bound of random walk.
    max : in
        Upper bound of random walk.
    step : int
        Maximal change of each step of the random walk compared to the previous.
    N : int
        Number of steps of random walk.
    seed : int, optional
        Seed of random number generator, default is ``None``.
    
    Returns
    -------
    np.ndarray
        Random walk.
    """
    # Set seed
    np.random.seed(seed)

    # Sample random walk
    walk = np.zeros(N)
    walk[0] = np.random.uniform(min, max, 1)
    for n in range(1, N):
        new = walk[n-1] + np.random.uniform(-step, step, 1)
        if new < min:
            walk[n] = min
        elif new > max:
            walk[n] = max
        else:
            walk[n] = new

    # Reset seed
    np.random.seed(None)
    return walk


def make_header(dims: int, env_dims: List[int]) -> str:
    """
    Make header for CSV files.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    env_dims : ``List`` of ``int``
        Indices of environmental variables.

    Returns
    -------
    str
        Header for CSV file.
    """
    header = ["controllable", ] * dims
    for i in env_dims:
        header[i] = "environmental"
    header.append("output")
    header = ",".join(header)

    return header


def gen_candidates(func: Callable,
                   bounds: Tensor,
                   num_candidates: int,
                   num_samples: int,
                   args=()) -> Tensor:
    """
    Generate candidates for multi-start optimisation using a maximin Latin 
    hypercube design.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    num_candidates : ``int``
        Number of candidates.
    num_samples : ``int``
        Number of samples from which to draw the starts.

    Returns
    -------
    ``torch.Tensor``
        (size `num_candidates` x d) Candidates.
    """

    dims = bounds.size(1)

    # Generate samples
    if dims == 1:
        samples = torch.rand((num_samples, 1))
    else:
        lhs = LatinHypercubeSampling(dims)
        samples = lhs.random(num_samples)
    
    samples = unnormalise(samples, bounds=bounds)

    # Evaluate samples
    samples_res = torch.zeros(num_samples)
    for n in range(num_samples):
        samples_res[n] = func(samples[n, :].reshape(1, -1), *args)

    # Select best candidates (smallest output)
    _, best_i = torch.topk(samples_res, num_candidates, largest=False)
    candidates = samples[best_i]
    
    return candidates


def slsqp(func: Callable,
          bounds: Tensor,
          constraints: dict | Tuple[dict],
          num_starts: Optional[int]=10,
          num_samples: Optional[int]=100,
          **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Multi-start SLSQP optimiser using the ``scipy.optimize.minimize``
    implementation from ``SciPy``.
    
    Used for optimising analytical acquisition functions or Monte Carlo
    acquisition function when base samples are fixed. Picks the best
    `num_starts` points from a total `num_samples` Latin hypercube samples to
    initialise the optimser. Returns the best result. Minimises `func`.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``
        Optimisation constraints.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.
    **kwargs : ``Any``
        Keyword argument passed to ``scipy.optimize.minimize``.

    Returns
    -------
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    dims = bounds.size(1)
    opt_bounds = bounds.numpy().T
    
    # Generate candidates
    candidates = gen_candidates(func, bounds, num_starts, num_samples)
    candidates = candidates.numpy()

    # Initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)

    # Iteratively optimise over candidates
    for i in range(num_starts):
        result = minimize(func,
                          x0=candidates[i],
                          method="SLSQP",
                          bounds=opt_bounds,
                          constraints=constraints,
                          **kwargs)
        
        results[i, :] = torch.from_numpy(result["x"].reshape(1, -1))
        func_results[i] = float(result["fun"])

    # Select best candidate
    best_i = torch.argmin(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = torch.reshape(func_results[best_i], (1,))
    
    return best_result, best_func_result


def cond_optim(func: Callable,
               env_dims: int | List[int],
               env_values: float | List[float],
               bounds: Tensor,
               constraints: Optional[dict | Tuple[dict]]=(),
               num_samples: Optional[int]=100,
               num_starts: Optional[int]=10,
               **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Conditional optimisation for environmental conditions. Holds envrionmental
    variables with indices `env_dims` fixed at measuremens `env_values`.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    env_dims : ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``
        Optimisation constraints.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.

    Returns
    -------
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    # Make sure env_dims and env_values are lists
    if not isinstance(env_dims, (int, list)):
        raise TypeError("env_dims must be an int or a list of ints.")
    if isinstance(env_dims, int):
        env_dims = [env_dims,]
    
    if not isinstance(env_values, (int, float, list)):
        raise TypeError("env_values must be an int, float or a list of ints/floats.")
    if isinstance(env_values, (int, float)):
        env_values = [env_values,]

    # Make sure constraints have the correct type
    if not isinstance(constraints, (dict, tuple)):
        raise TypeError("Constraints must be dict or a tuple of dicts.")
    if isinstance(constraints, dict):
        constraints = [constraints,]
    if isinstance(constraints, tuple):
        constraints = list(constraints)

    # Add environmental conditions to constraints
    def create_con(dim, value):
        return {"type": "eq", "fun": lambda x: x[dim] - value}

    for i in range(len(env_dims)):
        constraints.append(create_con(env_dims[i], env_values[i]))

    # Optimise conditionally on environmental conditions
    best_results, best_func_result = slsqp(func=func,
                                           bounds=bounds,
                                           constraints=constraints,
                                           num_starts=num_starts,
                                           num_samples=num_samples,
                                           **kwargs)
    
    return best_results, best_func_result