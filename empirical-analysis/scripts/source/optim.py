import torch
from torch import Tensor
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable, Any
from nubo.optimisation import gen_candidates
from nubo.utils import LatinHypercubeSampling, unnormalise


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

    # generate samples
    if dims == 1:
        samples = torch.rand((num_samples, 1))
    else:
        lhs = LatinHypercubeSampling(dims)
        samples = lhs.maximin(num_samples)
    
    samples = unnormalise(samples, bounds=bounds)

    # evaluate samples
    samples_res = torch.zeros(num_samples)
    for n in range(num_samples):
        samples_res[n] = func(samples[n, :].reshape(1, -1), *args)

    # select best candidates (smallest output)
    _, best_i = torch.topk(samples_res, num_candidates, largest=False)
    candidates = samples[best_i]
    
    return candidates


def lbfgsb(func: Callable,
           bounds: Tensor,
           num_starts: Optional[int]=10,
           num_samples: Optional[int]=100,
           **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Multi-start L-BFGS-B optimiser using the ``scipy.optimize.minimize``
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
    
    # generate candidates
    candidates = gen_candidates(func, bounds, num_starts, num_samples, **kwargs)
    candidates = candidates.numpy()

    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)

    # iteratively optimise over candidates
    for i in range(num_starts):
        result = minimize(func, x0=candidates[i], method="L-BFGS-B", bounds=opt_bounds, **kwargs)
        results[i, :] = torch.from_numpy(result["x"].reshape(1, -1))
        func_results[i] = float(result["fun"])

    # select best candidate
    best_i = torch.argmin(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = torch.reshape(func_results[best_i], (1,))

    return best_result, best_func_result


def optim(func, bounds, num_samples, num_starts, func_args=()):

    x, y = lbfgsb(func=func,
                  bounds=bounds,
                  num_starts=num_starts,
                  num_samples=num_samples,
                  args=func_args)
    
    return x, y
