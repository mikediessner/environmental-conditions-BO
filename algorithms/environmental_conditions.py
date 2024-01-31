import torch
from nubo.utils import normalise, unnormalise, standardise
from nubo.models import GaussianProcess, fit_gp
from nubo.acquisition import ExpectedImprovement
from nubo.optimisation import gen_candidates
from gpytorch.likelihoods import GaussianLikelihood
from copy import deepcopy
from scipy.optimize import minimize


from typing import Callable, List, Optional, Tuple, Any


def envbo(x_train: torch.Tensor,
          y_train: torch.Tensor,
          env_dims: int | List[int], 
          env_values: float | List[float],
          bounds: torch.Tensor,
          constraints: Optional[dict | Tuple[dict]]=(),
          normalise_x: Optional[bool]=False,
          standardise_y: Optional[bool]=False,
          num_starts: Optional[int]=10,
          num_samples: Optional[int]=100) -> torch.Tensor:
    """
    Optimisation step of Bayesian optimisation algorithm with environmental
    conditions.

    Parameters
    ----------
    x_train : ``torch.Tensor``
        (size n x d) Training inputs.
    y_train : ``torch.Tensor``
        (size n) Training outputs.
    env_dims : ``int`` or ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``float`` or ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``, optional
        Optimisation constraints on inputs, default is no constraints.
    normalise_x: bool, optional
        Whether inputs should be normalised before optimisation, default is
        False.
    standardise_y: bool, optional
        Whether outpus should be standardised before optimisation, default is
        False
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.

    Returns
    -------
    x_new : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    """

    # get number of parameters
    dims = bounds.size((1))

    opt_bounds = deepcopy(bounds)
    
    # normalise inputs
    if normalise_x:
        x_train = normalise(x_train, bounds)
        opt_bounds = torch.tensor([[0,]*dims, [1,]*dims])
        env_values = normalise(torch.tensor([env_values]), bounds[:, env_dims]).reshape(-1).tolist()

    # standardise outputs
    if standardise_y:
        y_train = standardise(y_train)

    # OPTIMISATION STEP
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

    # optimise acquisition function conditional on environmental conditions
    x_new, ei = _cond_optim(func=acq,
                            env_dims=env_dims,
                            env_values=env_values,
                            bounds=opt_bounds,
                            constraints=constraints,
                            num_starts=num_starts,
                            num_samples=num_samples)

    # unnormalise new point
    if normalise_x:
        x_new = unnormalise(x_new, bounds)

    return x_new


def _cond_optim(func: Callable,
                env_dims: int | List[int],
                env_values: float | List[float],
                bounds: torch.Tensor,
                constraints: Optional[dict | Tuple[dict]]=(),
                num_starts: Optional[int]=10,
                num_samples: Optional[int]=100,
                **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Optimisation constraints on inputs, default is no constraints.
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

    # make sure env_dims and env_values are lists
    if not isinstance(env_dims, (int, list)):
        raise TypeError("env_dims must be an int or a list of ints.")
    if isinstance(env_dims, int):
        env_dims = [env_dims,]
    
    if not isinstance(env_values, (int, float, list)):
        raise TypeError("env_values must be an int, float or a list of ints/floats.")
    if isinstance(env_values, (int, float)):
        env_values = [env_values,]

    # make sure constraints have the correct type
    if not isinstance(constraints, (dict, tuple)):
        raise TypeError("Constraints must be dict or a tuple of dicts.")
    if isinstance(constraints, dict):
        constraints = [constraints,]
    if isinstance(constraints, tuple):
        constraints = list(constraints)

    # add environmental conditions to constraints
    def create_con(dim, value):
        return {"type": "eq", "fun": lambda x: x[dim] - value}

    for i in range(len(env_dims)):
        constraints.append(create_con(env_dims[i], env_values[i]))

    # optimise
    best_results, best_func_result = slsqp(func=func,
                                           env_dims=env_dims,
                                           env_values=env_values,
                                           bounds=bounds,
                                           constraints=constraints,
                                           num_starts=num_starts,
                                           num_samples=num_samples,
                                           **kwargs)
    
    return best_results, best_func_result


def slsqp(func: Callable,
          env_dims,
          env_values,
          bounds: torch.Tensor,
          constraints: Optional[dict | Tuple[dict]]=(),
          num_starts: Optional[int]=10,
          num_samples: Optional[int]=100,
          **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
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
    constraints : ``dict`` or ``Tuple`` of ``dict``, optional
        Optimisation constraints, default is no constraints.
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

    # restrict bounds of environmental variables for candidate sampling in SLSQP
    env_bounds = deepcopy(bounds.float())
    for i in range(len(env_dims)):
        env_bounds[0, env_dims[i]] = env_values[i]
        env_bounds[1, env_dims[i]] = env_values[i]
    
    # generate candidates
    candidates = gen_candidates(func, env_bounds, num_starts, num_samples)
    candidates = candidates.numpy()
    
    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)
    
    # iteratively optimise over candidates
    for i in range(num_starts):
        result = minimize(func,
                          x0=candidates[i],
                          method="SLSQP",
                          bounds=bounds.numpy().T,
                          constraints=constraints,
                          **kwargs)
        results[i, :] = torch.from_numpy(result["x"].reshape(1, -1))
        func_results[i] = float(result["fun"])
    
    # select best candidate
    best_i = torch.argmin(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = torch.reshape(func_results[best_i], (1,))

    return best_result, best_func_result