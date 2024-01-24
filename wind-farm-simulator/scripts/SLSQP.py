import torch
from torch import Tensor
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable, List, Any 
from scipy.spatial.distance import pdist
from windfarm_simulator import simulate_aep, SPACING
from helper import gen_candidates


def slsqp(func: Callable,
          bounds: Tensor,
          constraints: dict | Tuple[dict],
          **kwargs: Any) -> Tuple[Tensor, Tensor, int]:
    """
    SLSQP optimiser using the ``scipy.optimize.minimize`` implementation from
    ``SciPy``. Minimises `func`. Also returns number of function evaluations
    SLSQP uses.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``
        Optimisation constraints.
    **kwargs : ``Any``
        Keyword argument passed to ``scipy.optimize.minimize``.

    Returns
    -------
    result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    evals : int
        Number of function evaluations used by SLSQP.
    """

    opt_bounds = bounds.numpy().T
    
    # Generate initial starting point
    x0 = gen_candidates(func, bounds, 1, 1)

    # Optimise
    result = minimize(func,
                        x0=x0.numpy(),
                        method="SLSQP",
                        bounds=opt_bounds,
                        constraints=constraints,
                        **kwargs)

    result = torch.from_numpy(result["x"].reshape(1, -1))
    func_result = torch.from_numpy(result["fun"])
    evals = int(result["nfev"])

    return result, func_result, evals


def cond_slsqp(func: Callable,
               env_dims: int | List[int],
               env_values: float | List[float],
               bounds: Tensor,
               constraints: Optional[dict | Tuple[dict]]=(),
               num_samples: Optional[int]=100,
               num_starts: Optional[int]=10,
               **kwargs: Any) -> Tuple[Tensor, Tensor, int]:
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
    **kwargs : ``Any``
        Keyword argument passed to ``scipy.optimize.minimize``.

    Returns
    -------
    result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    evals : int
        Number of function evaluations used by SLSQP.
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
    results, func_result, evals = slsqp(func=func,
                                        bounds=bounds,
                                        constraints=constraints,
                                        num_starts=num_starts,
                                        num_samples=num_samples,
                                        **kwargs)
    
    return results, func_result, evals


def slsqp_benchmark(env_dims: List[float],
                    env_values: List[int],
                    bounds: torch.Tensor,
                    **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Algorithm to run SLSQP optimisation benchmark.

    Parameters
    ----------

    env_dims : ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    **kwargs : ``Any``
        Keyword argument passed to ``scipy.optimize.minimize``.

    Returns
    -------
    x_new : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    """
            
    # Placement constraint
    def check_distance(x, spacing):
        distances = pdist(x[:-len(env_dims)].reshape(2, -1).T, "euclidean")
        min_distance = min(distances)

        return min_distance - spacing

    cons = ({"type": "ineq", "fun": lambda x: check_distance(x, SPACING)},)

    f_evals = []

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

        f_evals.append(float(aep))
        return -aep

    # Optimise
    x_new, y_new, evals = cond_slsqp(func=sim,
                                     env_dims=env_dims,
                                     env_values=env_values.tolist(),
                                     bounds=bounds,
                                     constraints=cons,
                                     **kwargs)
    
    return x_new, y_new, evals, f_evals
