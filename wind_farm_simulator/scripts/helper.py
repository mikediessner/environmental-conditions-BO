import numpy as np
from typing import Optional, List 


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
