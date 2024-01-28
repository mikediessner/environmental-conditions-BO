import torch
import numpy as np


def make_header(dims, dynamic_dims):
    header = ["controllable", ] * dims
    for i in dynamic_dims:
        header[i] = "dynamic"
    header.append("output")
    header = ",".join(header)

    return header


def reduce_dims(x, func, dynamic_dims, dynamic_values, negate):

    # ensure torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).reshape(-1)
    elif isinstance(x, torch.Tensor):
        x = x.reshape(-1)

    # insert column for dynamic parameters in controllable candidates
    for i, dim in enumerate(list(dynamic_dims)):
        x = torch.hstack([x[:dim], dynamic_values[i], x[dim:]])

    # convert to 1 x d tensor
    x = x.reshape(1, -1)

    y = float(func(x))

    if negate == True:
        y = -y

    return y


def update_dynamics(x, bounds, stepsize, random_walk, dynamic_dims):
        
    # compute next values for all dynamic parameters
    nexts = torch.zeros(len(dynamic_dims))
    for i, dim in enumerate(dynamic_dims):
        next = x[dim] + stepsize[i] * random_walk[i]
        nexts[i] = next.clamp(bounds[0, dim], bounds[1, dim])

    return nexts
