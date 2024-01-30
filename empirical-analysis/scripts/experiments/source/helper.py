import torch


def make_header(dims, env_dims):
    header = ["controllable", ] * dims
    for i in env_dims:
        header[i] = "environmental"
    header.append("output")
    header = ",".join(header)

    return header


def update_env(x, bounds, stepsize, random_walk, env_dims):
        
    # compute next values for all dynamic parameters
    nexts = torch.zeros(len(env_dims))
    for i, dim in enumerate(env_dims):
        next = x[dim] + stepsize[i] * random_walk[i]
        nexts[i] = next.clamp(bounds[0, dim], bounds[1, dim])

    return nexts
