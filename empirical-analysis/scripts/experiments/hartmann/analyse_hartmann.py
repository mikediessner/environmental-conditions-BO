import torch
import numpy as np
from nubo.test_functions import Hartmann6D
from source.optim import optim
from source.helper import reduce_dims


func = Hartmann6D(noise_std=0, minimise=False) # objective function
bounds = func.bounds                           # upper and lower bounds for parameters
dims = bounds.size(1)                          # number of parameters
num_tests = 21                                 # number of test points
num_starts = 20                                # number of initial starts for the optimiser
num_samples = 500                              # number of samples for num_starts
filename = "data/hartmann"


for DIM in range(dims):

    dynamic_dims = [DIM]

    control_dims = list(set(range(dims)).difference(set(dynamic_dims)))
    control_bounds = bounds[:, control_dims]

    x_test = torch.linspace(0., 1., num_tests).reshape((num_tests, 1))

    # find truth for comparison
    true = np.zeros((num_tests, dims+1))
    for test in range(num_tests):
        
        x_new, y_new = optim(func=reduce_dims,
                             bounds=control_bounds,
                             num_samples=num_samples,
                             num_starts=num_starts,
                             func_args=(func, dynamic_dims, x_test[test, :], True))

        for i, dim in enumerate(dynamic_dims):
            x_new = torch.hstack([x_new[:, :dim],
                                  x_test[test, i].reshape(1, 1),
                                  x_new[:, dim:]])

        true[test, :-1], true[test, -1] = x_new, y_new

        # save truths
        np.savetxt(f"{filename}_truth_dim_{DIM}.csv",
                   true,
                   delimiter=",",
                   comments='')
