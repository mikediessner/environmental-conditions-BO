import torch
import numpy as np
from nubo.test_functions import Hartmann6D
from nubo.optimisation import lbfgsb


#######################
## Hartmann function ##
#######################

# set seed to make reproducible
torch.manual_seed(2023)


# specify function for optimiation
func = Hartmann6D(minimise=False)

def opt(x):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.reshape((1, -1))

    return func(x)

# minimum
minimum = -3.32237

# maximum
_, maximum = lbfgsb(opt,
                    bounds=func.bounds,
                    num_starts=50,
                    num_samples=2000)


range_f = abs(float(maximum) - minimum)
print(f"Range: [{minimum}, {float(maximum)}]")
half_range_f = range_f/2


# 68.2%, 95.4%, 99.6% of added noise fall into ranges 
print(f"Sigma=0.025: {0.025/half_range_f:.4f}, {2*0.025/half_range_f:.4f}, {3*0.025/half_range_f:.4f}")
print(f"Sigma=0.05: {0.05/half_range_f:.4f}, {2*0.05/half_range_f:.4f}, {3*0.05/half_range_f:.4f}")
print(f"Sigma=0.1: {0.1/half_range_f:.4f}, {2*0.1/half_range_f:.4f}, {3*0.1/half_range_f:.4f}")
