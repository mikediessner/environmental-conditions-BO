import torch
from nubo.utils import gen_inputs
from nubo.models import GaussianProcess, fit_gp
from gpytorch.likelihoods import GaussianLikelihood
from nubo.test_functions import Hartmann6D


#######################
## Hartmann function ##
#######################

# set seed to make reproducible
torch.manual_seed(2023)

# specify function and generate training data
func = Hartmann6D(0, False)
x = gen_inputs(2000, 6, func.bounds)
y = func(x)
data = torch.hstack([x, y.reshape(-1, 1)])

# initialise Gaussian process
likelihood = GaussianLikelihood()
gp = GaussianProcess(x, y, likelihood=likelihood)

# fit Gaussian process
fit_gp(x, y, gp=gp, likelihood=likelihood, lr=0.1, steps=200)

# estimated length-scales: high => less variability, low => more variability
print(f"Covariance kernel length-scale: {gp.covar_module.base_kernel.lengthscale.detach()}")

# Parameters sorted by length-scale (low to high):
# x1: 1.3041
# x6: 1.4712
# x4: 1.4774
# x5: 1.5102
# x2: 1.8972
# x3: 4.8058
