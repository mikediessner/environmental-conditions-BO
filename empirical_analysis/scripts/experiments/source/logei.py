import torch
from torch import Tensor
from gpytorch.models import GP
from nubo.acquisition.acquisition_function import AcquisitionFunction
import math
from math import pi
import numpy as np
from nubo.models import GaussianProcess, fit_gp
from gpytorch.likelihoods import GaussianLikelihood
from nubo.algorithms import _cond_optim
from source.helper import make_header, update_env
import json


_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2
_log_sqrt_2pi = math.log(2 * pi) / 2
_log2 = math.log(2)
_inv_sqrt_2pi = 1 / math.sqrt(2 * pi)


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)


def _log_abs_u_Phi_div_phi(u: Tensor) -> Tensor:
    """Computes log(abs(u) * Phi(u) / phi(u)), where phi and Phi are the normal pdf
    and cdf, respectively. The function is valid for u < 0.

    NOTE: In single precision arithmetic, the function becomes numerically unstable for
    u < -1e3. For this reason, a second branch in _log_ei_helper is necessary to handle
    this regime, where this function approaches -abs(u)^-2 asymptotically.

    The implementation is based on the following implementation of the logarithm of
    the scaled complementary error function (i.e. erfcx). Since we only require the
    positive branch for _log_ei_helper, _log_abs_u_Phi_div_phi does not have a branch,
    but is only valid for u < 0 (so that _neg_inv_sqrt2 * u > 0).

        def logerfcx(x: Tensor) -> Tensor:
            return torch.where(
                x < 0,
                torch.erfc(x.masked_fill(x > 0, 0)).log() + x**2,
                torch.special.erfcx(x.masked_fill(x < 0, 0)).log(),
        )

    Further, it is important for numerical accuracy to move u.abs() into the
    logarithm, rather than adding u.abs().log() to logerfcx. This is the reason
    for the rather complex name of this function: _log_abs_u_Phi_div_phi.
    """
    # get_constants_like allocates tensors with the appropriate dtype and device and
    # caches the result, which improves efficiency.
    a, b = _neg_inv_sqrt2, _log_sqrt_pi_div_2
    return torch.log(torch.special.erfcx(a * u) * u.abs()) + b


def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    log2 = _log2
    is_small = -log2 < x  # x < 0
    return torch.where(
        is_small,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )


def log_phi(x: Tensor) -> Tensor:
    r"""Logarithm of standard normal pdf"""
    log_sqrt_2pi, neg_half = _log_sqrt_2pi, -0.5
    return neg_half * x.square() - log_sqrt_2pi


class LogExpectedImprovement(AcquisitionFunction):
    r"""
    https://arxiv.org/pdf/2310.20708v1.pdf#page=14&zoom=100,144,401
    """

    def __init__(self,
                 gp: GP,
                 y_best: Tensor) -> None:

        self.gp = gp
        self.y_best = y_best
    
    def _log_ei_helper(self, u: Tensor) -> Tensor:
        """Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner for u in
        [-10^100, 10^100] in double precision, and [-10^20, 10^20] in single precision.
        Beyond these intervals, a basic squaring of u can lead to floating point overflow.
        In contrast, the implementation in _ei_helper only yields usable gradients down to
        u ~ -10. As a consequence, _log_ei_helper improves the range of inputs for which a
        backward pass yields usable gradients by many orders of magnitude.
        """
        if not (u.dtype == torch.float32 or u.dtype == torch.float64):
            raise TypeError(
                f"LogExpectedImprovement only supports torch.float32 and torch.float64 "
                f"dtypes, but received {u.dtype = }."
            )
        # The function has two branching decisions. The first is u < bound, and in this
        # case, just taking the logarithm of the naive _ei_helper implementation works.
        bound = -1
        u_upper = u.masked_fill(u < bound, bound)  # mask u to avoid NaNs in gradients
        log_ei_upper = _ei_helper(u_upper).log()

        # When u <= bound, we need to be more careful and rearrange the EI formula as
        # log(phi(u)) + log(1 - exp(w)), where w = log(abs(u) * Phi(u) / phi(u)).
        # To this end, a second branch is necessary, depending on whether or not u is
        # smaller than approximately the negative inverse square root of the machine
        # precision. Below this point, numerical issues in computing log(1 - exp(w)) occur
        # as w approaches zero from below, even though the relative contribution to log_ei
        # vanishes in machine precision at that point.
        neg_inv_sqrt_eps = -1e6 if u.dtype == torch.float64 else -1e3

        # mask u for to avoid NaNs in gradients in first and second branch
        u_lower = u.masked_fill(u > bound, bound)
        u_eps = u_lower.masked_fill(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps)
        # compute the logarithm of abs(u) * Phi(u) / phi(u) for moderately large negative u
        w = _log_abs_u_Phi_div_phi(u_eps)

        # 1) Now, we use a special implementation of log(1 - exp(w)) for moderately
        # large negative numbers, and
        # 2) capture the leading order of log(1 - exp(w)) for very large negative numbers.
        # The second special case is technically only required for single precision numbers
        # but does "the right thing" regardless.
        log_ei_lower = log_phi(u) + (
            torch.where(
                u > neg_inv_sqrt_eps,
                log1mexp(w),
                # The contribution of the next term relative to log_phi vanishes when
                # w_lower << eps but captures the leading order of the log1mexp term.
                -2 * u_lower.abs().log(),
            )
        )
        return torch.where(u > bound, log_ei_upper, log_ei_lower)
        
    def eval(self, x: Tensor) -> Tensor:

        # check that only one point is queried
        if x.size(0) != 1:
            raise ValueError("Only one point (size 1 x d) can be computed at a time.")
        
        # set Gaussian Process to eval mode
        self.gp.eval()

        # make predictions
        pred = self.gp(x)

        mean = pred.mean
        variance = pred.variance
        std = torch.sqrt(variance).clamp_min(1e-10)

        # compute log Expected Improvement
        z = (mean - self.y_best)/std
        ei = self._log_ei_helper(z) + std.log()

        return -ei


def phi(x: Tensor) -> Tensor:
    r"""Standard normal PDF."""
    inv_sqrt_2pi, neg_half = _inv_sqrt_2pi, -0.5
    return inv_sqrt_2pi * (neg_half * x.square()).exp()


def Phi(x: Tensor) -> Tensor:
    r"""Standard normal CDF."""
    log_sqrt_2pi, neg_half = _log_sqrt_2pi, -0.5
    return neg_half * x.square() - log_sqrt_2pi


def envbo_step(x_train, y_train, bounds, env_values, env_dims, num_starts, num_samples):

    dims = range(bounds.size(1))

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
    acq = LogExpectedImprovement(gp=gp, y_best=torch.max(y_train))

    # optimise acquisition function conditional on dynamic parameter
    x_new, ei = _cond_optim(func=acq,
                            env_dims=env_dims,
                            env_values=env_values,
                            bounds=bounds,
                            num_starts=num_starts,
                            num_samples=num_samples)

    return x_new


def envbo_experiment(func, noise, bounds, random_walk, env_dims, num_starts, num_samples, evals, run, filename, stepsize, starts=None):

    # get number of parameters
    dims = bounds.size((1))

    # make training data
    if isinstance(starts, torch.Tensor):
        x_train = starts[:, :-1]
        y_train = starts[:, -1]
        # print("Used starts.")
    elif isinstance(starts, np.ndarray):
        x_train = torch.from_numpy(starts[:, :-1])
        y_train = torch.from_numpy(starts[:, -1])
        # print("Used starts.")

    # gather all algorithm parameters
    params = {"method": "LogEI",
              "noise": noise,
              "bounds": bounds.tolist(),
              "random_walk": random_walk.flatten().tolist(),
              "env_dims": env_dims,
              "num_starts": num_starts,
              "num_samples": num_samples,
              "evaluations": evals,
              "run": run,
              "filename": filename,
              "stepsize": stepsize,
              "starts": starts.flatten().tolist()}
    
    # save parameters
    with open(f"{filename}_params_run{run+1}.txt", "w") as convert_file:
        convert_file.write(json.dumps(params, indent=4))


    # Bayesian optimisation loop
    for iter in range(evals-1):

        # compute next random walk step
        env_values = update_env(x=x_train[-1, :],
                               bounds=bounds,
                               stepsize=stepsize,
                               random_walk=random_walk[iter, :],
                               env_dims=env_dims)
        
        # compute next candidate point
        x_new = envbo_step(x_train=x_train,
                           y_train=y_train,
                           bounds=bounds,
                           env_values=env_values.tolist(),
                           env_dims=env_dims,
                           num_starts=num_starts,
                           num_samples=num_samples)

        # evaluate new point
        y_new = func(x_new) + torch.normal(mean=0.0, std=noise, size=(1,))

        # add to training data
        x_train = torch.vstack((x_train, x_new))
        y_train = torch.hstack((y_train, y_new))

        # save results
        np.savetxt(f"{filename}_run{run+1}.csv",
                   torch.hstack([x_train, y_train.reshape(-1, 1)]).numpy(),
                   delimiter=",",
                   header=make_header(dims, env_dims),
                   comments="")
    